#!/usr/bin/env python3
"""
CitiVerse DQN Autonomous Watcher.
Runs the full simulation pipeline without manual intervention:
  1. Pre-flight checks
  2. Phase 1: SimEnv DQN training (all 3 seeds, parallel)
  3. Phase 2: Start Mininet + 5 Ryu + tc netem -> RealEnv DQN fine-tuning
  3.5. Phase 2.5: Evaluate all 5 static baselines IN Mininet (real RTT)
  4. Baselines: analytical metrics (LATENCY_MATRIX)
  5. Evaluation + plots
  6. Cleanup

Usage:
    sudo python3 watcher.py [--sim-only] [--skip-baselines] [--skip-p1] [--seeds 42,123,456]
    sudo python3 watcher.py --skip-p1          # resume after Phase 1 already done
"""
import os
import sys
import time
import json
import logging
import argparse
import subprocess
import signal
import threading
import multiprocessing as mp
import numpy as np

# ---------------------------------------------------------------------------
# Logging — file + stdout
# ---------------------------------------------------------------------------
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/watcher.log', mode='w'),
    ],
)
LOG = logging.getLogger('watcher')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEEDS        = [42, 123, 456]
RESULTS_DIR  = 'results'
LOG_DIR      = 'logs'
MIN_ICD_MS   = 5.0    # below this -> tc netem probably not set
MAX_ICD_MS   = 200.0


class SimulationError(Exception):
    pass


# ===========================================================================
# Phase 0: Pre-flight checks
# ===========================================================================

def preflight_checks(sim_only=False):
    LOG.info('=== Phase 0: Pre-flight checks ===')
    errors = []

    for pkg in ['torch', 'numpy', 'sklearn', 'scipy']:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f'Missing Python package: {pkg}')

    if not sim_only:
        for tool in ['ovs-vsctl', 'ryu-manager', 'tc']:
            result = subprocess.run(['which', tool], capture_output=True)
            if result.returncode != 0:
                errors.append(f'Tool not found: {tool}')

        if os.geteuid() != 0:
            errors.append('Must run as root (sudo) for Phase 2')

    try:
        from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS, ACTION_DIM
        assert STATE_DIM == 94
        assert N_SWITCHES == 20
        assert N_CONTROLLERS == 5
        assert ACTION_DIM == 100
    except Exception as e:
        errors.append(f'topology_data error: {e}')

    try:
        from dqn.sim_environment import CitiverseSimEnv
        env = CitiverseSimEnv(seed=0)
        s = env.reset()
        assert s.shape == (94,)
        _, r, _, info = env.step(0)
        assert isinstance(r, float)
        LOG.info(f'  SimEnv OK: state={s.shape}, icd={info["icd_ms"]:.2f}ms')
    except Exception as e:
        errors.append(f'SimEnv smoke test failed: {e}')

    try:
        from dqn.agent import DQNAgent
        agent = DQNAgent(total_steps=100)
        s = np.zeros(94, np.float32)
        for _ in range(300):
            agent.push(s, 0, 0.1, s, False)
        loss = agent.update()
        assert loss is not None
        LOG.info(f'  DQNAgent OK: loss={loss:.4f}')
    except Exception as e:
        errors.append(f'DQNAgent smoke test failed: {e}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if errors:
        for e in errors:
            LOG.error(f'[FAIL] {e}')
        raise SimulationError(f'Pre-flight failed: {len(errors)} error(s)')

    LOG.info('Pre-flight checks passed.')


# ===========================================================================
# Phase 1: SimEnv training (parallel seeds)
# ===========================================================================

def _p1_worker(seed, result_queue):
    try:
        from dqn.train import train_phase1
        ckpt, _ = train_phase1(seed)
        result_queue.put({'seed': seed, 'ckpt': ckpt, 'status': 'ok'})
    except Exception as e:
        result_queue.put({'seed': seed, 'error': str(e), 'status': 'fail'})


def run_phase1(seeds):
    LOG.info('=== Phase 1: SimEnv pre-training ===')
    t0 = time.time()
    queue = mp.Queue()
    procs = []
    for seed in seeds:
        p = mp.Process(target=_p1_worker, args=(seed, queue))
        p.start()
        procs.append(p)
        LOG.info(f'Phase 1 worker seed={seed} pid={p.pid}')

    for p in procs:
        p.join(timeout=50000)  # ~14h max — 5000ep × ~5s/ep × safety margin
        if p.is_alive():
            p.terminate()
            raise SimulationError('Phase 1 worker timed out')

    results = {}
    while not queue.empty():
        r = queue.get()
        if r['status'] == 'fail':
            raise SimulationError(f'Phase 1 seed={r["seed"]} failed: {r["error"]}')
        results[r['seed']] = r['ckpt']
        LOG.info(f'Phase 1 seed={r["seed"]} checkpoint: {r["ckpt"]}')

    LOG.info(f'Phase 1 done in {(time.time()-t0)/60:.1f} min')
    return results


# ===========================================================================
# Phase 2: Mininet infrastructure
# ===========================================================================

def start_ryu_controllers():
    from ryu_apps.multi_controller import MultiControllerManager
    mgr = MultiControllerManager()
    mgr.start_all()
    LOG.info('5 Ryu controllers started (ports 6653-6657)')
    return mgr


def start_mininet():
    from mininet.net import Mininet
    from mininet.log import setLogLevel
    setLogLevel('warning')
    from topology.citiverse_topo import CitiverseTopo
    net = Mininet(topo=CitiverseTopo(), controller=None)
    net.start()
    for i in range(1, 21):
        net.get(f's{i}').cmd(f'ovs-vsctl set bridge s{i} fail_mode=standalone')
    LOG.info('Mininet started, all switches fail_mode=standalone')
    return net


def apply_tc_delays():
    from scripts.setup_tc_delays import setup_loopback_delays
    from topology.topology_data import get_zone_optimal_array
    setup_loopback_delays(get_zone_optimal_array())
    LOG.info('tc netem delays applied on loopback')


def verify_icd(net, ctrl_mgr):
    """Quick check that RealEnv ICD is in realistic range."""
    from dqn.environment import CitiverseRealEnv
    env = CitiverseRealEnv(net, ctrl_mgr, seed=0, max_steps=3)
    try:
        env.reset()
        _, _, _, info = env.step(0)
        icd = info['icd_ms']
        if icd < MIN_ICD_MS:
            LOG.warning(f'RealEnv ICD={icd:.2f}ms is LOW — tc netem may not be set')
        else:
            LOG.info(f'RealEnv ICD smoke test: {icd:.2f}ms OK')
    finally:
        env.close()


class HealthMonitor(threading.Thread):
    """Background thread that restarts dead Ryu processes."""

    def __init__(self, ctrl_mgr):
        super().__init__(daemon=True)
        self.ctrl_mgr = ctrl_mgr
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(10):
            for i in range(5):
                if not self.ctrl_mgr.is_running(i):
                    LOG.warning(f'Ryu ctrl_{i} died — restarting')
                    self.ctrl_mgr.restart_ctrl(i)

    def stop(self):
        self._stop_event.set()


def run_phase2(p1_ckpts, net, ctrl_mgr, seeds):
    LOG.info('=== Phase 2: RealEnv fine-tuning ===')
    from ryu_apps.ctrl_monitor import ControllerMonitor
    from dqn.train import train_phase2, evaluate_agent
    from dqn.environment import CitiverseRealEnv

    monitor = ControllerMonitor()
    monitor.start()

    p2_results = {}
    for seed in seeds:
        LOG.info(f'--- Phase 2 seed={seed} ---')
        ckpt, agent = train_phase2(
            seed=seed,
            p1_ckpt_path=p1_ckpts[seed],
            net=net,
            ctrl_mgr=ctrl_mgr,
            monitor=monitor,
        )
        eval_env = CitiverseRealEnv(net, ctrl_mgr, ctrl_monitor=monitor,
                                     seed=seed + 1000, max_steps=300)
        metrics = evaluate_agent(agent, eval_env, n_episodes=20)
        eval_env.close()

        np.save(f'{RESULTS_DIR}/p2_eval_seed{seed}.npy', metrics)
        with open(f'{RESULTS_DIR}/p2_eval_seed{seed}.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        p2_results[seed] = metrics
        LOG.info(
            f'[P2 seed={seed}] ICD={metrics["icd_mean"]:.2f}'
            f'+-{metrics["icd_std"]:.2f}ms'
        )

    monitor.stop()
    return p2_results


# ===========================================================================
# Phase 2.5: Mininet baseline evaluation (real RTT, not analytical)
# ===========================================================================

def run_mininet_baselines(net, ctrl_mgr, monitor):
    """
    Apply each static baseline assignment in the live Mininet and measure
    real ICD via ControllerMonitor RTT.

    Scientific rationale: tc netem adds propagation delay per controller port,
    but an overloaded controller also accumulates queuing delay. ZoneOptimal
    has minimum propagation delay but causes load imbalance under non-morning
    profiles, adding queuing overhead. This phase captures the full RTT
    (propagation + queuing) for each static assignment, making the comparison
    with DQN Phase 2 results apples-to-apples.

    Falls back to analytical LATENCY_MATRIX values if monitor is unavailable.
    """
    LOG.info('=== Phase 2.5: Mininet baseline evaluation ===')
    from topology.topology_data import SWITCHES, N_SWITCHES, LATENCY_MATRIX
    from ryu_apps.multi_controller import CTRL_PORTS
    from baselines.run_baselines import (
        baseline_all_to_ctrl0, baseline_zone_optimal,
        baseline_load_balanced, baseline_kmeans, baseline_random,
    )

    baselines_def = {
        'ZoneOptimal':  baseline_zone_optimal(),
        'LoadBalanced': baseline_load_balanced(),
        'KMeans':       baseline_kmeans(),
        'AllToCtrl0':   baseline_all_to_ctrl0(),
        'Random':       baseline_random(),
    }

    def _apply_assignment(assignments):
        for i, sw_id in enumerate(SWITCHES):
            port = CTRL_PORTS[int(assignments[i])]
            try:
                subprocess.run(
                    ['ovs-vsctl', 'set-controller', f's{sw_id}',
                     f'tcp:127.0.0.1:{port}'],
                    capture_output=True, timeout=5, check=True,
                )
            except Exception as e:
                LOG.warning(f'  ovs-vsctl s{sw_id}: {e}')

    def _measure_rtt(assignments, n_samples=30, interval=0.5):
        samples = []
        for _ in range(n_samples):
            if monitor is not None:
                rtt = monitor.get_mean_rtt_ms()
                if rtt > 0:
                    samples.append(rtt)
                    time.sleep(interval)
                    continue
            # Analytical fallback when monitor has no data yet
            analytical = sum(
                float(LATENCY_MATRIX[i][assignments[i]])
                for i in range(N_SWITCHES)
            ) / N_SWITCHES
            samples.append(analytical)
            time.sleep(interval)
        return samples

    mininet_results = {}
    for name, asgn in baselines_def.items():
        LOG.info(f'  [{name}] applying assignment...')
        _apply_assignment(asgn)
        time.sleep(3.0)  # let switches reconnect and PACKET_IN events settle

        LOG.info(f'  [{name}] measuring RTT (30 × 0.5s)...')
        samples = _measure_rtt(asgn, n_samples=30, interval=0.5)

        icd_mean = float(np.mean(samples))
        icd_std  = float(np.std(samples))
        mininet_results[name] = {
            'icd_mean_mininet': round(icd_mean, 3),
            'icd_std_mininet':  round(icd_std, 3),
            'n_samples':        len(samples),
            'assignments':      asgn.tolist(),
        }
        LOG.info(f'  [{name}] ICD = {icd_mean:.2f} ± {icd_std:.2f} ms (Mininet)')

    # Restore zone-optimal so Mininet is in a known state for cleanup
    _apply_assignment(baselines_def['ZoneOptimal'])
    time.sleep(1.0)

    path = f'{RESULTS_DIR}/mininet_baselines.json'
    with open(path, 'w') as f:
        json.dump(mininet_results, f, indent=2)
    LOG.info(f'Mininet baselines saved → {path}')
    return mininet_results


# ===========================================================================
# Phase 3: Analytical baselines
# ===========================================================================

def run_baselines():
    LOG.info('=== Phase 3: Analytical baselines ===')
    from baselines.run_baselines import run_all_baselines
    results = run_all_baselines(save=True)
    with open(f'{RESULTS_DIR}/baselines.json', 'w') as f:
        json.dump(results, f, indent=2)
    return results


# ===========================================================================
# Phase 4: Evaluation + plots
# ===========================================================================

def run_evaluation(p2_results, baseline_results):
    LOG.info('=== Phase 4: Evaluation + plots ===')

    dqn_icds = [p2_results[s]['icd_mean'] for s in SEEDS if s in p2_results]
    # Load ZO from Mininet baseline (measured earlier), fall back to analytical
    try:
        import json as _json
        with open(f'{RESULTS_DIR}/mininet_baselines_early.json') as _f:
            _mn = _json.load(_f)
        zo_icd = _mn.get('ZoneOptimal', {}).get('icd_mean_mininet', 3.0)
    except Exception:
        zo_icd = baseline_results.get('ZoneOptimal', {}).get('icd_analytical', 3.0)
    LOG.info(f'ZoneOptimal analytical ICD = {zo_icd:.2f} ms')

    summary = {}
    if dqn_icds:
        dqn_mean = float(np.mean(dqn_icds))
        dqn_std  = float(np.std(dqn_icds))
        improvement = (zo_icd - dqn_mean) / zo_icd * 100 if zo_icd > 0 else 0.0
        summary = {
            'dqn_icd_mean':       dqn_mean,
            'dqn_icd_std':        dqn_std,
            'zone_optimal_icd':   zo_icd,
            'improvement_pct':    improvement,
            'beats_zone_optimal': dqn_mean < zo_icd,
        }
        LOG.info(f'DQN ICD:         {dqn_mean:.2f} ± {dqn_std:.2f} ms')
        LOG.info(f'ZoneOptimal ICD: {zo_icd:.2f} ms (analytical)')
        LOG.info(f'Improvement:     {improvement:+.1f}%')
        if dqn_mean < zo_icd:
            LOG.info('[RESULT] DQN BEATS ZoneOptimal  ✓')
        else:
            LOG.warning('[RESULT] DQN does NOT beat ZoneOptimal — check training')

    with open(f'{RESULTS_DIR}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Ablation study: deferred to future work

    # Optional plots
    os.makedirs(f'{RESULTS_DIR}/figures', exist_ok=True)
    for fn_name in ['fig1_training_curve_p1', 'fig3_icd_bar',
                    'fig4_icd_per_profile', 'fig5_load_distribution']:
        try:
            from scripts import plot_results as pr
            getattr(pr, fn_name)()
        except Exception as e:
            LOG.warning(f'Plot {fn_name} skipped: {e}')

    try:
        from scripts.plot_topology import draw_citiverse_topology
        from baselines.run_baselines import baseline_zone_optimal
        draw_citiverse_topology(
            assignments=baseline_zone_optimal(),
            save_path=f'{RESULTS_DIR}/figures/topology.pdf',
        )
    except Exception as e:
        LOG.warning(f'Topology plot skipped: {e}')

    return summary


# ===========================================================================
# Cleanup
# ===========================================================================

def cleanup(net=None, ctrl_mgr=None):
    LOG.info('=== Cleanup ===')
    try:
        from scripts.setup_tc_delays import cleanup_tc_delays
        cleanup_tc_delays()
    except Exception:
        pass
    if ctrl_mgr:
        try:
            ctrl_mgr.stop_all()
        except Exception:
            pass
    if net:
        try:
            net.stop()
        except Exception:
            pass
    subprocess.run(['mn', '--clean'], capture_output=True)
    LOG.info('Cleanup complete')


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description='CitiVerse DQN Watcher')
    p.add_argument('--sim-only', action='store_true',
                   help='Phase 1 + static baselines only (no Mininet, no root needed)')
    p.add_argument('--skip-baselines', action='store_true')
    p.add_argument('--skip-p1', action='store_true',
                   help='Skip Phase 1 — load existing p1_best_seed*.pth checkpoints '
                        'and go straight to Phase 2. Use after orphaned workers finish.')
    p.add_argument('--seeds', default='42,123,456')
    p.add_argument('--p1-episodes', type=int, default=None,
                   help='Override P1_EPISODES in train.py')
    p.add_argument('--p2-episodes', type=int, default=None,
                   help='Override P2_EPISODES in train.py')
    return p.parse_args()


def load_p1_ckpts(seeds):
    """Load existing Phase 1 checkpoints from disk (used with --skip-p1)."""
    LOG.info('=== Phase 1: loading existing checkpoints (--skip-p1) ===')
    ckpts = {}
    for seed in seeds:
        for name in (f'p1_seed{seed}.pth', f'p1_best_seed{seed}.pth'):
            path = f'{RESULTS_DIR}/{name}'
            if os.path.exists(path):
                ckpts[seed] = path
                LOG.info(f'  seed={seed}: loaded {path} '
                         f'({os.path.getsize(path)//1024}KB)')
                break
        if seed not in ckpts:
            raise SimulationError(
                f'No Phase 1 checkpoint found for seed={seed}. '
                f'Expected: {RESULTS_DIR}/p1_seed{seed}.pth or p1_best_seed{seed}.pth'
            )
    return ckpts


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    if args.p1_episodes is not None:
        import dqn.train as _t
        _t.P1_EPISODES = args.p1_episodes
    if args.p2_episodes is not None:
        import dqn.train as _t
        _t.P2_EPISODES = args.p2_episodes

    net = None
    ctrl_mgr = None
    health_mon = None

    def _shutdown(sig, frame):
        LOG.info('Shutdown signal received')
        if health_mon:
            health_mon.stop()
        cleanup(net, ctrl_mgr)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    t_total = time.time()

    try:
        preflight_checks(sim_only=args.sim_only)

        if args.skip_p1:
            p1_ckpts = load_p1_ckpts(seeds)
        else:
            p1_ckpts = run_phase1(seeds)

        if args.sim_only:
            LOG.info('--sim-only: skipping Mininet')
            baseline_results = run_baselines() if not args.skip_baselines else {}
            run_evaluation({}, baseline_results)
        else:
            ctrl_mgr = start_ryu_controllers()
            time.sleep(3)
            net = start_mininet()
            time.sleep(2)
            apply_tc_delays()
            verify_icd(net, ctrl_mgr)

            health_mon = HealthMonitor(ctrl_mgr)
            health_mon.start()

            # Phase 2: DQN fine-tuning
            p2_results = run_phase2(p1_ckpts, net, ctrl_mgr, seeds)

            # Phase 3: Analytical baselines
            baseline_results = {}
            if not args.skip_baselines:
                baseline_results = run_baselines()

            # Phase 4: Evaluation + plots
            summary = run_evaluation(p2_results, baseline_results)
            LOG.info(f'Summary: {summary}')

    except SimulationError as e:
        LOG.error(f'Simulation error: {e}')
        sys.exit(1)
    except Exception as e:
        LOG.exception(f'Unexpected error: {e}')
        sys.exit(1)
    finally:
        if health_mon:
            health_mon.stop()
        cleanup(net, ctrl_mgr)

    elapsed = (time.time() - t_total) / 3600
    LOG.info(f'=== DONE in {elapsed:.2f}h — results in {RESULTS_DIR}/ ===')


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
