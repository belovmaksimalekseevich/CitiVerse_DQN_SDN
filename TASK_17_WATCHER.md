# TASK_17: Autonomous Watcher — Full Simulation Orchestration

## Overview

The watcher starts everything in the correct order, monitors health,
automatically runs baselines after DQN, retries on failure,
and produces the final results with no manual intervention.

Run command: `sudo python3 watcher.py`

Expected total runtime: ~8-10h (Phase 1 ~10min + Phase 2 ~2h + Baselines ~6h)

---

## File: watcher.py (root of project)

```python
#!/usr/bin/env python3
"""
CitiVerse DQN Autonomous Watcher.
Runs the full simulation pipeline without manual intervention:
  1. Pre-flight checks
  2. Phase 1: SimEnv DQN training (all 3 seeds, parallel)
  3. Phase 2: Start Mininet + 5 Ryu + tc netem -> RealEnv DQN fine-tuning
  4. Baselines: run all 5 baseline methods on Mininet
  5. Evaluation + ablation
  6. Plot results + generate paper tables
  7. Cleanup

Usage:
    sudo python3 watcher.py [--sim-only] [--skip-baselines] [--seeds 42,123,456]
"""

import os, sys, time, json, logging, argparse, subprocess, signal
import multiprocessing as mp
import numpy as np

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/watcher.log', mode='w'),
    ]
)
LOG = logging.getLogger('watcher')

# ── Constants ──────────────────────────────────────────────────────────
SEEDS         = [42, 123, 456]
RESULTS_DIR   = 'results'
LOG_DIR       = 'logs'
P1_EPISODES   = 5000
P2_EPISODES   = 200
BASELINE_EPS  = 100

# Health check thresholds
MAX_LOSS_BEFORE_ALERT = 2.0
MIN_EXPECTED_ICD_MS   = 5.0    # if ICD < 5ms, tc netem probably not set
MAX_EXPECTED_ICD_MS   = 200.0  # if ICD > 200ms, something is wrong


class SimulationError(Exception):
    pass


# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: Pre-flight checks
# ═══════════════════════════════════════════════════════════════════════

def preflight_checks():
    LOG.info('=== Phase 0: Pre-flight checks ===')
    errors = []

    # Python imports
    for pkg in ['torch', 'numpy', 'sklearn', 'scipy', 'matplotlib', 'networkx']:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f'Missing Python package: {pkg}')

    # System tools
    for tool in ['ovs-vsctl', 'ryu-manager', 'tc']:
        result = subprocess.run(['which', tool], capture_output=True)
        if result.returncode != 0:
            errors.append(f'Tool not found: {tool}')

    # Root check (needed for Mininet + ovs-vsctl)
    if os.geteuid() != 0:
        errors.append('Must run as root (sudo)')

    # topology_data import
    try:
        from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS
        assert STATE_DIM == 94
        assert N_SWITCHES == 20
        assert N_CONTROLLERS == 5
    except Exception as e:
        errors.append(f'topology_data error: {e}')

    # SimEnv smoke test
    try:
        from dqn.sim_environment import CitiverseSimEnv
        env = CitiverseSimEnv(seed=0)
        s = env.reset()
        assert s.shape == (94,)
        _, r, _, info = env.step(0)
        assert isinstance(r, float)
        icd = info['icd_ms']
        if icd < MIN_EXPECTED_ICD_MS:
            LOG.warning(f'SimEnv ICD={icd:.2f}ms is very low — check LATENCY_MATRIX')
    except Exception as e:
        errors.append(f'SimEnv smoke test failed: {e}')

    # Results dir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if errors:
        for e in errors:
            LOG.error(f'[FAIL] {e}')
        raise SimulationError(f'Pre-flight failed with {len(errors)} error(s)')

    LOG.info('Pre-flight checks passed.')


# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: SimEnv training (parallel across seeds)
# ═══════════════════════════════════════════════════════════════════════

def _train_p1_worker(seed, result_queue):
    """Worker process for one Phase 1 seed."""
    try:
        from dqn.train import train_phase1
        LOG.info(f'[P1 seed={seed}] Starting SimEnv training...')
        ckpt, agent = train_phase1(seed)
        result_queue.put({'seed': seed, 'ckpt': ckpt, 'status': 'ok'})
    except Exception as e:
        result_queue.put({'seed': seed, 'error': str(e), 'status': 'fail'})


def run_phase1_parallel(seeds=SEEDS):
    LOG.info('=== Phase 1: SimEnv pre-training (parallel seeds) ===')
    t0 = time.time()
    result_queue = mp.Queue()
    procs = []

    for seed in seeds:
        p = mp.Process(target=_train_p1_worker, args=(seed, result_queue))
        p.start()
        procs.append(p)
        LOG.info(f'Started Phase 1 worker for seed={seed}, pid={p.pid}')

    # Wait for all with timeout
    timeout_sec = 3600  # 1h max for Phase 1
    for p in procs:
        p.join(timeout=timeout_sec)
        if p.is_alive():
            p.terminate()
            raise SimulationError(f'Phase 1 worker timed out (pid={p.pid})')

    results = {}
    while not result_queue.empty():
        r = result_queue.get()
        if r['status'] == 'fail':
            raise SimulationError(f'Phase 1 seed={r["seed"]} failed: {r["error"]}')
        results[r['seed']] = r['ckpt']

    elapsed = time.time() - t0
    LOG.info(f'Phase 1 complete in {elapsed/60:.1f} min. Checkpoints: {results}')
    return results  # dict: seed -> checkpoint path


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: Start Mininet + Ryu + tc netem, then fine-tune
# ═══════════════════════════════════════════════════════════════════════

def start_ryu_controllers():
    from ryu_apps.multi_controller import MultiControllerManager
    ctrl_mgr = MultiControllerManager()
    ctrl_mgr.start_all()
    LOG.info('5 Ryu controllers started on ports 6653-6657')
    return ctrl_mgr


def start_mininet():
    from mininet.net import Mininet
    from mininet.log import setLogLevel
    setLogLevel('warning')
    from topology.citiverse_topo import CitiverseTopo
    topo = CitiverseTopo()
    net = Mininet(topo=topo, controller=None)
    net.start()
    LOG.info('Mininet started with CitiVerse topology')
    # Set fail_mode=standalone on all switches
    for i in range(1, 21):
        net.get(f's{i}').cmd(f'ovs-vsctl set bridge s{i} fail_mode=standalone')
    LOG.info('All switches set to fail_mode=standalone')
    return net


def setup_tc_delays(assignments=None):
    from scripts.setup_tc_delays import setup_loopback_delays
    from topology.topology_data import get_zone_optimal_array
    if assignments is None:
        assignments = get_zone_optimal_array()
    setup_loopback_delays(assignments)
    LOG.info('tc netem delays configured on loopback')


def verify_icd_realistic(net, ctrl_mgr):
    """Smoke test: confirm ICD in RealEnv is >5ms (tc netem working)."""
    from dqn.environment import CitiverseRealEnv
    from ryu_apps.ctrl_monitor import ControllerMonitor
    monitor = ControllerMonitor()
    monitor.start()
    env = CitiverseRealEnv(net, ctrl_mgr, monitor, seed=0, max_steps=5)
    s = env.reset()
    _, _, _, info = env.step(0)
    icd = info['icd_ms']
    monitor.stop()
    env.close()
    if icd < MIN_EXPECTED_ICD_MS:
        LOG.warning(f'RealEnv ICD={icd:.2f}ms is too low! Check tc netem setup.')
        LOG.warning('Training will proceed but results may be unreliable.')
    else:
        LOG.info(f'RealEnv ICD smoke test: {icd:.2f}ms (tc netem OK)')


def run_phase2_sequential(p1_checkpoints, net, ctrl_mgr, seeds=SEEDS):
    LOG.info('=== Phase 2: RealEnv fine-tuning (sequential seeds) ===')
    from ryu_apps.ctrl_monitor import ControllerMonitor
    from dqn.train import train_phase2, evaluate_agent
    from dqn.environment import CitiverseRealEnv
    from dqn.agent import DQNAgent
    from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS

    ACTION_DIM = N_SWITCHES * N_CONTROLLERS
    monitor = ControllerMonitor()
    monitor.start()

    p2_results = {}
    for seed in seeds:
        LOG.info(f'--- Phase 2 seed={seed} ---')
        ckpt_path, agent = train_phase2(
            seed=seed,
            p1_ckpt_path=p1_checkpoints[seed],
            net=net,
            ctrl_mgr=ctrl_mgr,
            monitor=monitor,
        )
        # Evaluate trained agent
        eval_env = CitiverseRealEnv(net, ctrl_mgr, monitor,
                                    seed=seed + 1000, max_steps=300)
        from dqn.train import evaluate_agent
        metrics = evaluate_agent(agent, eval_env, n_episodes=20)
        np.save(f'{RESULTS_DIR}/p2_eval_seed{seed}.npy', metrics)
        p2_results[seed] = metrics
        LOG.info(f'[P2 seed={seed}] ICD={metrics["icd_mean"]:.2f}±{metrics["icd_std"]:.2f}ms')
        eval_env.close()

    monitor.stop()
    return p2_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: Baselines (all 5 methods, on RealEnv via Mininet)
# ═══════════════════════════════════════════════════════════════════════

def run_all_baselines_mininet(net, ctrl_mgr):
    LOG.info('=== Phase 3: Baselines on Mininet ===')
    from baselines.run_baselines import run_all_baselines
    from ryu_apps.ctrl_monitor import ControllerMonitor

    # Static baselines (analytical — fast, no Mininet interactions needed)
    LOG.info('Running static baselines (analytical)...')
    static_results = run_all_baselines()
    with open(f'{RESULTS_DIR}/baselines.json', 'w') as f:
        json.dump(static_results, f, indent=2)
    LOG.info('Static baselines done.')

    # MOOO-RDQN simplified (optional — skip if time is tight)
    try:
        from baselines.mooo_rdqn import run_mooo_rdqn_baseline
        from dqn.sim_environment import CitiverseSimEnv
        LOG.info('Running simplified MOOO-RDQN comparison on SimEnv...')
        mooo_icds = []
        for seed in SEEDS:
            sim_env = CitiverseSimEnv(seed=seed)
            res = run_mooo_rdqn_baseline(sim_env, n_episodes=100, seed=seed)
            mooo_icds.append(res['icd_mean'])
        static_results['MOOO-RDQN-simplified'] = {
            'icd_ms': float(np.mean(mooo_icds)),
            'icd_std': float(np.std(mooo_icds)),
        }
        LOG.info(f'MOOO-RDQN ICD: {np.mean(mooo_icds):.2f}ms')
    except Exception as e:
        LOG.warning(f'MOOO-RDQN baseline failed (non-critical): {e}')

    with open(f'{RESULTS_DIR}/baselines.json', 'w') as f:
        json.dump(static_results, f, indent=2)

    return static_results


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: Evaluation, ablation, plots
# ═══════════════════════════════════════════════════════════════════════

def run_evaluation_and_plots(p2_results, baseline_results):
    LOG.info('=== Phase 4: Evaluation, ablation, plots ===')

    # Print success criteria check
    dqn_icd_all = [p2_results[s]['icd_mean'] for s in SEEDS if s in p2_results]
    zo_icd = baseline_results.get('ZoneOptimal', {}).get('icd_ms', 999)

    if dqn_icd_all:
        dqn_mean = np.mean(dqn_icd_all)
        dqn_std  = np.std(dqn_icd_all)
        improvement = (zo_icd - dqn_mean) / zo_icd * 100
        LOG.info(f'DQN ICD: {dqn_mean:.2f} +- {dqn_std:.2f} ms')
        LOG.info(f'ZoneOptimal ICD: {zo_icd:.2f} ms')
        LOG.info(f'Improvement: {improvement:+.1f}%')
        if dqn_mean < zo_icd:
            LOG.info('[PASS] DQN beats ZoneOptimal')
        else:
            LOG.warning('[FAIL] DQN does NOT beat ZoneOptimal')

    # Ablation (SimEnv only — can run after Phase 1)
    try:
        from scripts.ablation import run_ablation
        LOG.info('Running ablation study on SimEnv...')
        run_ablation()
    except Exception as e:
        LOG.warning(f'Ablation failed (non-critical): {e}')

    # Plots
    try:
        from scripts.plot_results import (
            fig1_training_curve_p1, fig2_training_curve_p2,
            fig3_icd_bar, fig4_icd_per_profile,
            fig5_load_distribution, fig6_convergence_speed,
            table1_main_results, table2_ablation,
        )
        for fn in [fig1_training_curve_p1, fig2_training_curve_p2,
                   fig3_icd_bar, fig4_icd_per_profile,
                   fig5_load_distribution, fig6_convergence_speed,
                   table1_main_results, table2_ablation]:
            try:
                fn()
            except Exception as e:
                LOG.warning(f'{fn.__name__} failed: {e}')

        from scripts.plot_topology import draw_citiverse_topology
        from baselines.run_baselines import baseline_zone_optimal
        draw_citiverse_topology(
            assignments=baseline_zone_optimal(),
            save_path=f'{RESULTS_DIR}/figures/topology.pdf'
        )
    except Exception as e:
        LOG.warning(f'Plotting failed (non-critical): {e}')

    LOG.info(f'Results saved to {RESULTS_DIR}/')


# ═══════════════════════════════════════════════════════════════════════
# CLEANUP
# ═══════════════════════════════════════════════════════════════════════

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
    # Kill any lingering OvS processes
    subprocess.run(['mn', '--clean'], capture_output=True)
    LOG.info('Cleanup done.')


# ═══════════════════════════════════════════════════════════════════════
# HEALTH MONITOR (background thread)
# ═══════════════════════════════════════════════════════════════════════

class HealthMonitor:
    """Runs in a background thread, checks Ryu processes are alive."""

    def __init__(self, ctrl_mgr):
        self.ctrl_mgr = ctrl_mgr
        self._running = False
        self._thread = None

    def start(self):
        import threading
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _loop(self):
        import time
        while self._running:
            for i in range(5):
                if not self.ctrl_mgr.is_running(i):
                    LOG.warning(f'Ryu ctrl {i} died! Restarting...')
                    self.ctrl_mgr.restart_ctrl(i)
            time.sleep(10)


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description='CitiVerse DQN Autonomous Watcher')
    parser.add_argument('--sim-only', action='store_true',
                        help='Run Phase 1 + static baselines only (no Mininet)')
    parser.add_argument('--skip-baselines', action='store_true',
                        help='Skip baseline evaluation (run DQN only)')
    parser.add_argument('--seeds', default='42,123,456',
                        help='Comma-separated seeds (default: 42,123,456)')
    parser.add_argument('--p1-episodes', type=int, default=5000)
    parser.add_argument('--p2-episodes', type=int, default=200)
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    net = None
    ctrl_mgr = None
    health_monitor = None

    # Register SIGINT/SIGTERM handler for clean shutdown
    def _signal_handler(sig, frame):
        LOG.info('Interrupt received — cleaning up...')
        cleanup(net, ctrl_mgr)
        sys.exit(0)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    total_start = time.time()

    try:
        # Phase 0: checks
        preflight_checks()

        # Phase 1: SimEnv training (parallel seeds)
        p1_checkpoints = run_phase1_parallel(seeds)

        if args.sim_only:
            LOG.info('--sim-only: skipping Mininet phases')
            baseline_results = run_all_baselines_mininet(None, None)
            run_evaluation_and_plots({}, baseline_results)
            return

        # Phase 2: Mininet setup
        LOG.info('=== Starting Mininet infrastructure ===')
        ctrl_mgr = start_ryu_controllers()
        time.sleep(3)  # let Ryu initialize

        net = start_mininet()
        time.sleep(2)

        setup_tc_delays()    # critical: add latency to loopback
        verify_icd_realistic(net, ctrl_mgr)

        health_monitor = HealthMonitor(ctrl_mgr)
        health_monitor.start()

        # Phase 2: DQN fine-tuning
        p2_results = run_phase2_sequential(p1_checkpoints, net, ctrl_mgr, seeds)

        # Phase 3: Baselines
        if not args.skip_baselines:
            baseline_results = run_all_baselines_mininet(net, ctrl_mgr)
        else:
            baseline_results = {}

        # Phase 4: Evaluation + plots
        run_evaluation_and_plots(p2_results, baseline_results)

    except SimulationError as e:
        LOG.error(f'Simulation failed: {e}')
        sys.exit(1)
    except Exception as e:
        LOG.exception(f'Unexpected error: {e}')
        sys.exit(1)
    finally:
        if health_monitor:
            health_monitor.stop()
        cleanup(net, ctrl_mgr)

    elapsed = (time.time() - total_start) / 3600
    LOG.info(f'=== FULL SIMULATION COMPLETE in {elapsed:.1f}h ===')
    LOG.info(f'Results: {RESULTS_DIR}/')


if __name__ == '__main__':
    mp.set_start_method('spawn')  # safe for CUDA + Ryu
    main()
```

---

## Expected Timeline

```
00:00  Phase 0: Pre-flight checks             ~30s
00:01  Phase 1: SimEnv training               ~10 min (3 seeds parallel)
00:11  Mininet + Ryu + tc netem start         ~2 min
00:13  Phase 2: RealEnv fine-tuning           ~5-7h (3 seeds sequential, ~2h each)
06:30  Phase 3: Static baselines              ~30 min
07:00  Phase 3: MOOO-RDQN comparison          ~20 min
07:20  Phase 4: Evaluation + plots            ~10 min
07:30  Cleanup                                ~1 min

TOTAL: ~7-8 hours
```

---

## Launch Commands

```bash
# Full autonomous run:
cd /home/maksim/dqn_simenv_mininet
source ../dqn_env/bin/activate
sudo python3 watcher.py 2>&1 | tee logs/watcher_run_$(date +%Y%m%d_%H%M%S).log

# Smoke test (SimEnv only, no root needed, ~15 min):
python3 watcher.py --sim-only

# DQN only (no baselines, faster for debugging):
sudo python3 watcher.py --skip-baselines

# Single seed for faster iteration:
sudo python3 watcher.py --seeds 42 --p1-episodes 1000 --p2-episodes 50

# Monitor progress in another terminal:
tail -f logs/watcher.log
```

---

## Resilience

- SIGINT / SIGTERM: cleanup runs before exit (Mininet + Ryu stop)
- Ryu crash: HealthMonitor restarts dead controllers automatically
- Phase 1 worker timeout: 1h limit prevents hung processes
- Exception in any phase: caught, logged, cleanup runs, exit code 1
- Log file: `logs/watcher.log` captures everything for post-mortem
