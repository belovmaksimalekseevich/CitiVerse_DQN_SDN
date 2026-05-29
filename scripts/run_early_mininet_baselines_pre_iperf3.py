#!/usr/bin/env python3
"""
Standalone early Mininet baseline evaluation.
Runs in parallel with DQN Phase 1 (SimEnv) — no conflict.
Starts Mininet + Ryu + tc netem, measures real RTT for all 5 static
baselines via OFPEchoRequest (real propagation through tc netem),
saves results/mininet_baselines_early.json, then cleans up.
"""
import os, sys, time, json, subprocess, logging
sys.path.insert(0, '/home/maksim/dqn_simenv_mininet')
os.chdir('/home/maksim/dqn_simenv_mininet')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
LOG = logging.getLogger('early_baselines')

import numpy as np
from topology.topology_data import SWITCHES, N_SWITCHES, LATENCY_MATRIX
from ryu_apps.multi_controller import CTRL_PORTS
from scripts.setup_tc_delays import setup_loopback_delays, cleanup_tc_delays
from baselines.run_baselines import (
    baseline_all_to_ctrl0, baseline_zone_optimal,
    baseline_load_balanced, baseline_kmeans, baseline_random,
)

RESULTS_DIR = 'results'
N_SAMPLES   = 30
INTERVAL    = 0.5   # sec between RTT samples
SETTLE_SECS = 4.0   # wait after ovs-vsctl before measuring
ECHO_WARMUP = 7.0   # wait for OFPEchoReply data to populate


def apply_assignment(assignments):
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


def analytical_icd(assignments):
    return sum(float(LATENCY_MATRIX[i][assignments[i]])
               for i in range(N_SWITCHES)) / N_SWITCHES


def measure_icd(monitor, assignments, n=N_SAMPLES, interval=INTERVAL):
    """
    Collect ICD samples from real OFPEchoRequest RTT per switch.
    RTT/2 = one-way propagation delay through tc netem + OS network stack.
    Fallback to LATENCY_MATRIX if echo data unavailable.
    """
    samples = []
    for _ in range(n):
        icd = None
        if monitor is not None:
            echo_rtts = monitor.get_echo_rtt_ms_per_switch()
            if len(echo_rtts) >= N_SWITCHES // 2:
                total = 0.0
                for i in range(N_SWITCHES):
                    dpid = SWITCHES[i]
                    rtt = echo_rtts.get(dpid)
                    total += (rtt / 2.0) if rtt is not None else float(LATENCY_MATRIX[i][assignments[i]])
                icd = total / N_SWITCHES
        if icd is None:
            icd = analytical_icd(assignments)
        samples.append(icd)
        time.sleep(interval)
    return samples


def run():
    from ryu_apps.multi_controller import MultiControllerManager
    from ryu_apps.ctrl_monitor import ControllerMonitor
    from mininet.net import Mininet
    from mininet.log import setLogLevel
    from topology.citiverse_topo import CitiverseTopo
    from topology.topology_data import get_zone_optimal_array

    setLogLevel('warning')
    net = None
    mgr = None
    monitor = None
    iperf_procs = []

    try:
        # ── Ryu controllers ────────────────────────────────────────────────
        LOG.info('Starting 5 Ryu controllers (ports 6653-6657)...')
        mgr = MultiControllerManager()
        mgr.start_all()
        time.sleep(4)

        # ── Mininet ────────────────────────────────────────────────────────
        LOG.info('Starting Mininet (CitiVerse topology)...')
        net = Mininet(topo=CitiverseTopo(), controller=None)
        net.start()
        for i in range(1, 21):
            net.get(f's{i}').cmd(
                f'ovs-vsctl set bridge s{i} fail_mode=standalone'
            )
        time.sleep(2)

        # ── Initial zone-optimal assignment so switches connect ────────────
        LOG.info('Setting initial zone-optimal assignment...')
        initial_asgn = baseline_zone_optimal()
        apply_assignment(initial_asgn)

        # ── tc netem delays for initial assignment ─────────────────────────
        LOG.info('Applying tc netem delays on loopback...')
        setup_loopback_delays(initial_asgn)
        time.sleep(1)

        # ── ControllerMonitor ──────────────────────────────────────────────
        monitor = ControllerMonitor()
        monitor.start()

        # ── Background traffic to generate PACKET_IN events ───────────────
        LOG.info('Starting background iperf3 traffic...')
        try:
            h1  = net.get('h1')
            h10 = net.get('h10')
            srv = h10.popen('iperf3 -s -p 5201 -D')
            iperf_procs.append(srv)
            time.sleep(0.5)
            cli = h1.popen(f'iperf3 -c {h10.IP()} -p 5201 -u -b 10M -t 900')
            iperf_procs.append(cli)
        except Exception as e:
            LOG.warning(f'iperf3 failed (non-fatal, analytical fallback active): {e}')

        # Wait for OFPEchoReply data to populate (ctrl_app sends echo every 2s,
        # starts after 5s warmup → first data arrives ~7s after Mininet start)
        LOG.info(f'Waiting {ECHO_WARMUP}s for OFPEchoReply RTT data...')
        time.sleep(ECHO_WARMUP)

        # ── Measure each baseline ──────────────────────────────────────────
        baselines = {
            'ZoneOptimal':  baseline_zone_optimal(),
            'LoadBalanced': baseline_load_balanced(),
            'KMeans':       baseline_kmeans(),
            'AllToCtrl0':   baseline_all_to_ctrl0(),
            'Random':       baseline_random(),
        }

        results = {}
        for name, asgn in baselines.items():
            LOG.info(f'[{name}] applying assignment...')
            apply_assignment(asgn)

            # Update tc netem to reflect new assignment's per-controller delays
            setup_loopback_delays(asgn)

            LOG.info(f'[{name}] settling {SETTLE_SECS}s (tc netem + OFPEchoReply update)...')
            time.sleep(SETTLE_SECS)

            LOG.info(f'[{name}] measuring ({N_SAMPLES} samples × {INTERVAL}s)...')
            samples = measure_icd(monitor, asgn)

            icd_mean = float(np.mean(samples))
            icd_std  = float(np.std(samples))
            analytical = analytical_icd(asgn)
            used_echo = any(abs(s - analytical) > 0.01 for s in samples)

            results[name] = {
                'icd_mean_mininet':  round(icd_mean, 3),
                'icd_std_mininet':   round(icd_std, 3),
                'icd_analytical':    round(analytical, 3),
                'n_samples':         len(samples),
                'source':            'echo_rtt' if used_echo else 'analytical_fallback',
            }
            LOG.info(
                f'  [{name}] ICD = {icd_mean:.2f} ± {icd_std:.2f} ms '
                f'(analytical: {analytical:.2f} ms, src={results[name]["source"]})'
            )

        # ── Save ───────────────────────────────────────────────────────────
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = f'{RESULTS_DIR}/mininet_baselines_early.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        LOG.info(f'Results saved → {path}')

        print(f"\n{'Method':<16} {'ICD Mininet':>12} {'± std':>8} {'Analytical':>12} {'Source'}")
        print('-' * 62)
        for name, r in results.items():
            print(f"{name:<16} {r['icd_mean_mininet']:>12.2f} "
                  f"{r['icd_std_mininet']:>8.2f} "
                  f"{r['icd_analytical']:>12.2f}   {r['source']}")

    finally:
        LOG.info('Cleaning up...')
        for p in iperf_procs:
            try: p.terminate()
            except Exception: pass
        if monitor:
            try: monitor.stop()
            except Exception: pass
        if net:
            try: net.stop()
            except Exception: pass
        if mgr:
            try: mgr.stop_all()
            except Exception: pass
        subprocess.run(['mn', '--clean'], capture_output=True)
        try:
            cleanup_tc_delays()
        except Exception: pass
        LOG.info('Cleanup complete — Mininet fully stopped')


if __name__ == '__main__':
    if os.geteuid() != 0:
        print('Run as root: sudo python3 scripts/run_early_mininet_baselines.py')
        sys.exit(1)
    run()
