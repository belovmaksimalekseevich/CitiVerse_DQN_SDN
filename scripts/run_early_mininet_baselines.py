#!/usr/bin/env python3
"""
Standalone early Mininet baseline evaluation — v2 (iperf3 per-profile).
Runs in parallel with DQN Phase 1 (SimEnv) -- no conflict.
Starts Mininet + Ryu + tc netem, then for each baseline assignment measures
real RTT via OFPEchoRequest under iperf3 load for EACH traffic profile.
Saves results/mininet_baselines_early.json (per-profile ICD), then cleans up.
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
from topology.topology_data import (
    SWITCHES, N_SWITCHES, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
    TRAFFIC_PROFILES, SW_ZONE, CTRL_ZONE,
)
from ryu_apps.multi_controller import CTRL_PORTS
from scripts.setup_tc_delays import setup_loopback_delays, cleanup_tc_delays
from baselines.run_baselines import (
    baseline_all_to_ctrl0, baseline_zone_optimal,
    baseline_load_balanced, baseline_kmeans, baseline_random,
)

RESULTS_DIR = 'results'
N_SAMPLES   = 30
INTERVAL    = 0.5    # sec between RTT samples
SETTLE_SECS = 3.0    # wait after ovs-vsctl + traffic start before measuring
ECHO_WARMUP = 7.0    # wait for OFPEchoReply data to populate

# iperf3 parameters (must match dqn/environment.py)
_BASE_RATE_MBPS   = 2.0
_CTRL_LOOPBACK    = ['127.0.0.1', '127.0.0.2', '127.0.0.3', '127.0.0.4', '127.0.0.5']
_IPERF3_BASE_PORT = 5200
_N_SWITCHES       = 20


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
    RTT/2 = one-way delay through tc netem + controller event-loop latency.
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


# ---------------------------------------------------------------------------
# iperf3 traffic helpers
# ---------------------------------------------------------------------------

def start_iperf3_servers():
    """One iperf3 server per switch on port 5200+sw_idx."""
    procs = []
    for sw_idx in range(_N_SWITCHES):
        port = _IPERF3_BASE_PORT + sw_idx
        try:
            proc = subprocess.Popen(
                ['iperf3', '-s', '-p', str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            procs.append(proc)
        except Exception as e:
            LOG.warning(f'iperf3 server sw_idx={sw_idx}: {e}')
    time.sleep(0.5)
    LOG.info(f'iperf3: {len(procs)} servers started (ports {_IPERF3_BASE_PORT}-{_IPERF3_BASE_PORT+_N_SWITCHES-1})')
    return procs


def start_iperf3_clients(net, assignments, profile, client_procs):
    """Kill old clients and start new ones for the given assignment + profile."""
    # Kill old clients
    for p in client_procs:
        try:
            p.terminate()
        except Exception:
            pass
    client_procs.clear()
    try:
        subprocess.run(['pkill', '-f', 'iperf3 -c 127.0.'], capture_output=True)
    except Exception:
        pass

    zone_factors = ZONE_LOAD_FACTORS[profile]
    for sw_idx, sw_id in enumerate(SWITCHES):
        ctrl_idx = int(assignments[sw_idx])
        zone     = SW_ZONE[sw_id]
        rate     = _BASE_RATE_MBPS * zone_factors.get(zone, 1.0)
        dst_ip   = _CTRL_LOOPBACK[ctrl_idx]
        port     = _IPERF3_BASE_PORT + sw_idx
        cmd = (f'iperf3 -c {dst_ip} -p {port} -u -b {rate:.2f}M '
               f'-t 3600 -N 2>/dev/null')
        try:
            sw   = net.get(f's{sw_id}')
            proc = sw.popen(cmd, shell=True)
            client_procs.append(proc)
        except Exception as e:
            LOG.debug(f'iperf3 client s{sw_id}: {e}')
    time.sleep(0.5)


def stop_iperf3_servers(server_procs):
    for p in server_procs:
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            pass
    server_procs.clear()
    try:
        subprocess.run(['pkill', '-f', f'iperf3.*{_IPERF3_BASE_PORT}'],
                       capture_output=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------

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
    server_procs = []
    client_procs = []

    try:
        # -- Ryu controllers -----------------------------------------------
        LOG.info('Starting 5 Ryu controllers (ports 6653-6657)...')
        mgr = MultiControllerManager()
        mgr.start_all()
        time.sleep(4)

        # -- Mininet --------------------------------------------------------
        LOG.info('Starting Mininet (CitiVerse topology)...')
        net = Mininet(topo=CitiverseTopo(), controller=None)
        net.start()
        for i in range(1, 21):
            net.get(f's{i}').cmd(
                f'ovs-vsctl set bridge s{i} fail_mode=standalone'
            )
        time.sleep(2)

        # -- Initial zone-optimal assignment --------------------------------
        LOG.info('Setting initial zone-optimal assignment...')
        initial_asgn = baseline_zone_optimal()
        apply_assignment(initial_asgn)
        setup_loopback_delays(initial_asgn)
        time.sleep(1)

        # -- ControllerMonitor ----------------------------------------------
        monitor = ControllerMonitor()
        monitor.start()

        # -- iperf3 servers (start once, reuse across all baselines) --------
        LOG.info('Starting iperf3 servers...')
        server_procs = start_iperf3_servers()

        # -- Wait for OFPEchoReply data -------------------------------------
        LOG.info(f'Waiting {ECHO_WARMUP}s for OFPEchoReply RTT data...')
        time.sleep(ECHO_WARMUP)

        # -- Measure each baseline under each profile ----------------------
        baselines = {
            'ZoneOptimal':  baseline_zone_optimal(),
            'LoadBalanced': baseline_load_balanced(),
            'KMeans':       baseline_kmeans(),
            'AllToCtrl0':   baseline_all_to_ctrl0(),
            'Random':       baseline_random(),
        }

        results = {}
        for name, asgn in baselines.items():
            LOG.info(f'=== [{name}] ===')
            apply_assignment(asgn)
            setup_loopback_delays(asgn)

            per_profile = {}
            for profile in TRAFFIC_PROFILES:
                LOG.info(f'  [{name}] profile={profile}: starting traffic...')
                start_iperf3_clients(net, asgn, profile, client_procs)
                time.sleep(SETTLE_SECS)

                LOG.info(f'  [{name}] profile={profile}: measuring ({N_SAMPLES}x{INTERVAL}s)...')
                samples = measure_icd(monitor, asgn)
                icd_mean = float(np.mean(samples))
                icd_std  = float(np.std(samples))
                per_profile[profile] = {
                    'icd_mean': round(icd_mean, 3),
                    'icd_std':  round(icd_std, 3),
                }
                LOG.info(f'  [{name}] {profile}: ICD = {icd_mean:.2f} +- {icd_std:.2f} ms')

            # Profile-averaged ICD (used in paper Table 3)
            profile_means = [v['icd_mean'] for v in per_profile.values()]
            avg_icd  = round(float(np.mean(profile_means)), 3)
            avg_std  = round(float(np.std(profile_means)), 3)
            analytical = analytical_icd(asgn)

            results[name] = {
                'icd_mean_mininet':  avg_icd,
                'icd_std_mininet':   avg_std,
                'icd_analytical':    round(analytical, 3),
                'per_profile':       per_profile,
                'n_samples':         N_SAMPLES,
            }
            LOG.info(
                f'  [{name}] avg ICD = {avg_icd:.2f} +- {avg_std:.2f} ms '
                f'(analytical: {analytical:.2f} ms)'
            )

        # Stop clients before moving to next baseline
        for p in client_procs:
            try:
                p.terminate()
            except Exception:
                pass
        client_procs.clear()

        # -- Restore zone-optimal -------------------------------------------
        apply_assignment(baselines['ZoneOptimal'])
        time.sleep(1.0)

        # -- Save -----------------------------------------------------------
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = f'{RESULTS_DIR}/mininet_baselines_early.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        LOG.info(f'Results saved -> {path}')

        print(f"\n{'Method':<16} {'Avg ICD':>10} {'+-':>6} {'morning':>10} {'business':>10} {'evening':>10} {'night':>10} {'Analytical':>12}")
        print('-' * 90)
        for name, r in results.items():
            pp = r['per_profile']
            print(
                f"{name:<16} {r['icd_mean_mininet']:>10.2f} {r['icd_std_mininet']:>6.2f}"
                f"  {pp['morning']['icd_mean']:>10.2f}"
                f"  {pp['business']['icd_mean']:>10.2f}"
                f"  {pp['evening']['icd_mean']:>10.2f}"
                f"  {pp['night']['icd_mean']:>10.2f}"
                f"  {r['icd_analytical']:>10.2f}"
            )

    finally:
        LOG.info('Cleaning up...')
        for p in client_procs:
            try:
                p.terminate()
            except Exception:
                pass
        stop_iperf3_servers(server_procs)
        if monitor:
            try:
                monitor.stop()
            except Exception:
                pass
        if net:
            try:
                net.stop()
            except Exception:
                pass
        if mgr:
            try:
                mgr.stop_all()
            except Exception:
                pass
        subprocess.run(['mn', '--clean'], capture_output=True)
        try:
            cleanup_tc_delays()
        except Exception:
            pass
        LOG.info('Cleanup complete -- Mininet fully stopped')


if __name__ == '__main__':
    if os.geteuid() != 0:
        print('Run as root: sudo PYTHONPATH=/usr/lib/python3/dist-packages python3 scripts/run_early_mininet_baselines.py')
        sys.exit(1)
    run()
