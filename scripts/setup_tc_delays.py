# scripts/setup_tc_delays.py
"""
Apply tc netem delays on loopback per controller port.
Without this, RealEnv ICD is ~0.1ms (localhost) — agent learns nothing.
With this, RealEnv ICD matches LATENCY_MATRIX scale (~3-30ms).
Must run as root. Call before Phase 2 training.
"""
import subprocess
import logging
import numpy as np

LOG = logging.getLogger(__name__)

CTRL_PORT_BASE = 6653
N_CONTROLLERS = 5


def setup_loopback_delays(assignments):
    """
    Add tc netem delay on loopback for each controller port.
    assignments: np.array of ctrl_idx per switch (len=20).
    Delay per controller = mean LATENCY_MATRIX row for assigned switches.
    """
    from topology.topology_data import LATENCY_MATRIX, N_SWITCHES

    ctrl_mean_lat = {}
    for ctrl_idx in range(N_CONTROLLERS):
        assigned = [i for i in range(N_SWITCHES) if assignments[i] == ctrl_idx]
        if assigned:
            ctrl_mean_lat[ctrl_idx] = float(
                np.mean([LATENCY_MATRIX[i][ctrl_idx] for i in assigned])
            )
        else:
            ctrl_mean_lat[ctrl_idx] = 5.0

    # Remove existing root qdisc (ignore errors if none exists)
    subprocess.run(
        ['tc', 'qdisc', 'del', 'dev', 'lo', 'root'],
        capture_output=True,
    )

    # Root prio with enough bands
    subprocess.run(
        ['tc', 'qdisc', 'add', 'dev', 'lo', 'root',
         'handle', '1:', 'prio', 'bands', str(N_CONTROLLERS + 1)],
        check=True,
    )

    for ctrl_idx in range(N_CONTROLLERS):
        port = CTRL_PORT_BASE + ctrl_idx
        delay_ms = ctrl_mean_lat[ctrl_idx]
        band = ctrl_idx + 1
        handle = ctrl_idx + 10

        # netem qdisc for this band
        subprocess.run(
            ['tc', 'qdisc', 'add', 'dev', 'lo',
             'parent', f'1:{band}', 'handle', f'{handle}:',
             'netem', 'delay', f'{delay_ms:.0f}ms', '2ms'],
            check=True,
        )

        # Filter: src port (controller tx) -> match this band
        subprocess.run(
            ['tc', 'filter', 'add', 'dev', 'lo', 'protocol', 'ip',
             'parent', '1:0', 'prio', str(band), 'u32',
             'match', 'ip', 'sport', str(port), '0xffff',
             'flowid', f'1:{band}'],
            check=True,
        )
        # Filter: dst port (switch rx) -> same band
        subprocess.run(
            ['tc', 'filter', 'add', 'dev', 'lo', 'protocol', 'ip',
             'parent', '1:0', 'prio', str(band + N_CONTROLLERS), 'u32',
             'match', 'ip', 'dport', str(port), '0xffff',
             'flowid', f'1:{band}'],
            check=True,
        )

        LOG.info(f'ctrl_{ctrl_idx} port={port} delay={delay_ms:.1f}ms')

    LOG.info(f'tc netem delays set: {ctrl_mean_lat}')
    print(f'[tc netem] Delays set: { {f"ctrl{k}": f"{v:.1f}ms" for k,v in ctrl_mean_lat.items()} }')
    return ctrl_mean_lat


def cleanup_tc_delays():
    """Remove all tc qdiscs on loopback."""
    subprocess.run(
        ['tc', 'qdisc', 'del', 'dev', 'lo', 'root'],
        capture_output=True,
    )
    LOG.info('tc delays cleaned up')
    print('[tc netem] Delays cleaned up')


def verify_delays():
    """Print current tc qdisc config for debugging."""
    result = subprocess.run(
        ['tc', 'qdisc', 'show', 'dev', 'lo'],
        capture_output=True, text=True,
    )
    print(result.stdout)
    return result.stdout


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/home/maksim/dqn_simenv_mininet')
    from topology.topology_data import get_zone_optimal_array
    assignments = get_zone_optimal_array()
    setup_loopback_delays(assignments)
    print('\nCurrent tc config:')
    verify_delays()
