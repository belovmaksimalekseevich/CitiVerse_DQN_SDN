# baselines/run_baselines.py
import json
import os
import numpy as np
from sklearn.cluster import KMeans
from topology.topology_data import (
    SWITCHES, CONTROLLERS, SW_ZONE, ZONE_CTRL, CTRL_ZONE,
    LATENCY_MATRIX, N_SWITCHES, N_CONTROLLERS, MAX_CTRL_LOAD,
    ZONE_LOAD_FACTORS,
)

RESULTS_DIR = 'results'


# ---------------------------------------------------------------------------
# Assignment strategies
# ---------------------------------------------------------------------------

def baseline_all_to_ctrl0():
    """All switches -> controller 0 (worst case, centralised)."""
    return np.zeros(N_SWITCHES, dtype=int)


def baseline_zone_optimal():
    """Each switch -> its zone's home controller (true optimum for static traffic)."""
    return np.array([ZONE_CTRL[SW_ZONE[sw]] for sw in SWITCHES], dtype=int)


def baseline_load_balanced():
    """Round-robin by current load count (ignores latency)."""
    assignments = np.zeros(N_SWITCHES, dtype=int)
    loads = np.zeros(N_CONTROLLERS)
    for i in range(N_SWITCHES):
        c = int(np.argmin(loads))
        assignments[i] = c
        loads[c] += 1.0
    return assignments


def baseline_kmeans(n_clusters=N_CONTROLLERS, seed=42):
    """K-Means on latency feature matrix — clusters switches by controller proximity."""
    X = np.array(LATENCY_MATRIX, dtype=np.float32)   # shape (20, 5)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    return labels.astype(int)


def baseline_random(seed=42):
    """Uniform random assignment — lower bound baseline."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_CONTROLLERS, size=N_SWITCHES)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_icd(assignments):
    total = sum(float(LATENCY_MATRIX[i][assignments[i]]) for i in range(N_SWITCHES))
    return total / N_SWITCHES


def compute_load_std(assignments):
    loads = np.zeros(N_CONTROLLERS)
    for c in assignments:
        loads[c] += 1.0
    return float(np.std(loads))


def compute_zone_match(assignments):
    match = sum(
        1 for i, sw in enumerate(SWITCHES)
        if SW_ZONE[sw] == CTRL_ZONE.get(int(assignments[i]), '')
    )
    return match / N_SWITCHES


def compute_reward(assignments, traffic_profile='business'):
    """Same reward formula as CitiverseSimEnv/_compute_reward."""
    icd_ms = compute_icd(assignments)
    zone_factors = ZONE_LOAD_FACTORS[traffic_profile]
    loads = np.zeros(N_CONTROLLERS)
    for i, sw in enumerate(SWITCHES):
        factor = zone_factors.get(SW_ZONE[sw], 1.0)
        loads[int(assignments[i])] += factor
    load_std = float(np.std(loads))
    overload_penalty = sum(max(0.0, l - MAX_CTRL_LOAD) for l in loads)
    throughput = max(0.0, 100.0 - overload_penalty * 10.0)
    zone_match = compute_zone_match(assignments)

    r = -(0.6 * icd_ms / 100.0
          + 0.2 * (1.0 - throughput / 100.0)
          + 0.2 * load_std / 20.0)
    r += zone_match * 0.2
    return float(r)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_baselines(save=True):
    baselines = {
        'AllToCtrl0':   baseline_all_to_ctrl0(),
        'ZoneOptimal':  baseline_zone_optimal(),
        'LoadBalanced': baseline_load_balanced(),
        'KMeans':       baseline_kmeans(),
        'Random':       baseline_random(),
    }

    results = {}
    print(f"\n{'Baseline':<20} {'ICD_ms':>8} {'load_std':>10} {'zone_match':>12} {'reward':>8}")
    print('-' * 62)

    for name, assignments in baselines.items():
        metrics = {
            'icd_ms':      round(compute_icd(assignments), 3),
            'load_std':    round(compute_load_std(assignments), 3),
            'zone_match':  round(compute_zone_match(assignments), 3),
            'reward':      round(compute_reward(assignments), 4),
            'assignments': assignments.tolist(),
        }
        results[name] = metrics
        print(f"{name:<20} {metrics['icd_ms']:>8.2f} {metrics['load_std']:>10.3f} "
              f"{metrics['zone_match']:>12.3f} {metrics['reward']:>8.4f}")

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = f'{RESULTS_DIR}/baselines.json'
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nSaved {path}')

    return results


if __name__ == '__main__':
    run_all_baselines(save=True)
