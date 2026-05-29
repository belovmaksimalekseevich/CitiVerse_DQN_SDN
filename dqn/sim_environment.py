# dqn/sim_environment.py
import numpy as np
from topology.topology_data import (
    SWITCHES, CONTROLLERS, CTRL_PORTS,
    SW_ZONE, CTRL_ZONE, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
    N_SWITCHES, N_CONTROLLERS, STATE_DIM, MAX_CTRL_LOAD, LOAD_NORM, queuing_delay_ms,
    ZONE_CTRL, TRAFFIC_PROFILES, PROFILE_IDX,
)

_ZONE_ORDER = ['res1', 'res2', 'com1', 'com2', 'ind']


def _zone_idx(zone_name):
    try:
        return _ZONE_ORDER.index(zone_name)
    except ValueError:
        return 0


def _ctrl_zone(ctrl_idx):
    return CTRL_ZONE.get(ctrl_idx, 'res1')


class CitiverseSimEnv:
    """
    Analytical pre-training environment (no real network).
    Matches state/action/reward spec of CitiverseRealEnv exactly.
    Supports curriculum learning (TASK_10).
    """

    # Curriculum stages: (max_episode, n_switches, n_controllers, profiles)
    _STAGES = [
        (300,  5,  2, ['morning']),
        (800,  10, 3, ['morning', 'business']),
        (3000, 20, 5, ['morning', 'business', 'evening', 'night']),
    ]

    def __init__(self, seed=42, fault_prob=0.05, latency_jitter=0.1,
                 curriculum=True):
        self.rng = np.random.default_rng(seed)
        self.fault_prob = fault_prob
        self.latency_jitter = latency_jitter
        self.curriculum = curriculum
        self.global_episode = 0
        self.max_steps = 200

        self.assignments = np.zeros(N_SWITCHES, dtype=int)
        self.traffic_profile = 'morning'
        self.step_count = 0
        self.failed_ctrl = -1
        self.migration_count = 0
        self.active_n_sw = N_SWITCHES
        self.active_n_ctrl = N_CONTROLLERS
        self.active_profiles = TRAFFIC_PROFILES
        self.forced_failure = -1   # >=0 => sustained controller failure (eval)

    def set_episode(self, ep):
        """Called from train loop to update curriculum stage."""
        self.global_episode = ep

    def _get_curriculum_params(self):
        if not self.curriculum:
            return N_SWITCHES, N_CONTROLLERS, TRAFFIC_PROFILES
        for (max_ep, n_sw, n_ctrl, profiles) in self._STAGES:
            if self.global_episode < max_ep:
                return n_sw, n_ctrl, profiles
        return N_SWITCHES, N_CONTROLLERS, TRAFFIC_PROFILES

    def _get_stage_id(self):
        ep = self.global_episode
        if ep < 300:  return 0
        if ep < 800:  return 1
        return 2

    def reset(self):
        n_sw, n_ctrl, profiles = self._get_curriculum_params()
        self.active_n_sw = n_sw
        self.active_n_ctrl = n_ctrl
        self.active_profiles = profiles

        active_switches = SWITCHES[:n_sw]
        self.assignments = np.zeros(N_SWITCHES, dtype=int)
        for i, sw in enumerate(active_switches):
            self.assignments[i] = ZONE_CTRL[SW_ZONE[sw]] % n_ctrl

        n_perturb = self.rng.integers(1, min(4, n_sw))
        idxs = self.rng.choice(n_sw, size=n_perturb, replace=False)
        for i in idxs:
            self.assignments[i] = self.rng.integers(0, n_ctrl)

        self.traffic_profile = self.rng.choice(profiles)
        self.step_count = 0
        self.failed_ctrl = -1
        self.migration_count = 0
        return self._get_state()

    def step(self, action):
        assert 0 <= action < N_SWITCHES * N_CONTROLLERS
        n_sw = self.active_n_sw
        n_ctrl = self.active_n_ctrl
        sw_idx = action // N_CONTROLLERS
        ctrl_idx = action % N_CONTROLLERS
        sw_idx = min(sw_idx, n_sw - 1)
        ctrl_idx = min(ctrl_idx, n_ctrl - 1)

        old_ctrl = self.assignments[sw_idx]
        migrated = int(old_ctrl) != int(ctrl_idx)
        if migrated:
            self.migration_count += 1
        self.assignments[sw_idx] = ctrl_idx

        self.failed_ctrl = -1
        if self.rng.random() < self.fault_prob and n_ctrl > 1:
            self.failed_ctrl = int(self.rng.integers(0, n_ctrl))

        if self.rng.random() < 0.02:
            self.traffic_profile = self.rng.choice(self.active_profiles)

        state = self._get_state()
        reward = self._compute_reward(migrated)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        info = {
            'icd_ms':           self._compute_icd(),
            'load_std':         self._compute_load_std(),
            'profile':          self.traffic_profile,
            'failed_ctrl':      self.failed_ctrl,
            'curriculum_stage': self._get_stage_id(),
        }
        return state, reward, done, info

    def _resolve_ctrl(self, ctrl_idx):
        dead = self.forced_failure if self.forced_failure >= 0 else self.failed_ctrl
        if dead >= 0 and ctrl_idx == dead:
            return (ctrl_idx + 1) % self.active_n_ctrl
        return ctrl_idx

    def _get_latency(self, sw_idx, ctrl_idx):
        base = LATENCY_MATRIX[sw_idx][ctrl_idx]
        jitter = self.rng.uniform(1 - self.latency_jitter, 1 + self.latency_jitter)
        return float(base * jitter)

    def _compute_icd(self):
        n_sw = self.active_n_sw
        if n_sw == 0:
            return 0.0
        loads = self._compute_load()
        total = 0.0
        for i in range(n_sw):
            c = self._resolve_ctrl(self.assignments[i])
            total += self._get_latency(i, c) + queuing_delay_ms(loads[c])
        return total / n_sw

    def _compute_load(self):
        zone_factors = ZONE_LOAD_FACTORS[self.traffic_profile]
        loads = np.zeros(N_CONTROLLERS, dtype=float)
        for i, sw in enumerate(SWITCHES[:self.active_n_sw]):
            zone = SW_ZONE[sw]
            factor = zone_factors.get(zone, 1.0)
            c = self._resolve_ctrl(self.assignments[i])
            loads[c] += factor
        return loads

    def _compute_load_std(self):
        return float(np.std(self._compute_load()))

    def _compute_throughput(self):
        loads = self._compute_load()
        overload_penalty = sum(max(0.0, l - MAX_CTRL_LOAD) for l in loads)
        return max(0.0, 100.0 - overload_penalty * 10.0)

    def _compute_zone_match_ratio(self):
        n_sw = self.active_n_sw
        if n_sw == 0:
            return 0.0
        matches = sum(
            1 for i, sw in enumerate(SWITCHES[:n_sw])
            if SW_ZONE[sw] == _ctrl_zone(self.assignments[i])
        )
        return matches / n_sw

    def _compute_reward(self, migrated):
        # Reward is driven purely by ICD (propagation + M/M/1 queuing), so
        # minimising ICD = staying in-zone (low propagation) AND balancing
        # load (low queuing). Small migration penalty prevents thrashing.
        icd_ms = self._compute_icd()
        r = -(icd_ms / 20.0)
        if migrated:
            r -= 0.02
        return float(r)

    def _get_state(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        loads = self._compute_load()

        # [0:20]  assignments normalised
        state[0:20] = self.assignments / 4.0

        # [20:40] load of each switch's assigned controller
        for i in range(N_SWITCHES):
            state[20 + i] = min(loads[self.assignments[i]] / LOAD_NORM, 1.0)

        # [40:60] zone index of each switch
        for i, sw in enumerate(SWITCHES):
            state[40 + i] = _zone_idx(SW_ZONE[sw]) / 4.0

        # [60:80] zone match flag per switch
        for i, sw in enumerate(SWITCHES):
            state[60 + i] = float(SW_ZONE[sw] == _ctrl_zone(self.assignments[i]))

        # [80]    ICD normalised
        state[80] = self._compute_icd() / 100.0
        # [81]    throughput normalised
        state[81] = self._compute_throughput() / 100.0
        # [82]    max load normalised
        state[82] = min(float(np.max(loads)) / LOAD_NORM, 1.0)
        # [83]    load std normalised
        state[83] = self._compute_load_std() / 20.0

        # [84:89] PACKET_IN rate per controller (simulated from load)
        for c in range(N_CONTROLLERS):
            state[84 + c] = min(loads[c] * 10.0, 50.0) / 50.0

        # [89:93] traffic profile one-hot
        pidx = PROFILE_IDX.get(self.traffic_profile, 0)
        state[89 + pidx] = 1.0

        # [93]    migration count normalised
        state[93] = min(self.migration_count / 10.0, 1.0)

        return state

    def get_action_mask(self):
        """Valid actions = any (switch, controller) within active curriculum.
        No hard load cap: controller overload is penalised via queuing in ICD,
        so the agent is free to discover zone-optimal AND offloading actions."""
        n_sw = self.active_n_sw
        n_ctrl = self.active_n_ctrl
        mask = np.zeros(N_SWITCHES * N_CONTROLLERS, dtype=bool)
        for sw_idx in range(n_sw):
            for ctrl_idx in range(n_ctrl):
                mask[sw_idx * N_CONTROLLERS + ctrl_idx] = True
        return mask
