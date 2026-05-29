# dqn/environment.py
import subprocess
import time
import logging
import numpy as np
from topology.topology_data import (
    SWITCHES, CTRL_PORTS,
    SW_ZONE, CTRL_ZONE, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
    N_SWITCHES, N_CONTROLLERS, STATE_DIM, MAX_CTRL_LOAD,
    ZONE_CTRL, TRAFFIC_PROFILES, PROFILE_IDX,
)
from ryu_apps.multi_controller import CTRL_PORTS as RYU_PORTS

LOG = logging.getLogger(__name__)

_ZONE_ORDER = ['res1', 'res2', 'com1', 'com2', 'ind']


def _zone_idx(zone_name):
    try:
        return _ZONE_ORDER.index(zone_name)
    except ValueError:
        return 0


class CitiverseRealEnv:
    """
    Real environment: 5 Ryu instances + Mininet OvS switches.
    ICD measured via OFPEchoRequest RTT (real propagation through tc netem).
    Throughput from real PACKET_IN rates per controller.
    tc netem updated at each episode reset to reflect current assignments.
    """

    def __init__(self, net, ctrl_mgr, ctrl_monitor=None, seed=42, max_steps=500):
        self.net = net
        self.ctrl_mgr = ctrl_mgr
        self.ctrl_monitor = ctrl_monitor
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps

        self.assignments = np.zeros(N_SWITCHES, dtype=int)
        self.traffic_profile = 'morning'
        self.step_count = 0
        self.migration_count = 0
        self._icd_cache = 50.0
        self._throughput_cache = 80.0
        self._packet_in_rates = np.zeros(N_CONTROLLERS, dtype=np.float32)
        self._iperf_procs = []

    # ------------------------------------------------------------------
    def reset(self):
        """Zone-optimal init + small perturbation. Updates tc netem delays."""
        self.assignments = np.array(
            [ZONE_CTRL[SW_ZONE[sw]] for sw in SWITCHES], dtype=int
        )
        n_perturb = self.rng.integers(0, 4)
        idxs = self.rng.choice(N_SWITCHES, size=n_perturb, replace=False)
        for i in idxs:
            self.assignments[i] = int(self.rng.integers(0, N_CONTROLLERS))

        self.traffic_profile = self.rng.choice(TRAFFIC_PROFILES)
        self.step_count = 0
        self.migration_count = 0

        self._start_background_traffic()
        self._bulk_apply_assignments()

        # Update tc netem to match current episode assignments (real delays)
        try:
            from scripts.setup_tc_delays import setup_loopback_delays
            setup_loopback_delays(self.assignments)
        except Exception as e:
            LOG.warning(f'tc netem update failed in reset: {e}')

        if self.ctrl_monitor is not None:
            self.ctrl_monitor.reset_offsets()

        time.sleep(0.3)
        self._refresh_measurements()
        return self._get_state()

    def step(self, action):
        assert 0 <= action < N_SWITCHES * N_CONTROLLERS
        sw_idx = action // N_CONTROLLERS
        ctrl_idx = action % N_CONTROLLERS
        sw_name = f's{SWITCHES[sw_idx]}'
        old_ctrl = self.assignments[sw_idx]
        migrated = int(old_ctrl) != int(ctrl_idx)

        if migrated:
            self.migration_count += 1
            self.assignments[sw_idx] = ctrl_idx
            self._apply_assignment(sw_name, ctrl_idx)

        self._refresh_measurements()

        state = self._get_state()
        reward = self._compute_reward(migrated)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        info = {
            'icd_ms':          self._icd_cache,
            'throughput_mbps': self._throughput_cache,
            'load_std':        float(np.std(self._compute_load())),
            'profile':         self.traffic_profile,
        }
        return state, reward, done, info

    # ------------------------------------------------------------------
    def _apply_assignment(self, sw_name, ctrl_idx):
        """
        Reassign switch to new controller.
        FIX B01: calls ovs-vsctl.
        FIX B25/B26: clears flow table so new controller starts fresh.
        """
        port = RYU_PORTS[ctrl_idx]
        ctrl_str = f'tcp:127.0.0.1:{port}'

        try:
            subprocess.run(
                ['ovs-vsctl', 'set-controller', sw_name, ctrl_str],
                capture_output=True, timeout=5, check=True,
            )
        except subprocess.CalledProcessError as e:
            LOG.warning(f'set-controller failed for {sw_name}: {e.stderr}')
            return
        except subprocess.TimeoutExpired:
            LOG.error(f'set-controller timeout for {sw_name}')
            return

        try:
            subprocess.run(
                ['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', sw_name],
                capture_output=True, timeout=5,
            )
            LOG.debug(f'Cleared flow table on {sw_name}')
        except Exception as e:
            LOG.warning(f'del-flows failed for {sw_name}: {e}')

        time.sleep(0.2)
        self._wait_for_switch_connected(sw_name, ctrl_idx)
        time.sleep(0.1)

        if self.ctrl_monitor is not None:
            sw_id = int(sw_name[1:])
            self.ctrl_monitor.set_blackout(sw_id, duration=3.0)

    def _wait_for_switch_connected(self, sw_name, ctrl_idx, timeout=2.0):
        """Poll ovs-vsctl until switch shows connected to expected controller."""
        port = RYU_PORTS[ctrl_idx]
        expected = f'tcp:127.0.0.1:{port}'
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                result = subprocess.run(
                    ['ovs-vsctl', 'get-controller', sw_name],
                    capture_output=True, text=True, timeout=3,
                )
                if expected in result.stdout:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        LOG.warning(f'{sw_name} not connected to {expected} after {timeout}s')
        return False

    def _bulk_apply_assignments(self):
        """Apply all assignments in reset() with minimal total wait."""
        for i, sw in enumerate(SWITCHES):
            port = RYU_PORTS[self.assignments[i]]
            ctrl_str = f'tcp:127.0.0.1:{port}'
            try:
                subprocess.run(
                    ['ovs-vsctl', 'set-controller', f's{sw}', ctrl_str],
                    capture_output=True, timeout=5,
                )
            except Exception as e:
                LOG.warning(f'bulk set-controller failed for s{sw}: {e}')

        for sw in SWITCHES:
            try:
                subprocess.run(
                    ['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', f's{sw}'],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass

        time.sleep(0.5)

    # ------------------------------------------------------------------
    def _start_background_traffic(self):
        """Start iperf3 flows between hosts to generate measurable traffic."""
        self._stop_background_traffic()
        try:
            h1 = self.net.get('h1')
            h10 = self.net.get('h10')
            srv = h10.popen('iperf3 -s -p 5201 -D')
            self._iperf_procs.append(srv)
            time.sleep(0.3)
            cli = h1.popen(f'iperf3 -c {h10.IP()} -p 5201 -u -b 10M -t 7200')
            self._iperf_procs.append(cli)
        except Exception as e:
            LOG.warning(f'Failed to start background traffic: {e}')

    def _stop_background_traffic(self):
        for p in self._iperf_procs:
            try:
                p.terminate()
            except Exception:
                pass
        self._iperf_procs.clear()

    # ------------------------------------------------------------------
    def _refresh_measurements(self):
        """Measure ICD and throughput from real network."""
        self._packet_in_rates = self._measure_packet_in_rates()
        self._icd_cache = self._measure_icd()
        self._throughput_cache = self._measure_throughput()

    def _measure_icd(self):
        """
        ICD from real OFPEchoRequest RTT per switch.
        RTT/2 = one-way propagation delay through tc netem + OS stack.
        Fallback to LATENCY_MATRIX if echo RTT not yet available.
        """
        if self.ctrl_monitor is not None:
            echo_rtts = self.ctrl_monitor.get_echo_rtt_ms_per_switch()
            if len(echo_rtts) >= N_SWITCHES // 2:
                total = 0.0
                for i in range(N_SWITCHES):
                    dpid = SWITCHES[i]
                    rtt = echo_rtts.get(dpid)
                    total += (rtt / 2.0) if rtt is not None else float(LATENCY_MATRIX[i][self.assignments[i]])
                return total / N_SWITCHES
        return sum(float(LATENCY_MATRIX[i][self.assignments[i]]) for i in range(N_SWITCHES)) / N_SWITCHES

    def _measure_throughput(self):
        """
        Throughput from real PACKET_IN rates per controller.
        High rate on any controller = potential overload = lower throughput.
        """
        rates = self._packet_in_rates  # real, from ctrl_monitor
        MAX_RATE = 50.0                # events/s threshold
        overloaded_sum = sum(max(0.0, r - MAX_RATE) for r in rates)
        thr = max(0.0, 100.0 - overloaded_sum * 3.0)
        activity = min(10.0, float(np.sum(rates)) * 0.5)
        return min(100.0, thr + activity)

    def _measure_packet_in_rates(self):
        """Get PACKET_IN rates from ControllerMonitor; fallback to log parsing."""
        if self.ctrl_monitor is not None:
            return self.ctrl_monitor.get_packet_in_rates()
        rates = np.zeros(N_CONTROLLERS, dtype=np.float32)
        for i in range(N_CONTROLLERS):
            logf = f'/tmp/ryu_logs/ctrl_{i}.log'
            try:
                with open(logf, 'r') as f:
                    lines = f.readlines()
                count = sum(1 for l in lines[-200:] if 'PACKET_IN' in l)
                rates[i] = min(count / 5.0, 50.0)
            except Exception:
                pass
        return rates

    # ------------------------------------------------------------------
    def _compute_load(self):
        zone_factors = ZONE_LOAD_FACTORS[self.traffic_profile]
        loads = np.zeros(N_CONTROLLERS, dtype=np.float32)
        for i, sw in enumerate(SWITCHES):
            zone = SW_ZONE[sw]
            factor = zone_factors.get(zone, 1.0)
            loads[self.assignments[i]] += factor
        return loads

    def _compute_reward(self, migrated):
        icd_ms = self._icd_cache
        throughput = self._throughput_cache
        loads = self._compute_load()
        load_std = float(np.std(loads))
        zone_match_ratio = sum(
            1 for i, sw in enumerate(SWITCHES)
            if SW_ZONE[sw] == CTRL_ZONE.get(self.assignments[i], '')
        ) / N_SWITCHES

        r = -(
            0.6 * icd_ms / 100.0
            + 0.2 * (1.0 - throughput / 100.0)
            + 0.2 * load_std / 20.0
        )
        r += zone_match_ratio * 0.2
        if migrated:
            r -= 0.05
        for l in loads:
            if l > MAX_CTRL_LOAD:
                r -= 0.3 * (l - MAX_CTRL_LOAD)
        return float(r)

    def _get_state(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        loads = self._compute_load()

        state[0:20] = self.assignments / 4.0
        for i in range(N_SWITCHES):
            state[20 + i] = loads[self.assignments[i]] / MAX_CTRL_LOAD
        for i, sw in enumerate(SWITCHES):
            state[40 + i] = _zone_idx(SW_ZONE[sw]) / 4.0
        for i, sw in enumerate(SWITCHES):
            state[60 + i] = float(SW_ZONE[sw] == CTRL_ZONE.get(self.assignments[i], ''))
        state[80] = self._icd_cache / 100.0
        state[81] = self._throughput_cache / 100.0
        state[82] = float(np.max(loads)) / MAX_CTRL_LOAD
        state[83] = float(np.std(loads)) / 20.0
        state[84:89] = np.clip(self._packet_in_rates / 50.0, 0.0, 1.0)
        pidx = PROFILE_IDX.get(self.traffic_profile, 0)
        state[89 + pidx] = 1.0
        state[93] = min(self.migration_count / 10.0, 1.0)
        return state

    def get_action_mask(self):
        loads = self._compute_load()
        mask = np.ones(N_SWITCHES * N_CONTROLLERS, dtype=bool)
        for sw_idx in range(N_SWITCHES):
            for ctrl_idx in range(N_CONTROLLERS):
                if loads[ctrl_idx] + 1.0 > MAX_CTRL_LOAD * 2:
                    mask[sw_idx * N_CONTROLLERS + ctrl_idx] = False
        return mask

    def set_traffic_profile(self, profile):
        if profile in TRAFFIC_PROFILES:
            self.traffic_profile = profile

    def close(self):
        self._stop_background_traffic()
