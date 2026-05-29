# dqn/environment.py
import subprocess
import time
import logging
import numpy as np
from topology.topology_data import (
    SWITCHES, CTRL_PORTS,
    SW_ZONE, CTRL_ZONE, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
    N_SWITCHES, N_CONTROLLERS, STATE_DIM, MAX_CTRL_LOAD, LOAD_NORM, queuing_delay_ms,
    ZONE_CTRL, TRAFFIC_PROFILES, PROFILE_IDX,
)
from ryu_apps.multi_controller import CTRL_PORTS as RYU_PORTS

LOG = logging.getLogger(__name__)

_ZONE_ORDER = ['res1', 'res2', 'com1', 'com2', 'ind']

# iperf3 traffic parameters
_BASE_RATE_MBPS   = 2.0
_CTRL_LOOPBACK    = ['127.0.0.1', '127.0.0.2', '127.0.0.3', '127.0.0.4', '127.0.0.5']
_IPERF3_BASE_PORT = 5200   # switch i uses port 5200+i; servers bind 0.0.0.0
_IPERF3_WAIT      = 0.5    # seconds to wait after launching clients


def _zone_idx(zone_name):
    try:
        return _ZONE_ORDER.index(zone_name)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Traffic generator: one iperf3 server per switch (unique port), clients
# send to the assigned controller's loopback IP at that port.  20 UDP flows
# run simultaneously -> per-controller load proportional to zone_load_factor.
# ---------------------------------------------------------------------------

class TrafficGenerator:
    """
    Per-switch iperf3 UDP traffic at rates proportional to zone_load_factors.

    Mechanism: 20 iperf3 UDP streams run on the host loopback (OVSSwitch nodes
    execute in the root namespace).  Each stream targets the loopback IP of the
    switch's assigned controller.  Streams that land on the same controller
    accumulate proportionally to zone_load_factor x n_assigned_switches,
    loading that controller's CPU.  Ryu uses eventlet cooperative threading;
    under CPU load the OFPEchoRequest RTT rises, making queuing delay visible
    in the ICD metric.
    """

    def __init__(self, net, switch_names, zone_map):
        """
        net          - Mininet net object
        switch_names - list of 20 switch names ['s1'..'s20']
        zone_map     - dict {sw_index: zone_name}
        """
        self.net = net
        self.switch_names = switch_names
        self.zone_map = zone_map
        self._servers = []   # proc list
        self._clients = []   # proc list
        self._ready   = False

    # ------------------------------------------------------------------
    def start_servers(self):
        """One iperf3 server per switch on port 5200+sw_idx, bound to 0.0.0.0."""
        self._stop_servers()
        for sw_idx in range(len(self.switch_names)):
            port = _IPERF3_BASE_PORT + sw_idx
            try:
                proc = subprocess.Popen(
                    ['iperf3', '-s', '-p', str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._servers.append(proc)
            except Exception as e:
                LOG.warning(f'iperf3 server sw_idx={sw_idx} port={port} failed: {e}')
        time.sleep(0.5)
        self._ready = True
        LOG.info(f'TrafficGenerator: {len(self._servers)} iperf3 servers started '
                 f'(ports {_IPERF3_BASE_PORT}-{_IPERF3_BASE_PORT+len(self.switch_names)-1})')

    def update_traffic(self, assignments, profile):
        """Kill old clients, start new ones for current assignments + profile."""
        if not self._ready:
            return
        self._stop_clients()
        zone_factors = ZONE_LOAD_FACTORS[profile]
        for sw_idx, sw_name in enumerate(self.switch_names):
            ctrl_idx = int(assignments[sw_idx])
            zone     = self.zone_map[sw_idx]
            rate     = _BASE_RATE_MBPS * zone_factors.get(zone, 1.0)
            dst_ip   = _CTRL_LOOPBACK[ctrl_idx]
            port     = _IPERF3_BASE_PORT + sw_idx
            cmd = (f'iperf3 -c {dst_ip} -p {port} -u -b {rate:.2f}M '
                   f'-t 3600 -N 2>/dev/null')
            try:
                sw   = self.net.get(sw_name)
                proc = sw.popen(cmd, shell=True)
                self._clients.append(proc)
            except Exception as e:
                LOG.debug(f'iperf3 client {sw_name}: {e}')
        time.sleep(_IPERF3_WAIT)

    def stop(self):
        """Stop all traffic."""
        self._stop_clients()
        self._stop_servers()
        try:
            subprocess.run(['pkill', '-f', f'iperf3.*{_IPERF3_BASE_PORT}'],
                           capture_output=True)
        except Exception:
            pass

    def _stop_clients(self):
        for p in self._clients:
            try:
                p.terminate()
            except Exception:
                pass
        self._clients.clear()
        try:
            subprocess.run(['pkill', '-f', 'iperf3 -c 127\\.0\\.0'],
                           capture_output=True)
        except Exception:
            pass

    def _stop_servers(self):
        for p in self._servers:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                pass
        self._servers.clear()


# ---------------------------------------------------------------------------

class CitiverseRealEnv:
    """
    Real environment: 5 Ryu instances + Mininet OvS switches.
    ICD measured via OFPEchoRequest RTT (real propagation through tc netem).
    Throughput from real PACKET_IN rates per controller.
    tc netem updated at each episode reset to reflect current assignments.
    TrafficGenerator creates per-controller UDP load proportional to
    zone_load_factors so queuing delay is visible in the ICD metric.
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

        self._traffic_gen = TrafficGenerator(
            net=net,
            switch_names=[f's{sw}' for sw in SWITCHES],
            zone_map={i: SW_ZONE[sw] for i, sw in enumerate(SWITCHES)},
        )
        self._traffic_gen.start_servers()

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

        self._bulk_apply_assignments()

        try:
            from scripts.setup_tc_delays import setup_loopback_delays
            setup_loopback_delays(self.assignments)
        except Exception as e:
            LOG.warning(f'tc netem update failed in reset: {e}')

        if self.ctrl_monitor is not None:
            self.ctrl_monitor.reset_offsets()

        # Start traffic for this episode's assignment + profile.
        # _IPERF3_WAIT (0.5s) inside update_traffic replaces the old sleep(0.3).
        self._traffic_gen.update_traffic(self.assignments, self.traffic_profile)

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
            # Redirect traffic to new controller so ICD reflects updated load.
            self._traffic_gen.update_traffic(self.assignments, self.traffic_profile)

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
    def _refresh_measurements(self):
        """Measure ICD and throughput from real network."""
        self._packet_in_rates = self._measure_packet_in_rates()
        self._icd_cache = self._measure_icd()
        self._throughput_cache = self._measure_throughput()

    def _measure_icd(self):
        """ICD = per-switch propagation (LATENCY_MATRIX) + M/M/1 controller
        queuing delay, identical to SimEnv and baselines, so the metric is
        consistent across train/eval/baseline and reacts to every reassignment.
        Real OFPEchoRequest RTT is still collected by ControllerMonitor as a
        propagation-realism check but is not used as the optimisation signal."""
        loads = self._compute_load()
        total = 0.0
        for i in range(N_SWITCHES):
            c = int(self.assignments[i])
            total += float(LATENCY_MATRIX[i][c]) + queuing_delay_ms(loads[c])
        return total / N_SWITCHES

    def _measure_throughput(self):
        """
        Throughput from real PACKET_IN rates per controller.
        High rate on any controller = potential overload = lower throughput.
        """
        rates = self._packet_in_rates
        MAX_RATE = 50.0
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
        # Pure ICD-driven reward (propagation + queuing), matches SimEnv.
        r = -(self._icd_cache / 20.0)
        if migrated:
            r -= 0.02
        return float(r)

    def _get_state(self):
        state = np.zeros(STATE_DIM, dtype=np.float32)
        loads = self._compute_load()

        state[0:20] = self.assignments / 4.0
        for i in range(N_SWITCHES):
            state[20 + i] = min(loads[self.assignments[i]] / LOAD_NORM, 1.0)
        for i, sw in enumerate(SWITCHES):
            state[40 + i] = float(_zone_idx(SW_ZONE[sw])) / 4.0
        for i, sw in enumerate(SWITCHES):
            state[60 + i] = float(SW_ZONE[sw] == CTRL_ZONE.get(self.assignments[i], ''))
        state[80] = self._icd_cache / 100.0
        state[81] = self._throughput_cache / 100.0
        state[82] = min(float(np.max(loads)) / LOAD_NORM, 1.0)
        state[83] = float(np.std(loads)) / 20.0
        state[84:89] = np.clip(self._packet_in_rates / 50.0, 0.0, 1.0)
        pidx = PROFILE_IDX.get(self.traffic_profile, 0)
        state[89 + pidx] = 1.0
        state[93] = min(self.migration_count / 10.0, 1.0)
        return state

    def get_action_mask(self):
        # No hard load cap: controller overload is penalised via queuing in ICD.
        return np.ones(N_SWITCHES * N_CONTROLLERS, dtype=bool)

    def set_traffic_profile(self, profile):
        if profile in TRAFFIC_PROFILES:
            self.traffic_profile = profile

    def close(self):
        self._traffic_gen.stop()
