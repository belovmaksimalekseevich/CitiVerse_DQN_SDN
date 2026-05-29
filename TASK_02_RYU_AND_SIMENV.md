# TASK_02: Ryu Multi-Controller + CitiverseSimEnv

## Step 3 — ryu_apps/ctrl_app.py + multi_controller.py

### File: ryu_apps/ctrl_app.py
Single Ryu app that handles one controller. Logs PACKET_IN rate and ICD.

```python
# ryu_apps/ctrl_app.py
import time, logging
from collections import defaultdict
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4

LOG = logging.getLogger(__name__)

class CitiCtrlApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.packet_in_counts = defaultdict(int)   # dpid -> count
        self.window_start = time.time()
        self.window_size = 5.0  # seconds
        self.packet_in_rates = {}  # dpid -> rate/s (last window)
        self.mac_to_port = {}

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=0, match=match, instructions=inst)
        dp.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        self.packet_in_counts[dpid] += 1
        self._update_rates()
        ofp = dp.ofproto
        parser = dp.ofproto_parser
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        dst = eth.dst
        src = eth.src
        in_port = msg.match['in_port']
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(dp, 1, match, actions)
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        dp.send_msg(out)

    def _add_flow(self, dp, priority, match, actions):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=dp, priority=priority,
                                 match=match, instructions=inst)
        dp.send_msg(mod)

    def _update_rates(self):
        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= self.window_size:
            for dpid, cnt in self.packet_in_counts.items():
                self.packet_in_rates[dpid] = cnt / elapsed
            self.packet_in_counts.clear()
            self.window_start = now

    def get_packet_in_rate(self, dpid):
        return self.packet_in_rates.get(dpid, 0.0)
```

### File: ryu_apps/multi_controller.py
Manages 5 Ryu processes, ports 6653-6657. Used by CitiverseRealEnv.

```python
# ryu_apps/multi_controller.py
import subprocess, time, os, signal, logging

LOG = logging.getLogger(__name__)
CTRL_PORTS = [6653, 6654, 6655, 6656, 6657]
N_CONTROLLERS = 5

class MultiControllerManager:
    """Starts/stops 5 ryu-manager processes on ports 6653-6657."""

    def __init__(self, app_path='ryu_apps/ctrl_app.py', log_dir='/tmp/ryu_logs'):
        self.app_path = app_path
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.procs = {}   # port -> Popen

    def start_all(self):
        for i, port in enumerate(CTRL_PORTS):
            logfile = open(f'{self.log_dir}/ctrl_{i}.log', 'w')
            proc = subprocess.Popen(
                ['ryu-manager', self.app_path,
                 '--ofp-tcp-listen-port', str(port),
                 '--observe-links'],
                stdout=logfile, stderr=logfile
            )
            self.procs[port] = proc
            LOG.info(f'Started Ryu ctrl {i} on port {port}, pid={proc.pid}')
        time.sleep(3)  # wait for Ryu to initialize

    def stop_all(self):
        for port, proc in self.procs.items():
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
            LOG.info(f'Stopped Ryu on port {port}')
        self.procs.clear()

    def is_running(self, ctrl_idx):
        port = CTRL_PORTS[ctrl_idx]
        proc = self.procs.get(port)
        return proc is not None and proc.poll() is None

    def restart_ctrl(self, ctrl_idx):
        port = CTRL_PORTS[ctrl_idx]
        if port in self.procs:
            self.procs[port].kill()
        logfile = open(f'{self.log_dir}/ctrl_{ctrl_idx}_restart.log', 'w')
        proc = subprocess.Popen(
            ['ryu-manager', self.app_path,
             '--ofp-tcp-listen-port', str(port),
             '--observe-links'],
            stdout=logfile, stderr=logfile
        )
        self.procs[port] = proc
        time.sleep(2)
```

---

## Step 4 — dqn/sim_environment.py (CitiverseSimEnv)

Analytical fast simulation for Phase 1 pretraining. No real network. ~4 min for 5000 episodes.

Key design:
- No Mininet, no Ryu — pure Python math
- Simulates ICD from latency matrix
- Simulates load from ZONE_LOAD_FACTORS × traffic_profile
- Fault injection: random controller failure
- Domain randomization: jitter on latencies, zone loads

```python
# dqn/sim_environment.py
import numpy as np
from topology.topology_data import (
    SWITCHES, CONTROLLERS, ZONE_MAP, CTRL_PORTS,
    SW_ZONE, LATENCY_MATRIX, ZONE_LOAD_FACTORS,
    N_SWITCHES, N_CONTROLLERS, STATE_DIM, MAX_CTRL_LOAD
)

TRAFFIC_PROFILES = ['morning', 'business', 'evening', 'night']
PROFILE_IDX = {p: i for i, p in enumerate(TRAFFIC_PROFILES)}

class CitiverseSimEnv:
    """
    Analytical pre-training environment. No real network.
    Matches state/action/reward spec of CitiverseRealEnv exactly.
    """

    def __init__(self, seed=42, fault_prob=0.05, latency_jitter=0.1):
        self.rng = np.random.default_rng(seed)
        self.fault_prob = fault_prob
        self.latency_jitter = latency_jitter
        self.assignments = np.zeros(N_SWITCHES, dtype=int)
        self.traffic_profile = 'morning'
        self.step_count = 0
        self.max_steps = 200
        self.failed_ctrl = -1
        self.migration_count = 0

    # ------------------------------------------------------------------
    def reset(self):
        # Zone-optimal init with small noise (avoids random overload at start)
        from topology.topology_data import ZONE_CTRL
        self.assignments = np.array([ZONE_CTRL[SW_ZONE[sw]] for sw in SWITCHES], dtype=int)
        # Random perturbation: reassign 2-4 switches randomly
        n_perturb = self.rng.integers(2, 5)
        idxs = self.rng.choice(N_SWITCHES, size=n_perturb, replace=False)
        for i in idxs:
            self.assignments[i] = self.rng.integers(0, N_CONTROLLERS)
        self.traffic_profile = self.rng.choice(TRAFFIC_PROFILES)
        self.step_count = 0
        self.failed_ctrl = -1
        self.migration_count = 0
        return self._get_state()

    def step(self, action):
        assert 0 <= action < N_SWITCHES * N_CONTROLLERS
        sw_idx = action // N_CONTROLLERS
        ctrl_idx = action % N_CONTROLLERS
        old_ctrl = self.assignments[sw_idx]
        migrated = (old_ctrl != ctrl_idx)
        if migrated:
            self.migration_count += 1
        self.assignments[sw_idx] = ctrl_idx

        # Fault injection
        self.failed_ctrl = -1
        if self.rng.random() < self.fault_prob:
            self.failed_ctrl = self.rng.integers(0, N_CONTROLLERS)

        # Possibly switch traffic profile every ~50 steps
        if self.rng.random() < 0.02:
            self.traffic_profile = self.rng.choice(TRAFFIC_PROFILES)

        state = self._get_state()
        reward = self._compute_reward(migrated)
        self.step_count += 1
        done = (self.step_count >= self.max_steps)
        info = {
            'icd_ms': self._compute_icd(),
            'load_std': self._compute_load_std(),
            'profile': self.traffic_profile,
            'failed_ctrl': self.failed_ctrl,
        }
        return state, reward, done, info

    # ------------------------------------------------------------------
    def _get_latency(self, sw_idx, ctrl_idx):
        """Latency with domain randomization jitter."""
        base = LATENCY_MATRIX[sw_idx][ctrl_idx]
        jitter = self.rng.uniform(1 - self.latency_jitter, 1 + self.latency_jitter)
        return base * jitter

    def _compute_icd(self):
        total = 0.0
        for i in range(N_SWITCHES):
            c = self.assignments[i]
            if c == self.failed_ctrl:
                c = (c + 1) % N_CONTROLLERS
            total += self._get_latency(i, c)
        return total / N_SWITCHES

    def _compute_load(self):
        zone_factors = ZONE_LOAD_FACTORS[self.traffic_profile]
        loads = np.zeros(N_CONTROLLERS)
        for i, sw in enumerate(SWITCHES):
            zone = SW_ZONE[sw]
            factor = zone_factors.get(zone, 1.0)
            c = self.assignments[i]
            if c == self.failed_ctrl:
                c = (c + 1) % N_CONTROLLERS
            loads[c] += factor
        return loads

    def _compute_load_std(self):
        return float(np.std(self._compute_load()))

    def _compute_throughput(self):
        # Analytical: penalize overloaded controllers
        loads = self._compute_load()
        overload_penalty = sum(max(0, l - MAX_CTRL_LOAD) for l in loads)
        throughput = max(0.0, 100.0 - overload_penalty * 10.0)
        return throughput

    def _compute_reward(self, migrated):
        icd_ms = self._compute_icd()
        throughput = self._compute_throughput()
        load_std = self._compute_load_std()
        loads = self._compute_load()
        zone_match_ratio = sum(
            1 for i, sw in enumerate(SWITCHES)
            if SW_ZONE[sw] == _ctrl_zone(self.assignments[i])
        ) / N_SWITCHES

        r = -(0.6 * icd_ms / 100.0
              + 0.2 * (1.0 - throughput / 100.0)
              + 0.2 * load_std / N_SWITCHES)
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
        zone_factors = ZONE_LOAD_FACTORS[self.traffic_profile]

        state[0:20] = self.assignments / 4.0
        state[20:40] = loads[self.assignments] / MAX_CTRL_LOAD
        state[40:60] = np.array([_zone_idx(SW_ZONE[sw]) for sw in SWITCHES]) / 4.0
        state[60:80] = np.array([
            1.0 if SW_ZONE[sw] == _ctrl_zone(self.assignments[i]) else 0.0
            for i, sw in enumerate(SWITCHES)
        ])
        state[80] = self._compute_icd() / 100.0
        state[81] = self._compute_throughput() / 100.0
        state[82] = float(np.max(loads)) / MAX_CTRL_LOAD
        state[83] = self._compute_load_std() / 20.0
        # packet_in_rate: simulate as proportional to load
        for c in range(N_CONTROLLERS):
            state[84 + c] = min(loads[c] * 10.0, 50.0) / 50.0
        # traffic profile one-hot
        pidx = PROFILE_IDX[self.traffic_profile]
        state[89 + pidx] = 1.0
        state[93] = self.migration_count / 10.0
        return state

    def get_action_mask(self):
        """True = valid action."""
        loads = self._compute_load()
        mask = np.ones(N_SWITCHES * N_CONTROLLERS, dtype=bool)
        for sw_idx in range(N_SWITCHES):
            for ctrl_idx in range(N_CONTROLLERS):
                # block if target controller already at 2x capacity
                projected = loads[ctrl_idx] + 1.0
                if projected > MAX_CTRL_LOAD * 2:
                    mask[sw_idx * N_CONTROLLERS + ctrl_idx] = False
        return mask


def _zone_idx(zone_name):
    zones = ['res1', 'res2', 'com1', 'com2', 'ind']
    return zones.index(zone_name) if zone_name in zones else 0

def _ctrl_zone(ctrl_idx):
    from topology.topology_data import CTRL_ZONE
    return CTRL_ZONE.get(ctrl_idx, 'res1')
```

---

## Environment Setup Commands (run once on server)

```bash
cd /home/maksim
python3 -m venv dqn_env
source dqn_env/bin/activate
pip install ryu numpy torch gymnasium mininet-wifi 2>/dev/null || true
pip install scikit-learn matplotlib networkx scipy

mkdir -p dqn_simenv_mininet/{topology,ryu_apps,dqn,baselines,scripts,results}
touch dqn_simenv_mininet/__init__.py
touch dqn_simenv_mininet/{topology,ryu_apps,dqn,baselines,scripts}/__init__.py
```
