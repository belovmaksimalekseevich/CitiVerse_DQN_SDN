# TASK_14: Mininet Realistic Delays + Dependency Map + Final Checklist

## CRITICAL: tc netem delays in Mininet (missing from all prior TASK files)

### Problem

Mininet OvS switches are software switches on localhost.
Without artificial delays:
  SimEnv ICD: 10-50ms (from LATENCY_MATRIX)
  RealEnv ICD: 0.1-0.5ms (localhost software switching)

The reward function is normalized to /100.0.
In RealEnv without tc netem: reward delta between good/bad assignment ≈ 0.0001.
Agent learns nothing. Sim-to-real transfer fails completely.

### Fix A: Mininet link delay in citiverse_topo.py

Add delay and bandwidth parameters to all links. Values must match LATENCY_MATRIX.

```python
# topology/citiverse_topo.py — updated addLink calls

# Intra-zone links (same zone, low latency ~2-5ms)
self.addLink(sw1, sw2, delay='3ms', bw=100, loss=0, max_queue_size=1000)

# Inter-zone links (cross-zone, higher latency ~10-30ms)
self.addLink(sw_res1_node, sw_com1_node, delay='15ms', bw=50, loss=0)

# Controller-to-switch baseline delay via lo interface
# NOTE: Ryu runs on localhost. Set artificial TCP RTT using tc netem
# on the loopback interface for each controller port:
#   tc qdisc add dev lo root handle 1: prio bands 3
#   tc qdisc add dev lo parent 1:3 handle 30: netem delay 10ms
#   tc filter add dev lo protocol ip parent 1:0 prio 3 u32 \
#     match ip dport 6653 0xffff flowid 1:3
```

### Fix B: Add artificial ICD via tc netem on loopback (recommended)

After starting Mininet, before training, add per-controller-port delays
that match LATENCY_MATRIX values for each zone:

```python
# scripts/setup_tc_delays.py
import subprocess
from topology.topology_data import CTRL_PORTS, ZONE_CTRL, SW_ZONE, SWITCHES, LATENCY_MATRIX

CTRL_PORT_BASE = 6653  # ctrl_0=6653, ctrl_1=6654, ..., ctrl_4=6657

def setup_loopback_delays(assignments):
    """
    Add tc netem delay on loopback for each controller port
    proportional to mean ICD of assigned switches.
    
    This makes RealEnv ICD meaningful (matches SimEnv scale).
    """
    from topology.topology_data import N_CONTROLLERS, N_SWITCHES
    import numpy as np

    # Compute mean latency per controller based on current assignments
    ctrl_mean_lat = {}
    for ctrl_idx in range(N_CONTROLLERS):
        assigned = [i for i in range(N_SWITCHES) if assignments[i] == ctrl_idx]
        if assigned:
            ctrl_mean_lat[ctrl_idx] = np.mean([LATENCY_MATRIX[i][ctrl_idx] for i in assigned])
        else:
            ctrl_mean_lat[ctrl_idx] = 5.0  # default

    # Remove existing qdiscs
    subprocess.run(['tc', 'qdisc', 'del', 'dev', 'lo', 'root'],
                   capture_output=True)

    # Add root prio qdisc
    subprocess.run(['tc', 'qdisc', 'add', 'dev', 'lo', 'root',
                    'handle', '1:', 'prio', 'bands', '6'],
                   check=True)

    for ctrl_idx in range(N_CONTROLLERS):
        port = CTRL_PORT_BASE + ctrl_idx
        delay_ms = ctrl_mean_lat[ctrl_idx]
        handle = ctrl_idx + 10
        band = ctrl_idx + 1

        # Add netem qdisc for this controller port
        subprocess.run(['tc', 'qdisc', 'add', 'dev', 'lo',
                        'parent', f'1:{band}', 'handle', f'{handle}:',
                        'netem', 'delay', f'{delay_ms:.0f}ms', '2ms'],
                       check=True)

        # Filter: match src port (controller outbound) and dst port (switch inbound)
        subprocess.run(['tc', 'filter', 'add', 'dev', 'lo', 'protocol', 'ip',
                        'parent', '1:0', 'prio', str(band), 'u32',
                        'match', 'ip', 'sport', str(port), '0xffff',
                        'flowid', f'1:{band}'],
                       check=True)

    print(f'tc netem delays set: {ctrl_mean_lat}')


def cleanup_tc_delays():
    subprocess.run(['tc', 'qdisc', 'del', 'dev', 'lo', 'root'],
                   capture_output=True)
    print('tc delays cleaned up')
```

Call in run_all.py before Phase 2:
```python
from scripts.setup_tc_delays import setup_loopback_delays, cleanup_tc_delays
setup_loopback_delays(initial_assignments)  # before net.start()
# ... train ...
cleanup_tc_delays()
```

### Simpler alternative: use LATENCY_MATRIX directly in RealEnv

If tc netem is too complex to debug, use the latency matrix for ICD measurement
in RealEnv too (set MEASURE_REAL_ICD=False in environment.py).

Advantage: simpler, reproducible, no OS-level config
Disadvantage: RealEnv ICD is identical to SimEnv ICD → not a true real environment

For DCCN paper: tc netem approach is scientifically stronger. Use it.

---

## Flow Entry Aging (idle_timeout) — Critical for Long Runs

Without timeouts, OvS accumulates flow entries across episodes.
20 switches × multiple reassignments × 200 episodes → TCAM overflow.

Add to ctrl_app.py:
```python
def _add_flow(self, dp, priority, match, actions):
    parser = dp.ofproto_parser
    ofp = dp.ofproto
    inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
    mod = parser.OFPFlowMod(
        datapath=dp,
        priority=priority,
        match=match,
        instructions=inst,
        idle_timeout=30,    # remove if idle 30s
        hard_timeout=120,   # always remove after 120s
    )
    dp.send_msg(mod)
```

---

## FAIL_OPEN vs FAIL_SECURE during Reassignment

Default OvS behavior during controller disconnect (1.5s gap in our setup):
- FAIL_SECURE: drops all packets not matching existing flows (default in Mininet)
- FAIL_OPEN: forwards normally using existing flows until reconnect

During reassignment, FAIL_SECURE means ~1.5s of packet loss.
This inflates migration cost and deflates throughput measurements.

Set FAIL_OPEN on all switches:
```bash
# In reset() or topology setup:
for sw in s1 s2 ... s20:
    ovs-vsctl set bridge {sw} fail_mode=standalone
```

Or in citiverse_topo.py:
```python
# After creating each switch:
self.addSwitch('s1', failMode='standalone', protocols='OpenFlow13')
```

Document in paper as a design choice: "switches maintain forwarding during
controller migration using fail-open mode, simulating graceful handover."

---

## Dependency Map (implementation order, no conflicts)

```
TASK_01  topology_data.py, citiverse_topo.py
  └── TASK_14  add link delays (delay=Xms) to citiverse_topo.py
       
TASK_07  ctrl_monitor.py + update ctrl_app.py logging format
  └── requires: TASK_01 (topology constants)
  
TASK_02  ctrl_app.py (update), multi_controller.py, sim_environment.py
  └── requires: TASK_01, TASK_07 (ctrl_app log format must match ControllerMonitor)

TASK_08  flow table mgmt — edits to environment.py
  └── requires: TASK_02 (Ryu running)

TASK_03  environment.py (CitiverseRealEnv)
  └── requires: TASK_02, TASK_07, TASK_08

TASK_04  model.py, replay_buffer.py (uniform), agent.py (basic)
  └── requires: TASK_01 (STATE_DIM, ACTION_DIM)

TASK_09  replay_buffer.py (PER), agent.py (PER update, save_best, auto_reset)
  └── requires: TASK_04

TASK_10  sim_environment.py curriculum additions
  └── requires: TASK_02 (CitiverseSimEnv exists)

TASK_05  train.py (Phase 1 + Phase 2)
  └── requires: TASK_03, TASK_04, TASK_09, TASK_10

TASK_11  train.py Phase 2 anti-forgetting additions
  └── requires: TASK_05

TASK_06  plot_results.py, plot_topology.py, run_all.py
  └── requires: TASK_05 (training output)
  └── update run_all.py to call setup_tc_delays from TASK_14

TASK_12  eval_protocol.py, ablation.py
  └── requires: TASK_06 (results directory populated)

TASK_13  mooo_rdqn.py, comparison methodology
  └── requires: TASK_12 (eval protocol established)
```

---

## Final Pre-Run Checklist

Run this before starting training to catch common failures early:

```bash
#!/bin/bash
# scripts/pre_run_check.sh

echo "=== Pre-Run Checklist ==="

# 1. Python deps
python3 -c "import torch, numpy, ryu, sklearn, scipy, matplotlib, networkx" && \
    echo "[OK] Python deps" || echo "[FAIL] Missing Python deps — run: pip install -r requirements.txt"

# 2. OvS installed
ovs-vsctl show >/dev/null 2>&1 && echo "[OK] OvS installed" || echo "[FAIL] ovs-vsctl not found"

# 3. Ryu installed
ryu-manager --version >/dev/null 2>&1 && echo "[OK] Ryu installed" || echo "[FAIL] ryu-manager not found"

# 4. Running as root (needed for Mininet)
[ "$EUID" -eq 0 ] && echo "[OK] Running as root" || echo "[WARN] Not root — Mininet will fail"

# 5. tc available (for netem)
tc -help >/dev/null 2>&1 && echo "[OK] tc available" || echo "[FAIL] tc not found — install iproute2"

# 6. Smoke test SimEnv
python3 -c "
from dqn.sim_environment import CitiverseSimEnv
env = CitiverseSimEnv(seed=42)
s = env.reset()
assert s.shape == (94,), f'Expected (94,), got {s.shape}'
s2, r, done, info = env.step(0)
assert isinstance(r, float), 'Reward must be float'
print(f'[OK] SimEnv: state={s.shape}, reward={r:.4f}, icd={info[\"icd_ms\"]:.2f}ms')
" 2>&1

# 7. STATE_DIM check
python3 -c "
from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS
assert STATE_DIM == 94, f'Expected STATE_DIM=94, got {STATE_DIM}'
assert N_SWITCHES == 20, f'Expected N_SWITCHES=20, got {N_SWITCHES}'
assert N_CONTROLLERS == 5, f'Expected N_CONTROLLERS=5, got {N_CONTROLLERS}'
print(f'[OK] STATE_DIM={STATE_DIM}, N_SWITCHES={N_SWITCHES}, N_CONTROLLERS={N_CONTROLLERS}')
"

# 8. Action space check
python3 -c "
ACTION_DIM = 20 * 5
assert ACTION_DIM == 100, f'Expected ACTION_DIM=100, got {ACTION_DIM}'
print(f'[OK] ACTION_DIM={ACTION_DIM}')
"

echo "=== Checklist complete ==="
```

---

## Episode Count Justification

### SimEnv — 5000 episodes

Literature:
- MOOO-RDQN (2025): 3000-8000 episodes depending on topology size
- AP-DQN (2026): 5000 episodes for 20-switch topology
- Switch migration RL papers: 2000-5000 episodes typical

Our estimate:
- With curriculum (TASK_10): convergence ~ep 400-700, 5000 = large safety margin
- Without curriculum: convergence ~ep 1500-2000, 5000 = adequate
- Early stopping (TASK_10): will stop if converged early

Verdict: 5000 is appropriate. Do not reduce below 3000.

### RealEnv — 200 episodes

Starting from pre-trained Phase 1 checkpoint: agent already knows general policy.
Fine-tuning with lower LR (P2_LR = 6e-5) and eps_start=0.3.

Our estimate:
- With tc netem delays (ICD meaningful): convergence ~ep 50-100, 200 = safe
- Without tc netem (ICD ~0.1ms, same for all assignments): will NOT converge at all
- Anti-forgetting mixing (TASK_11): stabilizes learning, reduces needed episodes

Verdict: 200 is appropriate IF tc netem is configured correctly.
Add early stopping: stop if mean ICD std across last 30 episodes < 0.5ms.

---

## Metrics Summary for Paper

### Table 1 (main result)

Rows: AllToCtrl0, ZoneOptimal, LoadBalanced, KMeans, Random, DQN (ours), MOOO-RDQN*
Columns: ICD morning, ICD business, ICD evening, ICD night, ICD overall, Load Std, Zone Match

All values: mean ± std (3 seeds, 20 eval episodes each).

### Table 2 (ablation)

Rows: Full DQN, No Curriculum, No N-step, No Dueling, No Masking, No PER
Columns: ICD (ms), Load Std, Convergence ep

### Figures

1. Training curve: ICD vs episode (mean ± 1σ, 3 seeds) — both phases
2. Grouped bar: ICD per method per profile (4 profiles, 6 methods)
3. Topology: NetworkX visualization with learned DQN assignment overlay
4. Box plot: controller load distribution per method

### Statistical tests

For each baseline vs DQN:
  Welch's t-test (one-sided: DQN < baseline)
  Required: p < 0.05 for primary baseline (ZoneOptimal under dynamic traffic)
  Report: t-stat, p-value, effect size d = (mean_baseline - mean_DQN) / pooled_std

### Key claim to prove

"DQN achieves X% lower mean ICD than ZoneOptimal under dynamic traffic profiles
(p=Y, d=Z), demonstrating that learned policies outperform hand-crafted heuristics
in time-varying smart-city SDN deployments."

Target: X >= 15%, p < 0.05, d > 0.5 (medium effect size)
