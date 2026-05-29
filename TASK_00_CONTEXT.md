# TASK_00: Project Context, Architecture, State Vector
# READ THIS FIRST. Then read TASK_01 through TASK_14 in order.

## Project Overview

DQN agent dynamically assigns 20 OpenFlow switches to 5 Ryu controllers
in a smart-city topology (5 zones). Goal: minimize ICD (switch-to-controller latency)
under changing traffic profiles.
Conference: DCCN 2026. Server: maksim@111.88.252.13

SCIENTIFIC CONTRIBUTION: DQN adapts to dynamic traffic (4 profiles).
Static methods cannot. WITHOUT dynamic traffic profiles the paper is unpublishable.

## Why the Old Code Failed (3 runs)

B01 FATAL: ovs-vsctl was never called — agent assignments had no effect on real network
B02: ICD was missing from state vector
B04: Zone bonus 1.5 > network penalty — agent gamed zone labels, not ICD
B05: 95% of replay buffer transitions had stale reward
B06: ICC normalization /20 made ICC 2x more important than ICD per ms
Full bug list: PLAN_AND_REVIEW.md

## File Structure

```
dqn_simenv_mininet/
├── topology/topology_data.py      Step 1
├── topology/citiverse_topo.py     Step 2
├── ryu_apps/ctrl_app.py           Step 3 (updated in TASK_07)
├── ryu_apps/multi_controller.py   Step 3
├── ryu_apps/ctrl_monitor.py       TASK_07 (NEW)
├── dqn/sim_environment.py         Step 4 (updated in TASK_10)
├── dqn/environment.py             Step 5 (updated in TASK_08)
├── dqn/model.py                   Step 6
├── dqn/replay_buffer.py           Step 7 (updated in TASK_09)
├── dqn/agent.py                   Step 8 (updated in TASK_09)
├── dqn/train.py                   Step 9 (updated in TASK_11)
├── baselines/run_baselines.py     Step 10
├── baselines/mooo_rdqn.py         TASK_13 (NEW)
├── scripts/plot_results.py        Step 11
├── scripts/plot_topology.py       Step 12
├── scripts/eval_protocol.py       TASK_12 (NEW)
├── scripts/ablation.py            TASK_12 (NEW)
├── scripts/setup_tc_delays.py     TASK_14 (NEW)
├── scripts/pre_run_check.sh       TASK_14 (NEW)
└── run_all.py                     Step 13
```

## STATE VECTOR (STATE_DIM=94)

```
[0:20]   assignment[sw] / 4.0                   normalized controller index per switch
[20:40]  ctrl_load[sw_ctrl] / MAX_CTRL_LOAD(5)  load of the switch's assigned controller
[40:60]  zone_idx[sw] / 4.0                     zone identity per switch
[60:80]  zone_match[sw]  (0.0 or 1.0)           1 if switch is in its controller's zone
[80]     icd_ms / 100.0            CRITICAL: was missing in old code (B02)
[81]     throughput_mbps / 100.0   FIX: was /1000 -> signal was 0.076..0.099 (B08)
[82]     max_ctrl_load / 5.0
[83]     load_std / 20.0
[84:89]  packet_in_rate[ctrl0..4] / 50.0        from ControllerMonitor (TASK_07)
[89:93]  traffic_profile one-hot  (morning / business / evening / night)
[93]     migration_count / 10.0
```

ICC removed entirely from state and reward. (B06/B07 — ICC was incorrectly measured
and incorrectly weighted.)

## REWARD FUNCTION (no ICC)

```python
icd_norm        = icd_ms / 100.0
thr_norm        = throughput_mbps / 100.0
load_imbalance  = np.std(ctrl_loads) / 20.0

r = -(0.6 * icd_norm + 0.2 * (1.0 - thr_norm) + 0.2 * load_imbalance)
r += zone_match_ratio * 0.2    # was 1.5 -> 0.2 (B04 fix)
r -= 0.05 * int(migrated)      # migration cost
r -= 0.3 * (load - MAX_CTRL_LOAD)  # per overloaded controller
```

## TRAFFIC PROFILES (mandatory for scientific contribution)

```
morning:  res1=2.5, res2=2.0, com1=0.5, com2=0.5, ind=0.3
business: res1=0.7, res2=0.7, com1=2.5, com2=2.0, ind=1.0
evening:  res1=2.0, res2=2.0, com1=1.5, com2=1.5, ind=0.5
night:    res1=0.3, res2=0.3, com1=0.3, com2=0.3, ind=2.5
Rotate round-robin every episode.
```

## TRAINING SCHEDULE

```
Phase 1: CitiverseSimEnv   5000 ep  ~10 min   (analytical, no Mininet)
Phase 2: CitiverseRealEnv   200 ep  ~1.7 h    (5 Ryu + Mininet + tc netem)
Baselines: 5 methods x 100 ep x 3 seeds (sequential on Mininet)
3 seeds: [42, 123, 456] — for mean+-std in paper
```

## KEY CONSTANTS

```python
N_SWITCHES    = 20
N_CONTROLLERS = 5
STATE_DIM     = 94
ACTION_DIM    = 100   # N_SWITCHES * N_CONTROLLERS (no explicit no-op action)
MAX_CTRL_LOAD = 5     # switches per controller before overload penalty
SEEDS         = [42, 123, 456]
CTRL_PORTS    = [6653, 6654, 6655, 6656, 6657]
```

## ENVIRONMENT SETUP (run once on server)

```bash
cd /home/maksim
python3 -m venv dqn_env
source dqn_env/bin/activate
pip install torch numpy ryu scikit-learn scipy matplotlib networkx netgraph

cd /home/maksim/dqn_simenv_mininet
mkdir -p topology ryu_apps dqn baselines scripts results logs
touch __init__.py topology/__init__.py ryu_apps/__init__.py
touch dqn/__init__.py baselines/__init__.py scripts/__init__.py
```

## IMPLEMENTATION ORDER (read TASK files in this sequence)

```
TASK_01  -> topology_data.py, citiverse_topo.py
TASK_07  -> ctrl_monitor.py (implement before RealEnv)
TASK_02  -> ctrl_app.py (update log format), multi_controller.py, sim_environment.py
TASK_08  -> environment.py flow table additions
TASK_03  -> environment.py (CitiverseRealEnv, B01 fix)
TASK_04  -> model.py, replay_buffer.py, agent.py
TASK_09  -> PER, save_best, auto_reset additions to agent.py
TASK_10  -> curriculum additions to sim_environment.py
TASK_05  -> train.py (Phase 1 + Phase 2)
TASK_11  -> anti-forgetting additions to train.py
TASK_06  -> plot_results.py, plot_topology.py, run_all.py
TASK_12  -> eval_protocol.py, ablation.py
TASK_13  -> mooo_rdqn.py (optional SOTA comparison)
TASK_14  -> setup_tc_delays.py, pre_run_check.sh, final checklist
```
