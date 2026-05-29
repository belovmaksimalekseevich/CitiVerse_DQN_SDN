# Dynamic Clustering of Distributed SDN Controllers for CitiVerse Networks

A Deep Reinforcement Learning agent that assigns OpenFlow switches to distributed SDN
controllers **online**, and a measurement harness that validates it on a **live
Mininet + Ryu emulation** — not on synthetic numbers.

The agent learns control-plane load balancing: under a skewed traffic profile it spreads
the load of the busy zones across all controllers instead of overloading a few. On the
emulated testbed this lowers the measured **flow-setup latency** of the morning profile
from **419 ms** (zone-static assignment) to **38 ms**.

## Headline result (measured on Mininet + Ryu, mean over 3 seeds)

| Profile  | ZoneOptimal | LoadBalanced | KMeans | **DQN** |
|----------|------------:|-------------:|-------:|--------:|
| Morning  | 418.8 ms    | 206.7 ms     | 136.9 ms | **38.0 ± 19.0 ms** |
| Business | **14.5 ms** | 16.2 ms      | 89.5 ms  | 23.5 ± 4.8 ms |
| Evening  | 28.5 ms     | 32.7 ms      | 26.8 ms  | **26.7 ± 3.3 ms** |
| Night    | 11.7 ms     | 13.6 ms      | 12.2 ms  | **11.5 ± 0.8 ms** |

Flow-setup latency (lower is better). The agent dominates under load skew (morning) and
stays competitive elsewhere.

## Approach

- **Topology.** 20 Open vSwitch nodes (OpenFlow 1.3) in 5 zones, served by 5 Ryu
  controllers (ports 6653–6657). 40 hosts (2 per switch). See `paper/figures/`.
- **Agent.** Dueling **Double** DQN + Prioritized Experience Replay + 3-step returns +
  action masking + LayerNorm. State 94-dim, 100 discrete actions, reward `−ICD/20`.
- **Training (sim-to-real).** ≈1.8M steps would be infeasible on live Mininet, so the
  agent is trained in a fast analytical M/M/1 queuing environment with a 3-stage
  curriculum, then **transferred and measured** on the live emulation.
- **Measurement.** Switches run in `secure` fail mode; a raw-socket flooder (varying
  src MAC) drives real PACKET_IN load to a switch's controller; flow-setup latency is
  timed from a fresh ping. Baselines: ZoneOptimal, LoadBalanced, KMeans.

## Repository layout

```
dqn/                  DQN agent, model, replay buffer, training, simulation environment
topology/             topology_data.py — zones, switches, controllers, delays, profiles
baselines/            ZoneOptimal / LoadBalanced / KMeans assignment baselines
ryu_apps/             ctrl_app.py — reactive L2 controller app (PACKET_IN -> FLOW_MOD)
measure_realnet.py    live Mininet+Ryu measurement of flow-setup latency
extract_dqn_assignments.py   dump trained-agent assignments per profile
overnight_eval.py     analytical fallback summary (training-environment sanity check)
plot_results.py       result figures;  make_paper_figs.py / make_topology_v2.py — paper figures
watcher.py            full training+evaluation pipeline driver
results/              trained checkpoints (p1_seed*.pth), summaries, figures
paper/                article (.docx), METHODS, topology .drawio, all figures
TASK_*.md             development notes / design log
```

## Reproduce

System dependencies (not pip): **Mininet**, **Open vSwitch**, **Ryu** require Linux.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. train (3 seeds, curriculum) + evaluate
python3 watcher.py

# 2. extract trained assignments per profile
python3 extract_dqn_assignments.py --out results/dqn_assignments.json

# 3. measure flow-setup latency on live Mininet+Ryu (needs sudo)
sudo python3 measure_realnet.py --profiles morning,business,evening,night \
     --dqn-assignments results/dqn_assignments.json --base-pps 130 --reps 2 \
     --out results/measured_summary.json

# 4. build figures
python3 plot_results.py && python3 make_paper_figs.py
```

## Paper

The manuscript and methodology write-up are in `paper/`
(`Dynamic_clustering_SDN_DQN_CitiVerse_measured.docx`,
`METHODS_experimental_setup.md`). The network topology figure is editable in
`paper/CitiVerse_topology.drawio` (open with app.diagrams.net).
