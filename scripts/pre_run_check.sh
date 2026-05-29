#!/bin/bash
# scripts/pre_run_check.sh
# Pre-flight smoke tests before launching watcher.py
# Run as: sudo bash scripts/pre_run_check.sh

set -e
cd /home/maksim/dqn_simenv_mininet
source /home/maksim/dqn_env/bin/activate

PASS=0
FAIL=0

ok()   { echo "[OK]   $1"; PASS=$((PASS+1)); }
fail() { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "[WARN] $1"; }

echo "========================================"
echo " CitiVerse DQN — Pre-Run Checklist"
echo "========================================"

# 1. Python dependencies
python3 -c "import torch, numpy, sklearn, scipy, matplotlib, networkx" 2>/dev/null \
    && ok "Python deps (torch, numpy, sklearn, scipy, matplotlib, networkx)" \
    || fail "Missing Python deps — pip install torch numpy scikit-learn scipy matplotlib networkx"

# 2. Ryu available
/home/maksim/dqn_env/bin/ryu-manager --version >/dev/null 2>&1 \
    && ok "ryu-manager found" \
    || fail "ryu-manager not found — pip install ryu"

# 3. OvS installed
ovs-vsctl show >/dev/null 2>&1 \
    && ok "ovs-vsctl found" \
    || fail "ovs-vsctl not found — apt install openvswitch-switch"

# 4. Running as root
[ "$EUID" -eq 0 ] \
    && ok "Running as root (required for Mininet + ovs-vsctl)" \
    || warn "Not root — Phase 2 (RealEnv) will fail. Run: sudo bash $0"

# 5. tc available
tc help >/dev/null 2>&1 \
    && ok "tc (iproute2) found" \
    || fail "tc not found — apt install iproute2"

# 6. topology_data constants
python3 -c "
from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS, ACTION_DIM
assert STATE_DIM == 94,       f'STATE_DIM={STATE_DIM} != 94'
assert N_SWITCHES == 20,      f'N_SWITCHES={N_SWITCHES} != 20'
assert N_CONTROLLERS == 5,    f'N_CONTROLLERS={N_CONTROLLERS} != 5'
assert ACTION_DIM == 100,     f'ACTION_DIM={ACTION_DIM} != 100'
print(f'  STATE_DIM={STATE_DIM} ACTION_DIM={ACTION_DIM} N_SWITCHES={N_SWITCHES} N_CONTROLLERS={N_CONTROLLERS}')
" 2>/dev/null \
    && ok "topology_data constants (STATE_DIM=94, ACTION_DIM=100)" \
    || fail "topology_data constants mismatch"

# 7. SimEnv smoke test
python3 -c "
from dqn.sim_environment import CitiverseSimEnv
import numpy as np
env = CitiverseSimEnv(seed=42)
s = env.reset()
assert s.shape == (94,), f'state shape {s.shape}'
s2, r, done, info = env.step(0)
assert isinstance(r, float)
print(f'  state={s.shape} reward={r:.4f} icd={info[\"icd_ms\"]:.2f}ms')
" 2>/dev/null \
    && ok "CitiverseSimEnv step" \
    || fail "CitiverseSimEnv step failed"

# 8. DQN components
python3 -c "
import torch, numpy as np
from dqn.model import DuelingDQN
from dqn.replay_buffer import PrioritizedReplayBuffer
from dqn.agent import DQNAgent
m = DuelingDQN()
q = m(torch.randn(2, 94))
assert q.shape == (2, 100)
agent = DQNAgent(total_steps=100)
s = np.zeros(94, np.float32)
for i in range(300): agent.push(s, 0, 0.1, s, False)
loss = agent.update()
assert loss is not None
print(f'  DuelingDQN Q={q.shape} PER update loss={loss:.4f}')
" 2>/dev/null \
    && ok "DQN components (model, PER, agent)" \
    || fail "DQN components failed"

# 9. baselines importable
python3 -c "
from baselines.run_baselines import run_all_baselines
results = run_all_baselines(save=False)
zo_icd = results['ZoneOptimal']['icd_ms']
print(f'  ZoneOptimal ICD={zo_icd}ms (should be ~3.0)')
assert zo_icd < 10.0, f'ZoneOptimal ICD too high: {zo_icd}'
" 2>/dev/null \
    && ok "Baselines (ZoneOptimal ICD ~3ms)" \
    || fail "Baselines failed"

# 10. train.py importable
python3 -c "
from dqn.train import train_phase1, train_phase2, SEEDS, P1_EPISODES, P2_EPISODES
print(f'  SEEDS={SEEDS} P1={P1_EPISODES}ep P2={P2_EPISODES}ep')
" 2>/dev/null \
    && ok "train.py (Phase1+Phase2 functions)" \
    || fail "train.py import failed"

# 11. results directory writable
mkdir -p results && touch results/.test && rm results/.test \
    && ok "results/ directory writable" \
    || fail "results/ directory not writable"

# 12. watcher.py present
[ -f watcher.py ] \
    && ok "watcher.py found" \
    || fail "watcher.py NOT FOUND — write it first"

echo "========================================"
echo " Results: ${PASS} passed, ${FAIL} failed"
echo "========================================"

if [ "$FAIL" -gt 0 ]; then
    echo "Fix failures before launching watcher.py"
    exit 1
else
    echo "All checks passed. Ready to run: sudo python3 watcher.py"
    exit 0
fi
