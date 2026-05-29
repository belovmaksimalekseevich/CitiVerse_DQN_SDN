import sys, os, py_compile
ROOT = '/home/maksim/simenv_dqn_mininet_v2'
sys.path.insert(0, ROOT)
os.chdir(ROOT)

for f in ['topology/topology_data.py', 'dqn/sim_environment.py',
          'dqn/environment.py', 'dqn/train.py', 'watcher.py', 'dqn/agent.py']:
    py_compile.compile(ROOT + '/' + f, doraise=True)
print('COMPILE OK')

import numpy as np
from dqn.sim_environment import CitiverseSimEnv
from topology.topology_data import (STATE_DIM, N_SWITCHES, N_CONTROLLERS,
                                     get_zone_optimal_array)

# Curriculum boundaries: 300 / 800
env = CitiverseSimEnv(seed=1, curriculum=True)
for ep, exp in [(100, 0), (299, 0), (300, 1), (799, 1), (800, 2), (1500, 2)]:
    env.set_episode(ep)
    assert env._get_stage_id() == exp, f'stage id ep={ep} -> {env._get_stage_id()} exp {exp}'
print('stage ids OK (300/800 boundaries)')

# Stage 3 = FULL topology
env.set_episode(900); env.reset()
assert env.active_n_sw == 20 and env.active_n_ctrl == 5, (env.active_n_sw, env.active_n_ctrl)
print('ep=900 -> full topology', env.active_n_sw, 'sw', env.active_n_ctrl, 'ctrl')

# Stage 1 = reduced
env.set_episode(50); env.reset()
print('ep=50 -> stage1', env.active_n_sw, 'sw', env.active_n_ctrl, 'ctrl')

# info has load_std + icd; ZO morning ICD sanity (~20.5, metric unchanged)
env2 = CitiverseSimEnv(seed=42, curriculum=False)
env2.reset(); env2.traffic_profile = 'morning'
env2.assignments = get_zone_optimal_array().copy()
icd = env2._compute_icd()
s = env2._get_state()
_, _, _, info = env2.step(env2.get_action_mask().argmax())
assert 'load_std' in info and 'icd_ms' in info, list(info.keys())
print(f'ZO morning ICD={icd:.2f} (expect ~20.5); info has load_std={info["load_std"]:.3f}')

# eps decay now reaches ~0.05 within 3000 episodes
from dqn.agent import DQNAgent
T = 3000 * 200
ag = DQNAgent(state_dim=STATE_DIM, action_dim=N_SWITCHES * N_CONTROLLERS,
              hidden=256, total_steps=T)
# emulate stepping eps to 80% of schedule
ag.eps = max(ag.eps_end, 1.0 - ag.eps_decay * (T * 0.8))
print(f'eps after 80% of 3000-ep schedule = {ag.eps:.3f} (expect ~0.05)')

# evaluate_agent returns per-profile load_std
from dqn.train import evaluate_agent
env3 = CitiverseSimEnv(seed=7, curriculum=False)
res = evaluate_agent(ag, env3, n_episodes=4)
pp = res['icd_per_profile']
assert all('load_std' in pp[p] for p in pp), pp
print('evaluate_agent per-profile keys OK; sample:', {k: round(v['load_std'],2) for k,v in pp.items()})
print('SMOKE5 OK')
