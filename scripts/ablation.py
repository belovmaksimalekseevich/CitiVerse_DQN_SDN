# scripts/ablation.py
"""
Ablation study: contribution of each DQN component.
All variants trained on SimEnv only for speed (~2000 ep each).
"""
import os
import json
import logging
import numpy as np

LOG = logging.getLogger(__name__)

ABLATION_EPISODES = 2000
ABLATION_SEED = 42
RESULTS_DIR = 'results'

VARIANTS = {
    'Full_DQN':      dict(curriculum=True,  n_step=3, masking=True),
    'No_Curriculum': dict(curriculum=False, n_step=3, masking=True),
    'No_NStep':      dict(curriculum=True,  n_step=1, masking=True),
    'No_Masking':    dict(curriculum=True,  n_step=3, masking=False),
}


def run_ablation():
    from dqn.sim_environment import CitiverseSimEnv
    from dqn.agent import DQNAgent
    from topology.topology_data import STATE_DIM, N_SWITCHES, N_CONTROLLERS

    ACTION_DIM = N_SWITCHES * N_CONTROLLERS
    results = {}

    for variant_name, config in VARIANTS.items():
        LOG.info(f'Ablation variant: {variant_name}')
        env = CitiverseSimEnv(seed=ABLATION_SEED, curriculum=config['curriculum'])
        total_steps = ABLATION_EPISODES * 200

        agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            lr=3e-4,
            gamma=0.99,
            batch_size=256,
            n_step=config['n_step'],
            total_steps=total_steps,
            eps_start=1.0,
            eps_end=0.05,
            checkpoint_path=f'{RESULTS_DIR}/ablation_{variant_name}.pth',
        )

        ep_icds, ep_rewards = [], []
        for ep in range(ABLATION_EPISODES):
            env.set_episode(ep)
            state = env.reset()
            ep_icd, ep_r = [], 0.0
            for _ in range(200):
                mask = env.get_action_mask() if config['masking'] else None
                action = agent.select_action(state, action_mask=mask)
                next_s, r, done, info = env.step(action)
                agent.push(state, action, r, next_s, done)
                agent.update()
                state = next_s
                ep_r += r
                ep_icd.append(info['icd_ms'])
                if done:
                    break
            ep_icds.append(float(np.mean(ep_icd)))
            ep_rewards.append(ep_r)
            agent.record_episode_reward(ep_r)

        final_icds = ep_icds[-200:]
        # convergence episode: first time rolling mean drops below 35ms
        roll = np.convolve(ep_icds, np.ones(50) / 50, mode='valid')
        conv_ep = int(np.argmax(roll < 35)) if np.any(roll < 35) else ABLATION_EPISODES

        results[variant_name] = {
            'icd_mean': float(np.mean(final_icds)),
            'icd_std':  float(np.std(final_icds)),
            'conv_ep':  conv_ep,
        }
        LOG.info(f'{variant_name}: ICD={results[variant_name]["icd_mean"]:.2f}±'
                 f'{results[variant_name]["icd_std"]:.2f}ms conv@{conv_ep}')

    # No_PER: requires modifying replay buffer — document as manual
    results['No_PER'] = {
        'note': 'Requires uniform ReplayBuffer — run separately with modified agent'
    }
    results['No_SimPretrain'] = {
        'note': 'Phase 2 only (no P1 pretraining) — run separately with RealEnv'
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/ablation.json', 'w') as f:
        json.dump(results, f, indent=2)

    print('\n=== TABLE 2: Ablation Study ===')
    print(f'{"Variant":<22} {"ICD (ms)":<18} {"Conv ep"}')
    print('-' * 54)
    for name, res in results.items():
        if 'note' in res:
            print(f'{name:<22} {res["note"]}')
        else:
            print(f'{name:<22} {res["icd_mean"]:.2f}±{res["icd_std"]:.2f}           {res["conv_ep"]}')

    return results


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    sys.path.insert(0, '/home/maksim/dqn_simenv_mininet')
    run_ablation()
