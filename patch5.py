import io, sys
ROOT = '/home/maksim/simenv_dqn_mininet_v2'

def edit(path, repls):
    with open(path) as f:
        src = f.read()
    for old, new in repls:
        c = src.count(old)
        assert c == 1, f'EXPECTED 1 match in {path}, got {c} for:\n---\n{old}\n---'
        src = src.replace(old, new)
    with open(path, 'w') as f:
        f.write(src)
    print('PATCHED', path)

# ---------------------------------------------------------------------------
# A. dqn/sim_environment.py  — compress curriculum so Stage 3 (full topology)
#    actually runs and gets the bulk of episodes.
# ---------------------------------------------------------------------------
edit(ROOT + '/dqn/sim_environment.py', [
("""    _STAGES = [
        (500,  5,  2, ['morning']),
        (2000, 10, 3, ['morning', 'business']),
        (5000, 20, 5, ['morning', 'business', 'evening', 'night']),
    ]""",
"""    _STAGES = [
        (300,  5,  2, ['morning']),
        (800,  10, 3, ['morning', 'business']),
        (3000, 20, 5, ['morning', 'business', 'evening', 'night']),
    ]"""),
("""        if ep < 500:  return 0
        if ep < 2000: return 1""",
"""        if ep < 300:  return 0
        if ep < 800:  return 1"""),
])

# ---------------------------------------------------------------------------
# B. dqn/train.py — fewer episodes (eps now decays within training),
#    matching curriculum log, DISABLE the premature loss-based early stop,
#    and record per-profile load_std (2nd metric) in evaluate_agent.
# ---------------------------------------------------------------------------
edit(ROOT + '/dqn/train.py', [
("P1_EPISODES        = 5000", "P1_EPISODES        = 3000"),
("        if ep in (500, 2000):", "        if ep in (300, 800):"),
("""        # Early stopping
        if loss is not None:
            if loss < best_loss - 0.01:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 200 and loss < 0.2:
                LOG.info(f'[P1 seed={seed}] Early stop at ep={ep}, loss={loss:.4f}')
                break""",
"""        # Early stopping DISABLED. The loss-based stop fired around ep~1000,
        # BEFORE curriculum Stage 3 (full 20sw/5ctrl topology) and while eps
        # was still ~0.77 (policy mostly random) -> training never learned the
        # real problem. Fixed P1_EPISODES bounds the runtime instead.
        _ = (loss, best_loss, patience_counter)"""),
("""    icd_list, reward_list, load_std_list = [], [], []
    profile_icds = {p: [] for p in PROFILES}""",
"""    icd_list, reward_list, load_std_list = [], [], []
    profile_icds = {p: [] for p in PROFILES}
    profile_stds = {p: [] for p in PROFILES}"""),
("""        load_std_list.append(float(np.mean(ep_std)))
        profile_icds[profile].append(ep_mean_icd)""",
"""        load_std_list.append(float(np.mean(ep_std)))
        profile_icds[profile].append(ep_mean_icd)
        profile_stds[profile].append(float(np.mean(ep_std)))"""),
("""        'icd_per_profile': {
            p: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for p, v in profile_icds.items() if v
        },""",
"""        'icd_per_profile': {
            p: {'mean': float(np.mean(v)), 'std': float(np.std(v)),
                'load_std': float(np.mean(profile_stds[p])) if profile_stds[p] else 0.0}
            for p, v in profile_icds.items() if v
        },"""),
])

# ---------------------------------------------------------------------------
# C. watcher.py — record controller load_std per profile in baselines, and
#    surface DQN-vs-ZoneOptimal load balance in summary.json (2nd metric).
# ---------------------------------------------------------------------------
edit(ROOT + '/watcher.py', [
("""        per_profile = {p: {'icd_mean': round(_icd(asgn, p), 3), 'icd_std': 0.0}
                       for p in TRAFFIC_PROFILES}""",
"""        per_profile = {}
        for p in TRAFFIC_PROFILES:
            Lp = _loads(asgn, p)
            per_profile[p] = {'icd_mean': round(_icd(asgn, p), 3), 'icd_std': 0.0,
                              'load_std': round(float(np.std(Lp)), 3)}"""),
("""    zo_pp = {p: zo.get('per_profile', {}).get(p, {}).get('icd_mean') for p in PROFILES}""",
"""    zo_pp = {p: zo.get('per_profile', {}).get(p, {}).get('icd_mean') for p in PROFILES}
    zo_ls = {p: zo.get('per_profile', {}).get(p, {}).get('load_std') for p in PROFILES}"""),
("""    dqn_overall, dqn_pp = [], {p: [] for p in PROFILES}
    for s in SEEDS:
        try:
            with open(f'{RESULTS_DIR}/p2_eval_seed{s}.json') as f:
                m = json.load(f)
            if m.get('icd_mean') is not None:
                dqn_overall.append(m['icd_mean'])
            for p in PROFILES:
                v = m.get('icd_per_profile', {}).get(p, {}).get('mean')
                if v is not None:
                    dqn_pp[p].append(v)
        except Exception:
            pass""",
"""    dqn_overall, dqn_pp = [], {p: [] for p in PROFILES}
    dqn_ls = {p: [] for p in PROFILES}
    for s in SEEDS:
        try:
            with open(f'{RESULTS_DIR}/p2_eval_seed{s}.json') as f:
                m = json.load(f)
            if m.get('icd_mean') is not None:
                dqn_overall.append(m['icd_mean'])
            for p in PROFILES:
                v = m.get('icd_per_profile', {}).get(p, {}).get('mean')
                if v is not None:
                    dqn_pp[p].append(v)
                ls = m.get('icd_per_profile', {}).get(p, {}).get('load_std')
                if ls is not None:
                    dqn_ls[p].append(ls)
        except Exception:
            pass"""),
("""                per_profile[p] = {
                    'dqn_icd': round(dm, 3),
                    'zone_optimal_icd': round(zo_pp[p], 3),
                    'improvement_pct': round((zo_pp[p] - dm) / zo_pp[p] * 100, 1),
                }""",
"""                entry = {
                    'dqn_icd': round(dm, 3),
                    'zone_optimal_icd': round(zo_pp[p], 3),
                    'improvement_pct': round((zo_pp[p] - dm) / zo_pp[p] * 100, 1),
                }
                if dqn_ls[p] and zo_ls[p] is not None:
                    entry['dqn_load_std'] = round(float(np.mean(dqn_ls[p])), 3)
                    entry['zone_optimal_load_std'] = round(float(zo_ls[p]), 3)
                per_profile[p] = entry"""),
])

print('ALL PATCHES OK')
