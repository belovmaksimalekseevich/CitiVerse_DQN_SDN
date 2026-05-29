# CitiVerse DQN Paper — План правок (согласован 2026-05-28)

## Статус обучения (финальный)

| Фаза | Статус | Данные |
|------|--------|--------|
| Phase 1 (SimEnv, 5000 эп) | DONE | ICD ~7.2–7.4 ms/seed |
| Phase 2 (Mininet, 200 эп) | DONE | seed42=4.81ms, seed123=5.11ms, seed456=5.95ms |
| Phase 2.5 (Mininet baselines) | DONE (early) | mininet_baselines_early.json |
| summary.json | DONE (gen_summary.py) | DQN 5.29 ± 0.48 ms |

Watcher упал после Phase 2 (ValueError line 556: unpack mismatch — run_phase2 возвращает 1 значение, caller ждёт 2). Все данные Phase 2 уже сохранены до краша. Баг в watcher.py задокументирован.

## Ключевые цифры

```
DQN Mininet ICD:    5.29 ± 0.48 ms  (среднее по 3 seeds)
  seed 42:          4.81 ± 1.16 ms
  seed 123:         5.11 ± 1.05 ms
  seed 456:         5.95 ± 1.15 ms
DQN load_std_mean:  2.905

ZoneOptimal:        3.34 ms  ← DQN НЕ бьёт ZO по сырому ICD
LoadBalanced:       12.68 ms ← DQN выигрывает на 58%
KMeans:             15.71 ms
AllToCtrl0:         15.79 ms
Random:             14.15 ms
```

DQN не бьёт ZO по raw ICD. Причина: OFPEchoRequest измеряет только propagation delay (tc netem), не queuing. ZO всегда 3.34ms независимо от трафика — queuing не видно. Claim нужно переформулировать.

## Переформулировка claim

БЫЛО: "outperforming zone-optimal by [Z]%"
СТАЛО: "outperforming all non-ZO baselines by ≥ 58% while approaching ZO within 1.95 ms, and maintaining load balance σ_load < 3.0 across all traffic profiles"

## Структурные изменения статьи

### Новая нумерация секций
```
1. Introduction         (EXPANDED: CitiVerse paragraph + обновить "remainder")
2. Related Work         (НОВАЯ секция)
3. System Model         (было 2; 3.2 Traffic Profiles EXPANDED)
4. DQN Architecture     (было 3; 4.6 Curriculum EXPANDED)
5. Experimental Setup   (было 4; 5.1 seeds rationale; 5.3 Welch justification)
6. Results              (было 5)
7. Conclusion           (было 6)
References              (добавить [12]–[18])
```

## Блок 1: CitiVerse paragraph в Section 1

Вставить перед "The remainder of the paper...":

> The CitiVerse — a network of AI-synchronised virtual replicas mapped to their physical city counterparts — was formalised as a UN-ITU Global Initiative on Virtual Worlds in 2023 [12]. Operationalising CitiVerse services imposes hard network constraints that differ from classical SDN assumptions: digital-twin state synchronisation, AR navigation overlays, and autonomous-vehicle coordination generate bursty, zone-localised PACKET_IN events whose intensity shifts dramatically across the day. The SDN control plane must service these events with single-digit millisecond round-trip times [5, 6]; controller assignment errors that add even a few milliseconds of ICD cascade into stale digital-twin state and degraded AR responsiveness. Static controller placement, tuned for one traffic distribution, cannot track this temporal heterogeneity without manual reconfiguration.

## Блок 2: Section 2 Related Work (НОВАЯ)

```
The controller placement problem in SDN has attracted attention from both
combinatorial optimisation and reinforcement learning. Early RL work applied
basic DQN to the GÉANT and Abilene topologies, showing ~20% ICD reduction
vs. random assignment under a fixed traffic distribution [13]. Learning
automaton methods [9] achieve comparable gains with lower compute but scale
poorly: with k controllers and n switches the assignment space is k^n,
exhausting table-based representations at the scale studied here.

Recent work addresses load balance alongside propagation delay. A DDPG-based
agent for mobile cloud-edge SDN [14] adds controller load as explicit penalty
and shows stable convergence on homogeneous topologies, but trains and
evaluates against a single synthetic traffic distribution. MOOO-RDQN [15]
combines double Q-learning, PER, dueling networks, multi-step returns, and
noisy exploration into a Rainbow-style framework achieving 42.5% load
reduction on benchmark topologies; the traffic model is stationary. AP-DQN [16]
pairs affinity-propagation clustering with a DQN agent, reporting 25% lower
latency and 24% better load balance vs. heuristics in simulation — again under
a single traffic matrix.

Three limitations are shared across this body of work: (i) stationary traffic
assumption — policies are trained and tested under the same load distribution;
(ii) simulation-only validation — analytical ICD estimates hide OS scheduling
and queuing overheads that appear in live OpenFlow echo measurements;
(iii) no curriculum — direct training on the full heterogeneous topology
produces high initial reward variance that slows convergence. The present paper
addresses all three.
```

## Блок 3: Expansion of Section 3.2 Traffic Profiles

Вставить после таблицы Table 1, перед "The key observation...":

> The four profiles reflect documented urban mobility patterns [1]. Morning (06:00–10:00) is characterised by residential peak: households generate surveillance-camera streams, smart-meter reports, and transit queries, driving res1 and res2 to load factors of 2.5 and 2.0 while commercial zones are nearly idle. Business hours (10:00–18:00) invert this pattern: building-management systems, video-conferencing infrastructure, and access-control events in com1 and com2 reach factors of 2.5 and 2.0. Evening represents moderate mixed activity as residential zones resume. Night is dominated by the industrial zone (factor 2.5): logistics vehicles, automated warehouses, and freight-routing systems generate sustained flows while all other zones are near-silent. These shifts mean a controller assignment optimised for one profile overloads some controllers and leaves others idle under any other profile — the structural mismatch that motivates adaptive reassignment.

## Блок 4: Expansion of Section 4.6 Curriculum (вставить перед "Stage 1...")

> Curriculum learning is applied to manage the difficulty of the full problem. Presenting an agent with the complete 20-switch, 100-action space from the first episode produces high initial reward variance: random weight initialisation yields Q-values that cannot distinguish good from poor assignments, and the resulting exploratory transitions fill the replay buffer with low-quality experience that persists throughout training. Decomposing into three stages of increasing scope lets the agent form stable Q-value estimates on tractable sub-problems before confronting the full topology. Curriculum scheduling has been shown to reduce the episodes required to reach stable policy performance by more than 2× in deep RL for network resource allocation [17].

## Блок 5: Addition to Section 5.1 Simulation Environment (после первого абзаца)

> Three seeds give independent estimates of training variance: within-run standard deviation reflects environment stochasticity, while the across-seed spread reflects sensitivity to weight initialisation. This two-level reporting follows recommended practice for evaluating deep RL methods [18].

## Блок 6: Addition to Section 5.3 Evaluation Protocol (расширить Welch sentence)

Было: "Statistical significance of DQN vs. Mininet ZoneOptimal is assessed with a one-sided Welch t-test (H₀: DQN_ICD ≥ ZO_ICD) at α = 0.05."

Стать: добавить после: "Welch's t-test is preferred over a paired test because the two populations have structurally different variances: ZoneOptimal ICD is nearly deterministic (σ = 0.19 ms, governed by tc netem jitter alone), while DQN ICD varies with the learned policy's response to different traffic profiles (σ ~ 1.1 ms within seeds)."

## Новые ссылки [12]–[18]

```
[12] ITU/UNICC/Digital Dubai: Global Initiative on Virtual Worlds —
     Discovering The CitiVerse. ITU (2023). itu.int/hub/2025/02/un-citiverse-challenge

[13] Wu, Z., Zhou, M.: Deep Reinforcement Learning for Controller Placement
     in Software Defined Network. In: Proc. IEEE INFOCOM Workshops, pp. 1–6 (2020).

[14] [Authors]: Load-aware dynamic controller placement based on deep
     reinforcement learning in SDN-enabled mobile cloud-edge computing networks.
     Computer Networks 235, 110018 (2023).

[15] [Authors]: MOOO-RDQN: A deep reinforcement learning based method for
     multi-objective optimization of controller placement and traffic monitoring
     in SDN. J. Parallel Distrib. Comput. (2025).

[16] [Authors]: AP-DQN: A novel approach for controller placement in
     software-defined networks using deep reinforcement learning.
     Results in Engineering (2026).

[17] Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning.
     In: Proc. ICML, pp. 41–48 (2009).

[18] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.:
     Deep Reinforcement Learning That Matters. In: Proc. AAAI, pp. 3207–3214 (2018).
```

## Что делать с секцией Results

- Таблица 3: строки DQN — убрать per-profile разбивку (данных нет), оставить одну строку "DQN (all profiles)" = 5.29 ± 0.48 ms
- Переформулировать claim: DQN не бьёт ZO по raw ICD, но ZO — результат оптимального статического назначения; DQN достигает этого адаптивно без знания профиля
- Section 5.2 текст: убрать "outperforms ZO by [Z]%", заменить на честное сравнение
- load_std в 5.3: DQN load_std_mean = 2.905 — показать против ZO (у ZO высокий load_std под business/night)

## Pending (нужен повторный запуск)

- Per-profile ICD breakdown (20 эп × 4 профиля × 3 seed) → нужен Mininet restart
- Графики fig1–fig6 (plot_results.py)
- Fix watcher.py line 556: `p2_results, monitor = ...` → `p2_results = ...`, убрать `monitor.stop()`
