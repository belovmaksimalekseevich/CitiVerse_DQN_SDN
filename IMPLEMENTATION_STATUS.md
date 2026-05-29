# CitiVerse DQN — Implementation Status

**Обновлено:** 2026-05-27 10:30 МСК  
**Состояние:** ВСЕ ФАЙЛЫ НАПИСАНЫ — СИМУЛЯЦИЯ ЗАПУЩЕНА

---

## Сервер
- Host: 111.88.252.13 | User: maksim | Key: ~/.ssh/ssh-key-1779457478677
- Project: /home/maksim/dqn_simenv_mininet
- Python env: /home/maksim/dqn_env

---

## Симуляция

**watcher.py запущен:** 2026-05-27 07:19 UTC (PID ~1226725)  
**Лог:** logs/watcher_run.log  
**Текущая фаза:** DQN Фаза 1 (SimEnv) — 3 параллельных запуска

### Прогресс DQN Фаза 1
| Запуск | seed | Статус | Прогресс |
|--------|------|--------|---------|
| 1 | 42 | RUNNING | ep ~200/5000 |
| 2 | 123 | RUNNING | ep ~200/5000 |
| 3 | 456 | RUNNING | ep ~200/5000 |

**ETA Фаза 1:** ~15:00 МСК  
**ETA Фаза 2 (Mininet):** ~15:00-21:00 МСК  
**ETA Готово:** ~22:00 МСК

---

## Все файлы (21 шт)

### Core — DONE
| Файл | Назначение |
|------|-----------|
| topology/topology_data.py | STATE_DIM=94, LATENCY_MATRIX, зоны |
| topology/citiverse_topo.py | Mininet топология |
| ryu_apps/ctrl_monitor.py | Мониторинг RTT из логов Ryu |
| ryu_apps/ctrl_app.py | Ryu app, LLDP filter, barrier |
| ryu_apps/multi_controller.py | 5 Ryu процессов, порты 6653-6657 |
| dqn/model.py | DuelingDQN + LayerNorm |
| dqn/replay_buffer.py | SumTree + PER + NStep(3) |
| dqn/agent.py | DQNAgent, Double DQN, save_best, auto_reset |
| dqn/sim_environment.py | CitiverseSimEnv + curriculum 3 stages |
| dqn/environment.py | CitiverseRealEnv + anti-forgetting mix |
| dqn/train.py | train_phase1, train_phase2, evaluate_agent |
| baselines/run_baselines.py | 5 статических бейслайнов |
| scripts/setup_tc_delays.py | tc netem delays на loopback |
| scripts/pre_run_check.sh | Pre-flight (12/12 PASS) |
| watcher.py | Автономный оркестратор |

### Paper/Optional — DONE
| Файл | Назначение |
|------|-----------|
| scripts/plot_results.py | 6 фигур + 2 таблицы |
| scripts/plot_topology.py | NetworkX визуализация |
| scripts/eval_protocol.py | 3-seed × 4-profile eval + Welch t-test |
| scripts/ablation.py | Аблейшн (4 варианта) |
| scripts/check_success_criteria.py | PASS/FAIL критерии |

### Убран
| Файл | Причина |
|------|---------|
| ~~baselines/mooo_rdqn.py~~ | Без recurrent слоя — нечестное сравнение |

---

## Изменения watcher.py (важно знать)
-  — было 7200 (приводило к краши Phase 1 после 2ч)
-  флаг — для рестарта когда Phase 1 уже готова
- MOOO-RDQN удалён из 

---

## Curriculum stages (DQN Фаза 1)
- **Stage 1** (ep 0-499): 5sw, 2ctrl, только 'morning' — лёгкая задача
- **Stage 2** (ep 500-1999): 10sw, 3ctrl, morning+business — сложнее, reward временно падает (НОРМА)
- **Stage 3** (ep 2000-4999): 20sw, 5ctrl, все профили — полная задача

---

## Бейслайны (аналитические, без обучения)
| Метод | ICD (аналит.) |
|-------|--------------|
| ZoneOptimal | 3.0ms |
| AllToCtrl0 | 12.20ms |
| LoadBalanced | ~13ms |
| KMeans | ~14ms |
| Random | ~12ms |

**Цель DQN:** ICD < ZoneOptimal при динамических трафик-профилях  
**Claim статьи:** DQN < ZO в динамике, p < 0.05 (Welch t-test, 3 seed)

---

## Команды мониторинга
```bash
sudo tail -f /home/maksim/dqn_simenv_mininet/logs/watcher_run.log
ls -la /home/maksim/dqn_simenv_mininet/results/
sudo ps aux | grep python3 | grep -v grep | grep -v unattended
```
