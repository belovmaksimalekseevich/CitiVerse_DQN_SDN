#!/bin/bash
# wait_and_resume.sh
# Ждёт завершения orphan-воркеров Phase 1 (seeds 123, 456),
# затем автоматически запускает watcher.py --skip-p1 для Phase 2+.
#
# Запуск: sudo bash wait_and_resume.sh &

set -e
cd /home/maksim/dqn_simenv_mininet

PYTHON=/home/maksim/dqn_env/bin/python3
LOG=logs/resume.log
RESULTS=results

mkdir -p logs

echo "[$(date)] wait_and_resume.sh started" | tee -a "$LOG"
echo "[$(date)] Watching orphan worker PIDs: 1214888 (seed=123), 1214889 (seed=456)" | tee -a "$LOG"

# ── Ждём завершения обоих воркеров ────────────────────────────────────────
while true; do
    alive_123=false
    alive_456=false

    if kill -0 1214888 2>/dev/null; then
        alive_123=true
    fi
    if kill -0 1214889 2>/dev/null; then
        alive_456=true
    fi

    if ! $alive_123 && ! $alive_456; then
        echo "[$(date)] Both workers finished!" | tee -a "$LOG"
        break
    fi

    msg="[$(date)] Still running:"
    $alive_123 && msg="$msg seed=123(PID 1214888)"
    $alive_456 && msg="$msg seed=456(PID 1214889)"
    echo "$msg" | tee -a "$LOG"
    sleep 120
done

# ── Проверяем наличие чекпоинтов ─────────────────────────────────────────
echo "[$(date)] Checking checkpoints..." | tee -a "$LOG"
for seed in 42 123 456; do
    for name in "p1_seed${seed}.pth" "p1_best_seed${seed}.pth"; do
        if [ -f "$RESULTS/$name" ]; then
            sz=$(du -h "$RESULTS/$name" | cut -f1)
            echo "[$(date)]   seed=$seed: $RESULTS/$name ($sz)" | tee -a "$LOG"
            break
        fi
    done
done

# ── Убиваем зависший watcher.py (если ещё жив) ───────────────────────────
if kill -0 1214869 2>/dev/null; then
    echo "[$(date)] Killing stale watcher.py (PID 1214869)..." | tee -a "$LOG"
    sudo kill -9 1214869 2>/dev/null || true
    sleep 2
fi

# ── Очистка Mininet на всякий случай ─────────────────────────────────────
echo "[$(date)] Running mn --clean..." | tee -a "$LOG"
mn --clean 2>/dev/null || true
sleep 3

# ── Запускаем Phase 2 через watcher.py --skip-p1 ─────────────────────────
echo "[$(date)] Launching watcher.py --skip-p1..." | tee -a "$LOG"
nohup sudo "$PYTHON" watcher.py --skip-p1 \
    > logs/watcher_phase2.log 2>&1 &

P2_PID=$!
echo "[$(date)] watcher.py --skip-p1 launched, PID=$P2_PID" | tee -a "$LOG"
echo "[$(date)] Follow: sudo tail -f logs/watcher_phase2.log" | tee -a "$LOG"
