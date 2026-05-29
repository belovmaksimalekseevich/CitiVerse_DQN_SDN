#!/bin/bash
# Waits until overnight.sh finishes ("ALL DONE" in logs/measured.log), then builds
# figures. Independent process -> does NOT touch the already-running overnight.sh.
# Safety timeout: give up waiting after 6h, but still try to plot whatever exists.
cd /home/maksim/simenv_dqn_mininet_v2
PY=/home/maksim/dqn_env/bin/python3
LOG=logs/plots.log
export PYTHONPATH=/usr/lib/python3/dist-packages:/home/maksim/simenv_dqn_mininet_v2

echo "[plots] waiting for overnight ALL DONE $(date -u)" > $LOG
for i in $(seq 1 360); do          # 360 * 60s = 6h cap
    if grep -q 'ALL DONE' logs/measured.log 2>/dev/null; then
        echo "[plots] overnight done, plotting $(date -u)" >> $LOG
        break
    fi
    sleep 60
done
$PY plot_results.py >> $LOG 2>&1
echo "[plots] rc=$? at $(date -u)" >> $LOG
ls -la results/figures >> $LOG 2>&1
echo "[plots] FINISHED $(date -u)" >> $LOG
