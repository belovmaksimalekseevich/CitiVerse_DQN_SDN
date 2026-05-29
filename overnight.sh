#!/bin/bash
# Overnight orchestrator.
#  1. wait until Phase 1 training finishes ("Phase 1 done" in watcher_v3.log)
#  2. stop the watcher (skip the historically-flaky old analytical Phase 2 Mininet)
#  3. analytical FALLBACK summary (SimEnv only -> always succeeds)   -> analytical_summary.json
#  4. extract trained DQN assignments                                -> dqn_assignments.json
#  5. MEASURED Mininet+Ryu eval (free cores), hard 100-min cap        -> measured_summary.json
cd /home/maksim/simenv_dqn_mininet_v2
LOG=logs/measured.log
PY=/home/maksim/dqn_env/bin/python3
export PYTHONPATH=/usr/lib/python3/dist-packages:/home/maksim/simenv_dqn_mininet_v2

echo "[overnight] start $(date -u). Waiting for Phase 1 to finish..." > $LOG
while ! grep -q 'Phase 1 done' logs/watcher_v3.log 2>/dev/null; do
    if ! pgrep -f 'python3 -u watcher.py' >/dev/null 2>&1; then
        echo "[overnight] watcher died before 'Phase 1 done' at $(date -u)" >> $LOG
        break
    fi
    sleep 60
done
echo "[overnight] Phase 1 complete at $(date -u). Stopping watcher + old Phase 2." >> $LOG

# stop watcher and any ryu/mininet it started, free cores+ports
sudo pkill -9 -f 'python3 -u watcher.py' 2>/dev/null
sleep 3
sudo pkill -9 -f 'ofp-tcp-listen-port 665' 2>/dev/null
sudo pkill -9 -f flooder.py 2>/dev/null
sudo pkill -9 -f measure_realnet 2>/dev/null
sudo mn -c >> $LOG 2>&1
sleep 5
echo "[overnight] loadavg: $(cat /proc/loadavg)" >> $LOG

# 3) analytical fallback (reliable, SimEnv)
echo "[overnight] analytical fallback $(date -u)..." >> $LOG
$PY overnight_eval.py >> $LOG 2>&1
echo "[overnight] fallback rc=$?" >> $LOG

# 4) extract trained DQN assignments
echo "[overnight] extract DQN assignments $(date -u)..." >> $LOG
$PY extract_dqn_assignments.py --out results/dqn_assignments.json >> $LOG 2>&1
echo "[overnight] extract rc=$?" >> $LOG

# 5) MEASURED eval on live Mininet (secure mode), hard cap 100 min
echo "[overnight] MEASURED eval start $(date -u)..." >> $LOG
sudo timeout 6000 env PYTHONPATH=$PYTHONPATH $PY measure_realnet.py \
    --profiles morning,business,evening,night \
    --dqn-assignments results/dqn_assignments.json \
    --base-pps 130 --reps 2 \
    --out results/measured_summary.json >> $LOG 2>&1
echo "[overnight] measure rc=$? at $(date -u)" >> $LOG

sudo pkill -9 -f 'ofp-tcp-listen-port 665' 2>/dev/null
sudo pkill -9 -f flooder.py 2>/dev/null
sudo mn -c >> $LOG 2>&1
echo "[overnight] ALL DONE $(date -u)" >> $LOG
