# TASK_07: ControllerMonitor Thread (implements before TASK_03 RealEnv)

## Context

TASK_02 wrote a basic `ctrl_app.py` that tracks `packet_in_rates` as a dict inside the Ryu app.
The problem: CitiverseRealEnv (TASK_03) cannot directly call methods on a Ryu app object — Ryu
runs in its own eventlet thread. We need a shared, thread-safe monitor that:

1. Reads Ryu log files (written per-controller to /tmp/ryu_logs/ctrl_N.log)
2. Computes real ICD from PACKET_IN → FlowMod round-trip time
3. Exposes PACKET_IN rate per controller (for state[84:89])

This replaces the placeholder `_measure_packet_in_rates` in TASK_03.

---

## File: ryu_apps/ctrl_monitor.py

```python
# ryu_apps/ctrl_monitor.py
import threading, time, re, logging
from collections import defaultdict, deque

LOG = logging.getLogger(__name__)

N_CONTROLLERS = 5
LOG_DIR = '/tmp/ryu_logs'
WINDOW_SEC = 10.0     # PACKET_IN rate window
RTT_HISTORY = 50      # how many RTT samples to keep per switch


class ControllerMonitor:
    """
    Thread-safe monitor. Reads Ryu log files to compute:
      - PACKET_IN rate per controller (events / WINDOW_SEC)
      - Per-switch FlowMod RTT (approx ICD) from log timestamps
    
    Log line format written by ctrl_app.py:
      PACKET_IN  {dpid}  {timestamp_float}
      FLOW_MOD   {dpid}  {timestamp_float}
    """

    def __init__(self, log_dir=LOG_DIR, poll_interval=1.0):
        self.log_dir = log_dir
        self.poll_interval = poll_interval
        self._lock = threading.Lock()

        # Per-controller: deque of (timestamp,) for PACKET_IN events
        self._pkt_in_times = [deque() for _ in range(N_CONTROLLERS)]
        # Per-switch dpid: deque of RTT samples (seconds)
        self._rtt_samples = defaultdict(lambda: deque(maxlen=RTT_HISTORY))
        # File offsets (to avoid re-reading from start each poll)
        self._file_offsets = [0] * N_CONTROLLERS

        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        LOG.info('ControllerMonitor started')

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ------------------------------------------------------------------
    def get_packet_in_rates(self):
        """Returns np.array shape (N_CONTROLLERS,) — events per second."""
        import numpy as np
        rates = np.zeros(N_CONTROLLERS, dtype=np.float32)
        now = time.time()
        cutoff = now - WINDOW_SEC
        with self._lock:
            for i, dq in enumerate(self._pkt_in_times):
                # drop old events
                while dq and dq[0] < cutoff:
                    dq.popleft()
                rates[i] = len(dq) / WINDOW_SEC
        return rates

    def get_mean_rtt_ms(self, dpid=None):
        """
        Returns mean RTT in ms for a specific switch dpid, or average across all.
        Falls back to 0 if no samples.
        """
        with self._lock:
            if dpid is not None:
                samples = list(self._rtt_samples.get(dpid, []))
            else:
                all_samples = []
                for dq in self._rtt_samples.values():
                    all_samples.extend(dq)
                samples = all_samples
        if not samples:
            return 0.0
        return float(sum(samples) / len(samples)) * 1000.0  # ms

    def get_switch_rtt_ms(self, ctrl_idx):
        """Mean RTT (ms) for all switches currently assigned to ctrl_idx."""
        # This is called from RealEnv — it has assignment info
        # For now returns average across all known RTTs for that ctrl log
        with self._lock:
            all_samples = []
            for dq in self._rtt_samples.values():
                all_samples.extend(dq)
        if not all_samples:
            return 0.0
        return float(sum(all_samples) / len(all_samples)) * 1000.0

    # ------------------------------------------------------------------
    def _poll_loop(self):
        while self._running:
            for i in range(N_CONTROLLERS):
                self._read_ctrl_log(i)
            time.sleep(self.poll_interval)

    def _read_ctrl_log(self, ctrl_idx):
        logpath = f'{self.log_dir}/ctrl_{ctrl_idx}.log'
        try:
            with open(logpath, 'r') as f:
                f.seek(self._file_offsets[ctrl_idx])
                new_lines = f.readlines()
                self._file_offsets[ctrl_idx] = f.tell()
        except FileNotFoundError:
            return

        now = time.time()
        pending_pkt_in = {}  # dpid -> timestamp of most recent PACKET_IN

        for line in new_lines:
            line = line.strip()
            # Format: "PACKET_IN 1 1716800000.123"
            m = re.match(r'PACKET_IN\s+(\d+)\s+([\d.]+)', line)
            if m:
                dpid = int(m.group(1))
                ts = float(m.group(2))
                pending_pkt_in[dpid] = ts
                with self._lock:
                    self._pkt_in_times[ctrl_idx].append(ts)
                continue

            # Format: "FLOW_MOD 1 1716800000.145"
            m = re.match(r'FLOW_MOD\s+(\d+)\s+([\d.]+)', line)
            if m:
                dpid = int(m.group(1))
                ts = float(m.group(2))
                if dpid in pending_pkt_in:
                    rtt = ts - pending_pkt_in[dpid]
                    if 0 < rtt < 5.0:  # sanity: RTT < 5s
                        with self._lock:
                            self._rtt_samples[dpid].append(rtt)
                    del pending_pkt_in[dpid]
```

---

## Update ctrl_app.py to emit structured log lines

Add these two lines to `ctrl_app.py` in the appropriate handlers:

```python
# In packet_in_handler, after getting dp and dpid:
LOG.info(f'PACKET_IN {dpid} {time.time():.6f}')

# In _add_flow (or at the send_msg call for FlowMod):
LOG.info(f'FLOW_MOD {dpid} {time.time():.6f}')
```

Also configure the Ryu app logger to write to the per-controller log file:
```python
# In MultiControllerManager.start_all(), add --log-file flag:
proc = subprocess.Popen(
    ['ryu-manager', self.app_path,
     '--ofp-tcp-listen-port', str(port),
     '--observe-links',
     '--log-file', f'{self.log_dir}/ctrl_{i}.log'],  # ADD THIS
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
```

---

## Integration with CitiverseRealEnv

In `dqn/environment.py`, replace the stub `_measure_packet_in_rates` with:

```python
# In __init__:
self.ctrl_monitor = ctrl_monitor  # pass ControllerMonitor from outside

# In reset():
# (start monitor before reset if not already running)

# In _measure_packet_in_rates():
def _measure_packet_in_rates(self):
    return self.ctrl_monitor.get_packet_in_rates()

# In _measure_icd() when MEASURE_REAL_ICD=True:
def _measure_icd(self):
    # Use mean RTT from monitor (covers all switches)
    rtt = self.ctrl_monitor.get_mean_rtt_ms()
    if rtt > 0:
        return rtt
    # Fallback to latency matrix
    return sum(LATENCY_MATRIX[i][self.assignments[i]] for i in range(N_SWITCHES)) / N_SWITCHES
```

---

## run_all.py integration

```python
from ryu_apps.ctrl_monitor import ControllerMonitor
monitor = ControllerMonitor()
monitor.start()
# ... train ...
monitor.stop()
```

---

## Verification

```bash
# After starting Ryu, check log format:
tail -f /tmp/ryu_logs/ctrl_0.log | grep -E 'PACKET_IN|FLOW_MOD'

# Should see lines like:
# PACKET_IN 1 1716800000.123456
# FLOW_MOD 1 1716800000.145678

# Test ControllerMonitor standalone:
python3 -c "
from ryu_apps.ctrl_monitor import ControllerMonitor
import time
m = ControllerMonitor()
m.start()
time.sleep(5)
print('PACKET_IN rates:', m.get_packet_in_rates())
print('Mean RTT ms:', m.get_mean_rtt_ms())
m.stop()
"
```

## Implementation order note

Implement this BEFORE finishing `dqn/environment.py` (TASK_03).
The `ControllerMonitor` instance is created in `run_all.py` and passed
into `CitiverseRealEnv.__init__()` as a parameter.
