# ryu_apps/ctrl_monitor.py
import threading
import time
import re
import logging
from collections import defaultdict, deque

LOG = logging.getLogger(__name__)

N_CONTROLLERS = 5
LOG_DIR = '/tmp/ryu_logs'
WINDOW_SEC = 10.0
RTT_HISTORY = 50


class ControllerMonitor:
    """
    Thread-safe monitor. Reads Ryu log files to compute:
      - PACKET_IN rate per controller (events / WINDOW_SEC)
      - Per-switch FlowMod RTT from log timestamps
      - Per-switch OFPEchoReply RTT (real measured propagation delay)

    Log line formats written by ctrl_app.py:
      PACKET_IN  {dpid}  {timestamp_float}
      FLOW_MOD   {dpid}  {timestamp_float}
      ECHO_RTT   {dpid}  {rtt_ms_float}
    """

    def __init__(self, log_dir=LOG_DIR, poll_interval=1.0):
        self.log_dir = log_dir
        self.poll_interval = poll_interval
        self._lock = threading.Lock()

        self._pkt_in_times = [deque() for _ in range(N_CONTROLLERS)]
        self._rtt_samples = defaultdict(lambda: deque(maxlen=RTT_HISTORY))
        self._echo_rtt_ms = {}        # dpid -> latest OFPEchoReply RTT in ms
        self._file_offsets = [0] * N_CONTROLLERS
        self._blackout_until = {}     # dpid -> timestamp

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

    def reset_offsets(self):
        """Reset file offsets and clear history. Call at episode start."""
        with self._lock:
            self._file_offsets = [0] * N_CONTROLLERS
            for dq in self._pkt_in_times:
                dq.clear()
            self._rtt_samples.clear()
            self._echo_rtt_ms.clear()
            self._blackout_until.clear()

    def set_blackout(self, dpid, duration=3.0):
        """Suppress measurements for dpid after reassignment (LLDP flood period)."""
        self._blackout_until[dpid] = time.time() + duration

    def get_packet_in_rates(self):
        """Returns np.array shape (N_CONTROLLERS,) — events per second."""
        import numpy as np
        rates = np.zeros(N_CONTROLLERS, dtype=np.float32)
        now = time.time()
        cutoff = now - WINDOW_SEC
        with self._lock:
            for i, dq in enumerate(self._pkt_in_times):
                while dq and dq[0] < cutoff:
                    dq.popleft()
                rates[i] = len(dq) / WINDOW_SEC
        return rates

    def get_mean_rtt_ms(self, dpid=None):
        """Mean PACKET_IN→FLOW_MOD RTT in ms. Returns 0.0 if no samples."""
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
        return float(sum(samples) / len(samples)) * 1000.0

    def get_echo_rtt_ms_per_switch(self):
        """
        Returns dict: dpid -> measured OFPEchoRequest RTT in ms.
        RTT/2 approximates one-way propagation delay (includes tc netem).
        Only populated after OFPEchoReply events are received (~5s warmup).
        """
        with self._lock:
            return dict(self._echo_rtt_ms)

    def get_switch_rtt_ms(self, ctrl_idx):
        """Mean RTT (ms) across all switches."""
        with self._lock:
            all_samples = []
            for dq in self._rtt_samples.values():
                all_samples.extend(dq)
        if not all_samples:
            return 0.0
        return float(sum(all_samples) / len(all_samples)) * 1000.0

    def get_flow_counts(self):
        """Returns dict: sw_name -> flow entry count (via ovs-ofctl)."""
        import subprocess
        counts = {}
        for i in range(1, 21):
            sw = f's{i}'
            try:
                result = subprocess.run(
                    ['ovs-ofctl', '-O', 'OpenFlow13', 'dump-flows', sw],
                    capture_output=True, text=True, timeout=3,
                )
                counts[sw] = max(0, result.stdout.count('\n') - 1)
            except Exception:
                counts[sw] = -1
        return counts

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

        pending_pkt_in = {}  # dpid -> timestamp of most recent PACKET_IN

        for line in new_lines:
            line = line.strip()

            m = re.match(r'PACKET_IN\s+(\d+)\s+([\d.]+)', line)
            if m:
                dpid = int(m.group(1))
                ts = float(m.group(2))
                if ts < self._blackout_until.get(dpid, 0):
                    continue
                pending_pkt_in[dpid] = ts
                with self._lock:
                    self._pkt_in_times[ctrl_idx].append(ts)
                continue

            m = re.match(r'FLOW_MOD\s+(\d+)\s+([\d.]+)', line)
            if m:
                dpid = int(m.group(1))
                ts = float(m.group(2))
                if dpid in pending_pkt_in:
                    rtt = ts - pending_pkt_in[dpid]
                    if 0 < rtt < 5.0:
                        with self._lock:
                            self._rtt_samples[dpid].append(rtt)
                    del pending_pkt_in[dpid]
                continue

            m = re.match(r'ECHO_RTT\s+(\d+)\s+([\d.]+)', line)
            if m:
                dpid = int(m.group(1))
                rtt_ms = float(m.group(2))
                with self._lock:
                    self._echo_rtt_ms[dpid] = rtt_ms
