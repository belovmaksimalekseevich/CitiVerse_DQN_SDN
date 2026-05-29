# TASK_16: Switch State Management — fail_mode, TCAM, ofctl Statistics

## Issue 1 (MEDIUM): fail_mode=standalone Prevents Packet Loss During Reassignment

### Problem

Default OvS fail_mode in Mininet is `secure` — the switch drops ALL packets
when disconnected from controller. During our 1.5s reconnect window:
- All packets through that switch are dropped
- throughput measurement shows 0 Mbps (wrong: this is a control-plane artifact)
- reward is corrupted by artificial throughput penalty

### Fix: Set fail_mode=standalone on all switches

In `topology/citiverse_topo.py`:
```python
def build(self):
    for i in range(1, N_SWITCHES + 1):
        sw_name = f's{i}'
        self.addSwitch(
            sw_name,
            protocols='OpenFlow13',
            failMode='standalone',    # forward based on existing flows during disconnect
        )
```

Or set via ovs-vsctl after Mininet start (in run_all.py):
```python
def set_all_fail_standalone(net):
    """Call once after net.start() before any training."""
    for i in range(1, 21):
        sw = net.get(f's{i}')
        sw.cmd(f'ovs-vsctl set bridge s{i} fail_mode=standalone')
    LOG.info('All switches set to fail_mode=standalone')
```

### What standalone means

- During disconnect: switch forwards traffic using EXISTING flow table entries
- New traffic with no matching flow: FLOOD (same as initial state)
- Effect on measurement: throughput does NOT drop to 0 during reassignment
- ICD: the switch still handles packets, just with old flow rules for 1.5s

This is a realistic model of production SDN behavior (fail-open is standard practice).
Note it in the paper: "Switches operate in fail-open mode during controller migration,
consistent with production deployments."

---

## Issue 2 (MEDIUM): TCAM Overflow — Flow Table Size Limits

### Problem

OvS software switch default: up to 1,000,000 flow entries (software table, no TCAM limit).
BUT in Mininet with limited memory: after many episodes of MAC learning, the flow
table grows unboundedly. At ~200 RealEnv episodes × 20 switches × ~50 flows per switch
= 200,000 flow entries → memory pressure → OvS slowdown.

We add `del-flows` in TASK_08 after each reassignment, but MAC-learning flows
accumulate BETWEEN reassignments.

### Fix: Set idle_timeout on all installed flows

In `ctrl_app.py`:
```python
def _add_flow(self, dp, priority, match, actions):
    parser = dp.ofproto_parser
    ofp = dp.ofproto
    inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
    mod = parser.OFPFlowMod(
        datapath=dp,
        priority=priority,
        match=match,
        instructions=inst,
        idle_timeout=30,    # remove flow if no matching packets for 30s
        hard_timeout=120,   # always remove after 120s regardless
    )
    dp.send_msg(mod)
```

Exception: table-miss flow (priority=0) must NOT have timeouts:
```python
# In switch_features_handler:
mod = parser.OFPFlowMod(
    datapath=dp, priority=0, match=match, instructions=inst,
    idle_timeout=0, hard_timeout=0    # table-miss: permanent
)
```

### Monitor flow table size

Add to ControllerMonitor:
```python
def get_flow_counts(self):
    """Returns dict: sw_name -> flow entry count (via ovs-ofctl)."""
    import subprocess
    counts = {}
    for i in range(1, 21):
        sw = f's{i}'
        result = subprocess.run(
            ['ovs-ofctl', '-O', 'OpenFlow13', 'dump-flows', sw],
            capture_output=True, text=True, timeout=3
        )
        counts[sw] = result.stdout.count('\n') - 1  # subtract header
    return counts
```

Alert if any switch > 500 flows:
```python
flow_counts = monitor.get_flow_counts()
for sw, count in flow_counts.items():
    if count > 500:
        LOG.warning(f'Flow table large: {sw} has {count} entries')
```

---

## Issue 3 (MEDIUM): Per-Switch Flow Statistics for Better ICD Measurement

### Problem

Our current ICD measurement (TASK_07 ControllerMonitor) relies on parsing
log files for PACKET_IN → FLOW_MOD latency. This is indirect and has:
- Log write latency (~10-50ms)
- Only captures first-packet RTT (subsequent packets use installed flows)
- Not available if log format changes

### Better approach: ofctl dump-ports statistics

OvS can report per-port packet/byte counts via OpenFlow PortStats.
While this doesn't directly give ICD, it gives accurate throughput and load:

```python
# Add to ryu_apps/ctrl_app.py

from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3

class CitiCtrlApp(app_manager.RyuApp):
    ...
    def request_port_stats(self, datapath):
        """Request port statistics from switch."""
        ofp = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofp.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        body = ev.msg.body
        for stat in body:
            port = stat.port_no
            rx_bytes = stat.rx_bytes
            tx_bytes = stat.tx_bytes
            rx_packets = stat.rx_packets
            # Log for throughput estimation:
            LOG.info(f'PORT_STATS {dpid} {port} {rx_bytes} {tx_bytes} {rx_packets} {time.time():.3f}')
```

Parse PORT_STATS in ControllerMonitor for throughput:
```python
def get_throughput_mbps(self, ctrl_idx, window=5.0):
    """Estimate throughput from port stats delta over window."""
    logpath = f'{self.log_dir}/ctrl_{ctrl_idx}.log'
    # Parse last PORT_STATS lines, compute byte delta / time delta
    # Returns throughput in Mbps
    ...
```

### When to request stats

Call `request_port_stats` every 2-3 seconds from a background thread:
```python
# In MultiControllerManager or ControllerMonitor:
def _stats_request_loop(self):
    while self._running:
        for dp in self._connected_datapaths.values():
            self.app.request_port_stats(dp)
        time.sleep(2.0)
```

This gives us real throughput measurements independent of PACKET_IN rate.

### Priority

- Port stats: implement in Phase 2 (RealEnv) if simple throughput proxy is insufficient
- Phase 1 (SimEnv): uses analytical throughput — no port stats needed
- For Phase 2 smoke test: use simple load-based throughput proxy from TASK_03 first
- Only implement full port stats if throughput signal is too noisy

---

## Summary of Changes Per File

| File | Change | Priority |
|---|---|---|
| topology/citiverse_topo.py | failMode='standalone' in addSwitch | HIGH |
| ryu_apps/ctrl_app.py | idle_timeout=30, hard_timeout=120 in _add_flow | HIGH |
| ryu_apps/ctrl_app.py | OFPPortStatsRequest every 2s | MEDIUM |
| ryu_apps/ctrl_monitor.py | get_flow_counts() alert | MEDIUM |
| run_all.py | set_all_fail_standalone() after net.start() | HIGH |
