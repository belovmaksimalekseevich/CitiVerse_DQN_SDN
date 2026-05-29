# TASK_15: Critical OpenFlow Pitfalls Found in Open-Source SDN-RL Projects

## Pitfall 1 (CRITICAL): OFPBarrier — FlowMod is Async

### Problem

`ovs-vsctl set-controller` and Ryu's `dp.send_msg(OFPFlowMod)` are both
**asynchronous**. After calling them, OvS has NOT yet processed the command.

If we measure ICD immediately after _apply_assignment(), we measure BEFORE
the new controller's flow rules are installed. The measurement is wrong.

Found in: multiple open-source Ryu projects use OFPBarrierRequest to sync.
Without it: ~30-200ms measurement error during the transition window.

### Fix: send OFPBarrierRequest after every FlowMod batch

Add to `ryu_apps/ctrl_app.py`:

```python
@set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
def switch_features_handler(self, ev):
    dp = ev.msg.datapath
    ofp = dp.ofproto
    parser = dp.ofproto_parser

    # Install table-miss flow
    match = parser.OFPMatch()
    actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
    inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
    mod = parser.OFPFlowMod(
        datapath=dp, priority=0, match=match, instructions=inst,
        idle_timeout=0, hard_timeout=0    # table-miss: never expire
    )
    dp.send_msg(mod)

    # SEND BARRIER — wait for FlowMod to be processed before returning
    barrier = parser.OFPBarrierRequest(dp)
    dp.send_msg(barrier)
    # Ryu will fire EventOFPBarrierReply when OvS confirms processing

@set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
def barrier_reply_handler(self, ev):
    dp = ev.msg.datapath
    LOG.debug(f'Barrier reply from dpid={dp.id} — FlowMod processed')
    # Signal that the switch is ready for measurement
    self._barrier_received[dp.id] = True

def wait_for_barrier(self, dpid, timeout=3.0):
    """Block until barrier reply received or timeout."""
    import time
    self._barrier_received.setdefault(dpid, False)
    t0 = time.time()
    while not self._barrier_received.get(dpid, False):
        if time.time() - t0 > timeout:
            LOG.warning(f'Barrier timeout for dpid={dpid}')
            return False
        time.sleep(0.05)
    self._barrier_received[dpid] = False  # reset for next time
    return True
```

### Fix in environment.py: wait after assignment

```python
def _apply_assignment(self, sw_name, ctrl_idx):
    port = RYU_PORTS[ctrl_idx]
    # 1. Assign controller
    subprocess.run(['ovs-vsctl', 'set-controller', sw_name,
                    f'tcp:127.0.0.1:{port}'], capture_output=True, timeout=5)
    # 2. Clear stale flow table
    subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', sw_name],
                   capture_output=True, timeout=5)
    # 3. Wait for TCP reconnect
    time.sleep(1.5)
    # 4. Verify connection established before measuring
    self._wait_for_switch_connected(sw_name, ctrl_idx)
    # 5. Short settle for MAC relearn
    time.sleep(0.5)

def _wait_for_switch_connected(self, sw_name, ctrl_idx, timeout=5.0):
    """Poll ovs-vsctl until switch shows connected to expected controller."""
    import time
    port = RYU_PORTS[ctrl_idx]
    expected = f'tcp:127.0.0.1:{port}'
    t0 = time.time()
    while time.time() - t0 < timeout:
        result = subprocess.run(
            ['ovs-vsctl', 'get-controller', sw_name],
            capture_output=True, text=True, timeout=3
        )
        if expected in result.stdout:
            return True
        time.sleep(0.2)
    LOG.warning(f'{sw_name} not connected to {expected} after {timeout}s')
    return False
```

---

## Pitfall 2 (CRITICAL): LLDP Flooding When --observe-links Is Used

### Problem

Ryu's `--observe-links` flag makes Ryu send LLDP packets on all ports
to discover topology. After controller reassignment, the new controller
floods LLDP packets to all ports of the newly connected switch.

Effect:
- PACKET_IN flood in first 2-5 seconds after reassignment
- Inflated PACKET_IN rate counter (state[84:89] wrong)
- Corrupted throughput measurement (LLDP treated as traffic)

Found in: multiple GitHub SDN-RL projects disable --observe-links to avoid
measurement corruption. One paper (SDN Flow Entry Mgmt via RL, arXiv 1809.09003)
specifically notes this as a confounding variable.

### Fix A (recommended): Remove --observe-links

```python
# In multi_controller.py — REMOVE --observe-links:
proc = subprocess.Popen(
    ['ryu-manager', self.app_path,
     '--ofp-tcp-listen-port', str(port),
     # '--observe-links',   # REMOVE — causes LLDP flood after reassignment
     '--log-file', f'{self.log_dir}/ctrl_{i}.log'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)
```

We don't use topology discovery in our code — we have LATENCY_MATRIX hardcoded.
There is no reason to use --observe-links.

### Fix B: Filter LLDP from PACKET_IN rate counter

If --observe-links is needed for other reasons, filter LLDP in ctrl_app.py:

```python
@set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
def packet_in_handler(self, ev):
    msg = ev.msg
    dp = msg.datapath
    dpid = dp.id

    # Parse packet type
    pkt = packet.Packet(msg.data)
    eth = pkt.get_protocols(ethernet.ethernet)[0]

    # FILTER: ignore LLDP — don't count in PACKET_IN rate
    from ryu.lib.packet import lldp as lldp_proto
    if eth.ethertype == lldp_proto.LLDP_MAC_NEAREST_BRIDGE:
        return   # drop LLDP silently — do not count in rate

    # Normal handling
    LOG.info(f'PACKET_IN {dpid} {time.time():.6f}')
    self.packet_in_counts[dpid] += 1
    self._update_rates()
    # ... rest of handler ...
```

### Post-reassignment measurement blackout period

After controller assignment, add a blackout period where we don't
count PACKET_IN events for ICD/rate measurement:

```python
# In ControllerMonitor:
self._blackout_until = {}   # dpid -> timestamp

def set_blackout(self, dpid, duration=3.0):
    """Call this after reassignment to suppress measurements."""
    self._blackout_until[dpid] = time.time() + duration

def _read_ctrl_log(self, ctrl_idx):
    ...
    for line in new_lines:
        m = re.match(r'PACKET_IN\s+(\d+)\s+([\d.]+)', line)
        if m:
            dpid = int(m.group(1))
            ts = float(m.group(2))
            # Skip if in blackout period (post-reassignment LLDP flood)
            if ts < self._blackout_until.get(dpid, 0):
                continue
            ...
```

Usage in environment.py after _apply_assignment:
```python
self.ctrl_monitor.set_blackout(sw_idx + 1, duration=3.0)
```

---

## Verification

```bash
# Confirm --observe-links is NOT in Ryu startup:
grep -r 'observe-links' ryu_apps/
# Should return empty

# Confirm LLDP filter in ctrl_app.py:
grep -n 'LLDP\|lldp\|ethertype' ryu_apps/ctrl_app.py
# Should show LLDP check returning early

# Test: after reassignment, verify PACKET_IN rate returns to normal within 5s:
# Monitor /tmp/ryu_logs/ctrl_0.log — should not see burst of PACKET_IN lines
```
