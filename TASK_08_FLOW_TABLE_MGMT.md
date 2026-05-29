# TASK_08: Flow Table Management After Reassignment (fixes B25, B26)

## Context and Problem

From PLAN_AND_REVIEW.md Section 3.2:

**B25 — Ryu SimpleSwitch MAC flood after reassignment:**
SimpleSwitch uses FLOOD until MAC is learned. After reassignment, old flow table
entries from the old controller persist. New controller starts fresh → PACKET_IN flood.
This causes a temporary throughput spike that pollutes reward measurement.

**B26 — Flow tables not cleared after controller switch:**
When sw5 moves ctrl_0 → ctrl_2:
- ctrl_0 still holds sw5's flow table (but PACKET_IN stops coming)
- ctrl_2 receives PACKET_IN for new flows, adds new entries
- Result: split-brain — some flows via ctrl_0 rules, new flows via ctrl_2
- Measurements are corrupted for the first 2-5 seconds

**Fix:** After every `ovs-vsctl set-controller`:
1. `ovs-ofctl del-flows <switch>` — clear all flow entries
2. Wait 1.5s for TCP reconnect to new controller
3. Wait 0.5s for MAC relearn (first PACKET_IN triggers new flow installs)

---

## Changes to dqn/environment.py

Replace `_apply_assignment` in TASK_03 with this version:

```python
def _apply_assignment(self, sw_name, ctrl_idx):
    """
    Reassign switch to new controller.
    FIX B01: calls ovs-vsctl (was missing entirely).
    FIX B25/B26: clears flow table after reassignment so new controller
                 starts fresh without split-brain.
    """
    port = RYU_PORTS[ctrl_idx]
    ctrl_str = f'tcp:127.0.0.1:{port}'

    # Step 1: Disconnect from old controller
    try:
        subprocess.run(['ovs-vsctl', 'set-controller', sw_name, ctrl_str],
                       capture_output=True, timeout=5, check=True)
    except subprocess.CalledProcessError as e:
        LOG.warning(f'set-controller failed for {sw_name}: {e.stderr}')
        return
    except subprocess.TimeoutExpired:
        LOG.error(f'set-controller timeout for {sw_name}')
        return

    # Step 2: Clear flow table (FIX B26 — prevents split-brain)
    try:
        subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', sw_name],
                       capture_output=True, timeout=5)
        LOG.debug(f'Cleared flow table on {sw_name}')
    except Exception as e:
        LOG.warning(f'del-flows failed for {sw_name}: {e}')

    # Step 3: Wait for TCP reconnect + MAC relearn (FIX B25)
    time.sleep(1.5)   # TCP reconnect to new Ryu instance
    time.sleep(0.5)   # First PACKET_IN → FlowMod cycle for MAC relearn

def _bulk_apply_assignments(self):
    """Apply all assignments in reset() with minimal total wait time."""
    for i, sw in enumerate(SWITCHES):
        port = RYU_PORTS[self.assignments[i]]
        ctrl_str = f'tcp:127.0.0.1:{port}'
        subprocess.run(['ovs-vsctl', 'set-controller', sw, ctrl_str],
                       capture_output=True, timeout=5)

    # Single batch del-flows (faster than per-switch in reset)
    for sw in SWITCHES:
        subprocess.run(['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', sw],
                       capture_output=True, timeout=5)

    time.sleep(3.0)   # wait for all switches to reconnect and relearn MACs
```

---

## Changes to step() timing

The TASK_03 version has `time.sleep(0.2)` after migration — replace with:

```python
if migrated:
    self.migration_count += 1
    self.assignments[sw_idx] = ctrl_idx
    self._apply_assignment(sw, ctrl_idx)
    # _apply_assignment already sleeps 2.0s internally — no extra sleep needed

# Then measure fresh state
self._refresh_measurements()
```

---

## Changes to reset()

```python
def reset(self):
    # ... init assignments ...
    
    # Apply all at once (faster: batch set-controller, then batch del-flows)
    self._bulk_apply_assignments()
    
    time.sleep(1.0)  # extra settle time after bulk operation
    self._refresh_measurements()
    return self._get_state()
```

---

## Background iperf3 traffic in reset()

From PLAN_AND_REVIEW.md Step 3: Start iperf3 background traffic so throughput
measurement has something to measure. Without it, throughput is always ~100Mbps
(no traffic, no queuing, no useful signal).

```python
import subprocess, threading

class CitiverseRealEnv:
    def __init__(self, net, ctrl_mgr, ...):
        ...
        self._iperf_procs = []

    def _start_background_traffic(self):
        """Start iperf3 flows between hosts to generate measurable traffic."""
        self._stop_background_traffic()
        # Use hosts from different zones to create cross-zone traffic
        try:
            h1 = self.net.get('h1s1')   # res1 zone
            h10 = self.net.get('h1s10') # ind zone
            # server
            srv = h10.popen('iperf3 -s -p 5201 -D')
            self._iperf_procs.append(srv)
            time.sleep(0.3)
            # client: continuous UDP stream
            cli = h1.popen('iperf3 -c {} -p 5201 -u -b 10M -t 3600'.format(
                h10.IP()))
            self._iperf_procs.append(cli)
        except Exception as e:
            LOG.warning(f'Failed to start background traffic: {e}')

    def _stop_background_traffic(self):
        for p in self._iperf_procs:
            try:
                p.terminate()
            except Exception:
                pass
        self._iperf_procs.clear()

    def close(self):
        self._stop_background_traffic()

    def reset(self):
        self._start_background_traffic()  # BEFORE measuring
        self._bulk_apply_assignments()
        time.sleep(1.0)
        self._refresh_measurements()
        return self._get_state()
```

---

## Throughput measurement via iperf3 (replaces placeholder)

PLAN_AND_REVIEW.md Section 3.2 warns: "h1→h10 iperf is not network throughput, it's
one end-to-end path". Correct fix: measure average of 3-5 flows from different zones,
OR use controller PACKET_IN handling rate as a proxy.

```python
def _measure_throughput(self):
    """
    Proxy: ctrl load inversely correlates with throughput capacity.
    Overloaded controllers drop PACKET_INs → lower effective throughput.
    This avoids the h1→h10 single-path bias.
    """
    loads = self._compute_load()
    max_load = float(np.max(loads))
    # Throughput degrades when any controller exceeds MAX_CTRL_LOAD
    overload = max(0.0, max_load - MAX_CTRL_LOAD)
    # Linear penalty: 15 Mbps per unit of overload
    throughput_mbps = max(0.0, 100.0 - overload * 15.0)
    # Add PACKET_IN rate as secondary signal
    pkt_rates = self._packet_in_rates
    total_pkt_rate = float(np.sum(pkt_rates))
    # High PACKET_IN rate = high activity = good throughput signal
    activity_bonus = min(10.0, total_pkt_rate * 0.2)
    return min(100.0, throughput_mbps + activity_bonus)
```

---

## Verification Checklist

```bash
# 1. Verify del-flows is called after reassignment:
grep -n 'del-flows' dqn/environment.py
# Expected: appears in _apply_assignment AND _bulk_apply_assignments

# 2. Verify sleep timing:
grep -n 'sleep' dqn/environment.py
# Expected: 1.5 + 0.5 = 2.0s in _apply_assignment, 3.0s in _bulk_apply_assignments

# 3. Manual test: reassign s1 and check flow table is cleared:
# Before:
ovs-ofctl -O OpenFlow13 dump-flows s1
# Reassign:
ovs-vsctl set-controller s1 tcp:127.0.0.1:6654
ovs-ofctl -O OpenFlow13 del-flows s1
# After 2s:
ovs-ofctl -O OpenFlow13 dump-flows s1  # should have new flows from ctrl on :6654

# 4. Check controller assignment:
ovs-vsctl get-controller s1  # should show tcp:127.0.0.1:6654
```

## Implementation note

These changes go INTO `dqn/environment.py` (TASK_03 file).
Do NOT create a separate environment file — modify the existing one.
Implement TASK_07 (ControllerMonitor) first so it's available when testing.
