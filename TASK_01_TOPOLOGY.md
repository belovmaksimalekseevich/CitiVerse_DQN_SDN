# TASK_01: Steps 1-2 — topology_data.py and citiverse_topo.py

## STEP 1: topology/topology_data.py (create from scratch)

```python
# topology/topology_data.py

N_SWITCHES    = 20
N_CONTROLLERS = 5
N_ZONES       = 5
STATE_DIM     = 94
ACTION_DIM    = 100   # N_SWITCHES * N_CONTROLLERS
MAX_CTRL_LOAD = 5     # max switches per controller before overload
INTRA_DELAY_MS = 3    # delay within same zone (ms)

# Inter-zone latency matrix (ms). Asymmetric in general.
INTER_ZONE_DELAYS = {
    ('res1', 'res2'): 8,   ('res2', 'res1'): 8,
    ('res1', 'com1'): 15,  ('com1', 'res1'): 15,
    ('res2', 'com2'): 12,  ('com2', 'res2'): 12,
    ('com1', 'com2'): 5,   ('com2', 'com1'): 5,
    ('com1', 'ind'):  20,  ('ind',  'com1'): 20,
    ('com2', 'ind'):  18,  ('ind',  'com2'): 18,
    ('res1', 'com2'): 22,  ('com2', 'res1'): 22,
    ('res2', 'com1'): 20,  ('com1', 'res2'): 20,
    ('res1', 'ind'):  30,  ('ind',  'res1'): 30,
    ('res2', 'ind'):  28,  ('ind',  'res2'): 28,
}

def get_inter_delay(zone_a, zone_b):
    if zone_a == zone_b:
        return INTRA_DELAY_MS
    return INTER_ZONE_DELAYS.get((zone_a, zone_b), 25)

# Zone definitions: switches 1-20
# res1: switches 1-6,  controller 0
# res2: switches 7-11, controller 1
# com1: switches 12-15, controller 2
# com2: switches 16-18, controller 3
# ind:  switches 19-20, controller 4
ZONES = {
    'res1': {'switches': list(range(1, 7)),   'controller': 0, 'name': 'Residential-1'},
    'res2': {'switches': list(range(7, 12)),  'controller': 1, 'name': 'Residential-2'},
    'com1': {'switches': list(range(12, 16)), 'controller': 2, 'name': 'Commercial-1'},
    'com2': {'switches': list(range(16, 19)), 'controller': 3, 'name': 'Commercial-2'},
    'ind':  {'switches': list(range(19, 21)), 'controller': 4, 'name': 'Industrial'},
}

# Derived mappings
SWITCHES    = list(range(1, N_SWITCHES + 1))    # [1, 2, ..., 20]
CONTROLLERS = list(range(N_CONTROLLERS))         # [0, 1, 2, 3, 4]
CTRL_PORTS  = [6653 + i for i in range(N_CONTROLLERS)]  # [6653..6657]

SW_ZONE   = {}   # sw_id -> zone_name
ZONE_CTRL = {}   # zone_name -> controller_idx
CTRL_ZONE = {}   # controller_idx -> zone_name
SWITCH_DEFAULT_CTRL = {}  # sw_id -> default controller_idx

for _zone, _data in ZONES.items():
    _ctrl = _data['controller']
    ZONE_CTRL[_zone] = _ctrl
    CTRL_ZONE[_ctrl] = _zone
    for _sw in _data['switches']:
        SW_ZONE[_sw] = _zone
        SWITCH_DEFAULT_CTRL[_sw] = _ctrl

# Latency matrix: LATENCY_MATRIX[sw_idx][ctrl_idx] in ms
# sw_idx = sw_id - 1 (0-indexed)
def _build_latency_matrix():
    import numpy as np
    mat = np.zeros((N_SWITCHES, N_CONTROLLERS), dtype=np.float32)
    for sw_idx, sw in enumerate(SWITCHES):
        sw_zone = SW_ZONE[sw]
        for ctrl_idx in CONTROLLERS:
            ctrl_zone = CTRL_ZONE[ctrl_idx]
            mat[sw_idx][ctrl_idx] = get_inter_delay(sw_zone, ctrl_zone)
    return mat

LATENCY_MATRIX = _build_latency_matrix()   # shape (20, 5)

# Traffic profiles (MANDATORY for scientific contribution)
TRAFFIC_PROFILES = ['morning', 'business', 'evening', 'night']
PROFILE_IDX = {p: i for i, p in enumerate(TRAFFIC_PROFILES)}

ZONE_LOAD_FACTORS = {
    'morning':  {'res1': 2.5, 'res2': 2.0, 'com1': 0.5, 'com2': 0.5, 'ind': 0.3},
    'business': {'res1': 0.7, 'res2': 0.7, 'com1': 2.5, 'com2': 2.0, 'ind': 1.0},
    'evening':  {'res1': 2.0, 'res2': 2.0, 'com1': 1.5, 'com2': 1.5, 'ind': 0.5},
    'night':    {'res1': 0.3, 'res2': 0.3, 'com1': 0.3, 'com2': 0.3, 'ind': 2.5},
}

# Inter-zone links for Mininet topology
MININET_INTER_LINKS = [
    ('res1', 'com1'), ('res1', 'res2'),
    ('res2', 'com2'), ('com1', 'ind'),
]

def get_intra_links():
    """Returns list of dicts: {src, dst, delay_ms, bw_mbps}"""
    links = []
    for zone, data in ZONES.items():
        sws = data['switches']
        for i in range(len(sws) - 1):
            links.append({
                'src':      sws[i],
                'dst':      sws[i + 1],
                'delay_ms': INTRA_DELAY_MS,
                'bw_mbps':  100,
            })
    return links

def get_zone_optimal_assignments():
    """Returns dict: sw_id -> default controller_idx."""
    return {sw: SWITCH_DEFAULT_CTRL[sw] for sw in SWITCHES}

def get_zone_optimal_array():
    """Returns np.array shape (N_SWITCHES,) with zone-optimal controller indices."""
    import numpy as np
    return np.array([SWITCH_DEFAULT_CTRL[sw] for sw in SWITCHES], dtype=int)
```

### Verification

```bash
python3 -c "
from topology.topology_data import *
import numpy as np
assert STATE_DIM == 94
assert ACTION_DIM == 100
assert len(SWITCHES) == 20
assert len(CTRL_PORTS) == 5
assert LATENCY_MATRIX.shape == (20, 5)
assert LATENCY_MATRIX[0][0] == INTRA_DELAY_MS   # res1 sw -> res1 ctrl: intra
assert LATENCY_MATRIX[0][4] == 30               # res1 sw -> ind ctrl: inter
print('topology_data OK')
print('LATENCY_MATRIX[0]:',  LATENCY_MATRIX[0])
"
```

---

## STEP 2: topology/citiverse_topo.py

Copy `/home/maksim/citiverse-dqn/topology/citiverse_topo.py` as base.
Make EXACTLY these changes:

### Change 1: Fix link bandwidth (FIX B03)

```python
# BEFORE (incorrect — Mininet ignores bw > 1000):
self.addLink(swA, swB, bw=10000)

# AFTER (correct + add delay to match LATENCY_MATRIX):
from topology.topology_data import get_inter_delay, SW_ZONE, INTRA_DELAY_MS

# Intra-zone links:
self.addLink(swA, swB,
             bw=100,
             delay=f'{INTRA_DELAY_MS}ms',
             loss=0,
             max_queue_size=1000)

# Inter-zone links:
delay_ms = get_inter_delay(SW_ZONE[sw_id_a], SW_ZONE[sw_id_b])
self.addLink(swA, swB,
             bw=50,
             delay=f'{delay_ms}ms',
             loss=0,
             max_queue_size=1000)
```

### Change 2: Remove controller from create_network (FIX: controllers assigned by ovs-vsctl)

```python
# REMOVE this line from create_network():
# net.addController('c0', controller=RemoteController, ...)

# Switches start with no controller — assigned by environment.py via ovs-vsctl
# Set fail_mode=standalone so switches forward during controller transition:
for sw_name in switch_names:
    net.get(sw_name).cmd(f'ovs-vsctl set bridge {sw_name} fail_mode=standalone')
    net.get(sw_name).cmd(f'ovs-vsctl set bridge {sw_name} protocols=OpenFlow13')
```

### Change 3: Add helper functions

```python
def get_all_switch_names():
    """Returns list of OvS switch names: ['s1', 's2', ..., 's20']"""
    return [f's{i}' for i in range(1, 21)]


def reset_all_flows(net=None):
    """Clear all flow tables across all switches. Call after each episode reset."""
    import subprocess
    for i in range(1, 21):
        sw = f's{i}'
        subprocess.run(
            ['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', sw],
            capture_output=True, timeout=5
        )


def assign_all_controllers(assignments, ctrl_ports):
    """
    Bulk-assign all switches to controllers via ovs-vsctl.
    assignments: array of ctrl_idx per switch (0-indexed)
    ctrl_ports: list of Ryu ports [6653, 6654, 6655, 6656, 6657]
    """
    import subprocess
    for sw_idx, ctrl_idx in enumerate(assignments):
        sw = f's{sw_idx + 1}'
        port = ctrl_ports[ctrl_idx]
        subprocess.run(
            ['ovs-vsctl', 'set-controller', sw, f'tcp:127.0.0.1:{port}'],
            capture_output=True, timeout=5
        )
```

### Change 4: Set OpenFlow 1.3 on all switches

```python
# In CitiverseTopo.__init__ or build():
for sw_name in self.switch_names:
    self.addSwitch(sw_name,
                   protocols='OpenFlow13',
                   failMode='standalone')
```

### Verification

```bash
# Syntax check:
python3 -c "from topology.citiverse_topo import CitiverseTopo; print('import OK')"

# Check no controller added:
grep -n 'addController\|RemoteController' topology/citiverse_topo.py
# Should return empty (no controller in topology file)

# Check bw limits:
grep -n 'bw=' topology/citiverse_topo.py
# All values should be <= 1000
```
