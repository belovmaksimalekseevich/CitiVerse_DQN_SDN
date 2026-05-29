# topology/topology_data.py

N_SWITCHES    = 20
N_CONTROLLERS = 5
N_ZONES       = 5
STATE_DIM     = 94
ACTION_DIM    = 100   # N_SWITCHES * N_CONTROLLERS
MAX_CTRL_LOAD = 5     # max switches per controller before overload

# --- Controller queuing model (M/M/1-style): load-dependent ICD term ---
CTRL_CAPACITY = 14.0   # mu: controller service capacity (offered-load units)
Q_COEF        = 0.4    # queuing-delay coefficient
W_MAX_MS      = 50.0   # cap on queuing delay (ms)
LOAD_NORM     = 20.0   # reference for normalising load state features

def queuing_delay_ms(load):
    """M/M/1-style queuing delay (ms) vs controller offered load."""
    load = float(load)
    denom = CTRL_CAPACITY - load
    if denom < 0.5:
        denom = 0.5
    q = Q_COEF * load * load / denom
    return q if q < W_MAX_MS else W_MAX_MS
INTRA_DELAY_MS = 3    # delay within same zone (ms)

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

# Zone definitions: switches 1-20 (1-indexed)
# res1: sw 1-6,   controller 0  (6 switches)
# res2: sw 7-11,  controller 1  (5 switches)
# com1: sw 12-15, controller 2  (4 switches)
# com2: sw 16-18, controller 3  (3 switches)
# ind:  sw 19-20, controller 4  (2 switches)
ZONES = {
    'res1': {'switches': list(range(1, 7)),   'controller': 0, 'name': 'Residential-1'},
    'res2': {'switches': list(range(7, 12)),  'controller': 1, 'name': 'Residential-2'},
    'com1': {'switches': list(range(12, 16)), 'controller': 2, 'name': 'Commercial-1'},
    'com2': {'switches': list(range(16, 19)), 'controller': 3, 'name': 'Commercial-2'},
    'ind':  {'switches': list(range(19, 21)), 'controller': 4, 'name': 'Industrial'},
}

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

TRAFFIC_PROFILES = ['morning', 'business', 'evening', 'night']
PROFILE_IDX = {p: i for i, p in enumerate(TRAFFIC_PROFILES)}

ZONE_LOAD_FACTORS = {
    'morning':  {'res1': 2.5, 'res2': 2.0, 'com1': 0.5, 'com2': 0.5, 'ind': 0.3},
    'business': {'res1': 0.7, 'res2': 0.7, 'com1': 2.5, 'com2': 2.0, 'ind': 1.0},
    'evening':  {'res1': 2.0, 'res2': 2.0, 'com1': 1.5, 'com2': 1.5, 'ind': 0.5},
    'night':    {'res1': 0.3, 'res2': 0.3, 'com1': 0.3, 'com2': 0.3, 'ind': 2.5},
}

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
