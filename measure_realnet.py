#!/usr/bin/env python3
"""Measured control-plane evaluation on live Mininet + Ryu.

For a switch->controller assignment under a traffic profile, each switch emits
control-plane load (raw-socket flooder, varying src MAC -> fresh table-miss ->
PACKET_IN) at a rate proportional to its zone's traffic factor. The assignment
decides how that load distributes across the 5 single-threaded Ryu controllers.
We then MEASURE flow-setup latency per switch (fresh-flow ping). A balanced
assignment keeps every controller below saturation -> low latency; a skewed one
creates a hotspot controller (single eventlet thread) -> high latency.

This replaces the analytical M/M/1 with a real, measured quantity.
"""
import sys, os, time, subprocess, statistics, json, argparse
for _p in ['/usr/lib/python3/dist-packages', '/usr/local/lib/python3/dist-packages']:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from functools import partial
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel

sys.path.insert(0, '/home/maksim/simenv_dqn_mininet_v2')
from topology.topology_data import (
    SWITCHES, N_SWITCHES, N_CONTROLLERS, SW_ZONE, ZONE_LOAD_FACTORS,
    CTRL_PORTS, get_intra_links, get_inter_delay, MININET_INTER_LINKS, ZONES,
)
from baselines.run_baselines import (
    baseline_zone_optimal, baseline_load_balanced, baseline_kmeans,
    baseline_all_to_ctrl0, baseline_random,
)

RYU = '/home/maksim/dqn_env/bin/ryu-manager'
VENVPY = '/home/maksim/dqn_env/bin/python3'
APP = 'ryu_apps/ctrl_app.py'
PROFILES_ALL = ['morning', 'business', 'evening', 'night']

FLOODER = r'''
import socket, sys, time, struct, random
iface, dst_mac, rate, dur = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
dstb = bytes(int(x,16) for x in dst_mac.split(':'))
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW); s.bind((iface,0))
payload = b'x'*46; end = time.time()+dur; iv = 1.0/rate if rate>0 else 0
while time.time() < end:
    srcb = bytes([0x02]+[random.randint(0,255) for _ in range(5)])
    s.send(dstb+srcb+struct.pack('!H',0x0800)+payload)
    if iv: time.sleep(iv)
'''

def sh(cmd, t=15):
    return subprocess.run(cmd, capture_output=True, timeout=t, text=True)

# host naming: switch i -> hA=h{2i-1} (flooder src), hB=h{2i} (probe)
def hA(i): return f'h{2*i-1}'
def hB(i): return f'h{2*i}'

class FullTopo(Topo):
    def build(self):
        sw = {}
        for i in range(1, N_SWITCHES+1):
            sw[i] = self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
        for lk in get_intra_links():
            self.addLink(sw[lk['src']], sw[lk['dst']],
                         delay=f"{lk['delay_ms']}ms", bw=lk.get('bw_mbps',100), max_queue_size=1000)
        for za, zb in MININET_INTER_LINKS:
            ga, gb = ZONES[za]['switches'][-1], ZONES[zb]['switches'][0]
            self.addLink(sw[ga], sw[gb], delay=f"{get_inter_delay(za,zb)}ms", bw=50, max_queue_size=1000)
        hn = 1
        for i in range(1, N_SWITCHES+1):
            for _ in range(2):
                self.addHost(f'h{hn}', ip=f'10.0.{hn//256}.{hn%256}/16'); self.addLink(f'h{hn}', sw[i]); hn += 1

def start_controllers():
    procs = []
    for i, p in enumerate(CTRL_PORTS):
        procs.append(subprocess.Popen(
            [RYU, APP, '--ofp-tcp-listen-port', str(p), '--log-file', f'/tmp/meas_ctrl{i}.log'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    return procs

def apply_assignment(asg):
    for i in range(1, N_SWITCHES+1):
        sh(['ovs-vsctl', 'set-controller', f's{i}', f'tcp:127.0.0.1:{CTRL_PORTS[int(asg[i-1])]}'])

def reset_miss(i):
    sh(['ovs-ofctl','-O','OpenFlow13','del-flows',f's{i}'])
    sh(['ovs-ofctl','-O','OpenFlow13','add-flow',f's{i}','priority=0,actions=CONTROLLER:65535'])

def seed_arp_and_learn(net):
    """static ARP for each switch's host pair + one learning packet, so probe
    pings are unicast (no ARP broadcast propagation) and controllers know ports."""
    for i in range(1, N_SWITCHES+1):
        a, b = net.get(hA(i)), net.get(hB(i))
        a.cmd(f'arp -s {b.IP()} {b.MAC()}'); b.cmd(f'arp -s {a.IP()} {a.MAC()}')
    for i in range(1, N_SWITCHES+1):
        net.get(hB(i)).cmd(f'ping -c1 -W1 {net.get(hA(i)).IP()} >/dev/null 2>&1')

def flow_setup_latency(net, i):
    reset_miss(i); time.sleep(0.15)
    a, b = net.get(hA(i)), net.get(hB(i))
    t0 = time.time()
    out = b.cmd(f'ping -c1 -W3 {a.IP()}')
    dt = (time.time()-t0)*1000.0
    return dt if ('1 received' in out or '1 packets received' in out) else None

def start_floods(net, asg, profile, base_pps):
    open('/tmp/flooder.py','w').write(FLOODER)
    zf = ZONE_LOAD_FACTORS[profile]
    for i in range(1, N_SWITCHES+1):
        rate = base_pps * zf[SW_ZONE[SWITCHES[i-1]]]
        a, b = net.get(hA(i)), net.get(hB(i))
        intf = a.defaultIntf().name
        a.cmd(f'{VENVPY} /tmp/flooder.py {intf} {b.MAC()} {rate:.1f} 9999 >/dev/null 2>&1 &')

def stop_floods(net):
    for i in range(1, N_SWITCHES+1):
        net.get(hA(i)).cmd('pkill -f flooder.py 2>/dev/null')
    sh(['pkill','-f','flooder.py'])

def ctrl_loads(asg, profile):
    zf = ZONE_LOAD_FACTORS[profile]; L = [0.0]*N_CONTROLLERS
    for i in range(N_SWITCHES):
        L[int(asg[i])] += zf[SW_ZONE[SWITCHES[i]]]
    return L

def measure(net, asg, profile, base_pps, reps=2, settle=5):
    apply_assignment(asg)
    for i in range(1, N_SWITCHES+1):
        reset_miss(i)
    time.sleep(2)
    start_floods(net, asg, profile, base_pps)
    time.sleep(settle)
    per_sw = {}
    for rep in range(reps):
        for i in range(1, N_SWITCHES+1):
            t = flow_setup_latency(net, i)
            if t is not None:
                per_sw.setdefault(i, []).append(t)
    stop_floods(net)
    time.sleep(1)
    sw_mean = {i: statistics.mean(v) for i, v in per_sw.items() if v}
    vals = list(sw_mean.values())
    L = ctrl_loads(asg, profile)
    # per-controller mean latency (avg over its switches)
    cmean = {}
    for c in range(N_CONTROLLERS):
        cs = [sw_mean[i] for i in range(1, N_SWITCHES+1)
              if int(asg[i-1]) == c and i in sw_mean]
        if cs: cmean[c] = round(statistics.mean(cs), 1)
    return {
        'mean_ms': round(statistics.mean(vals), 1) if vals else None,
        'max_ms': round(max(vals), 1) if vals else None,
        'p95_ms': round(sorted(vals)[int(0.95*len(vals))-1], 1) if len(vals) >= 3 else (round(max(vals),1) if vals else None),
        'ctrl_loads': [round(x,2) for x in L],
        'ctrl_latency': cmean,
        'n_switches_measured': len(vals),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-pps', type=float, default=150.0)
    ap.add_argument('--profiles', default='morning,night')
    ap.add_argument('--dqn-assignments', default=None, help='JSON {profile:{name:assignment}}')
    ap.add_argument('--out', default='results/measured_summary.json')
    ap.add_argument('--reps', type=int, default=2)
    args = ap.parse_args()
    profiles = args.profiles.split(',')

    setLogLevel('warning'); sh(['mn','-c'])
    ctrls = start_controllers(); time.sleep(4)
    net = Mininet(topo=FullTopo(), controller=None,
                  switch=partial(OVSSwitch, protocols='OpenFlow13'),
                  link=TCLink, autoSetMacs=True)
    net.start()
    print('connecting switches...'); time.sleep(8)
    seed_arp_and_learn(net)

    dqn_asg = {}
    if args.dqn_assignments and os.path.exists(args.dqn_assignments):
        dqn_asg = json.load(open(args.dqn_assignments))

    base = {
        'ZoneOptimal':  baseline_zone_optimal(),
        'LoadBalanced': baseline_load_balanced(),
        'KMeans':       baseline_kmeans(),
    }
    results = {}
    for prof in profiles:
        results[prof] = {}
        named = dict(base)
        for nm, a in (dqn_asg.get(prof, {}) or {}).items():
            named[nm] = a
        for nm, asg in named.items():
            import numpy as np
            asg = np.asarray(asg, dtype=int)
            r = measure(net, asg, prof, args.base_pps, reps=args.reps)
            results[prof][nm] = r
            print(f'[{prof}] {nm:14s} mean={r["mean_ms"]} max={r["max_ms"]} '
                  f'loads={r["ctrl_loads"]} ctrl_lat={r["ctrl_latency"]}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=2)
    print('saved', args.out)
    net.stop()
    for c in ctrls: c.terminate()
    sh(['mn','-c'])

if __name__ == '__main__':
    try:
        main()
    finally:
        subprocess.run(['pkill','-f','flooder.py'], capture_output=True)
        subprocess.run(['pkill','-f','meas_ctrl'], capture_output=True)
        subprocess.run(['mn','-c'], capture_output=True)
