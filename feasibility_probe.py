#!/usr/bin/env python3
"""Feasibility probe: does flow-setup latency rise when the Ryu controller is
loaded with PACKET_IN traffic? If yes, the 'hybrid / measured' approach is real.
Run as root:  sudo PYTHONPATH=/usr/lib/python3/dist-packages:. python3 feasibility_probe.py
"""
import sys, time, subprocess, re, statistics
for _p in ['/usr/lib/python3/dist-packages', '/usr/local/lib/python3/dist-packages']:
    if _p not in sys.path:
        sys.path.insert(0, _p)
from functools import partial
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.topo import Topo
from mininet.log import setLogLevel

RYU = '/home/maksim/dqn_env/bin/ryu-manager'
APP = 'ryu_apps/ctrl_app.py'
PORT = 6653
NSW = 6

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, timeout=10, text=True)

class ProbeTopo(Topo):
    def build(self):
        sws = [self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
               for i in range(1, NSW + 1)]
        for i in range(len(sws) - 1):
            self.addLink(sws[i], sws[i + 1], delay='3ms', bw=100, max_queue_size=1000)
        hn = 1
        for s in sws:
            for _ in range(2):
                self.addHost(f'h{hn}', ip=f'10.0.0.{hn}/24')
                self.addLink(f'h{hn}', s)
                hn += 1

def clear_flows():
    """Remove learned flows but RE-INSTALL the table-miss rule, otherwise a
    secure-mode switch drops everything and the controller sees no PACKET_IN."""
    for i in range(1, NSW + 1):
        sh(['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', f's{i}'])
        sh(['ovs-ofctl', '-O', 'OpenFlow13', 'add-flow', f's{i}',
            'priority=0,actions=CONTROLLER:65535'])

def setup_flow_time(net, a, b):
    """Wall-clock time to establish connectivity for a FRESH flow (ARP + first ICMP),
    i.e. the controller-dependent flow-setup latency, in ms. None on failure."""
    clear_flows()
    ha, hb = net.get(a), net.get(b)
    ha.cmd('ip neigh flush all'); hb.cmd('ip neigh flush all')
    time.sleep(0.3)
    t0 = time.time()
    out = ha.cmd(f'ping -c1 -W3 {hb.IP()}')
    dt = (time.time() - t0) * 1000.0
    if '1 received' in out or '1 packets received' in out:
        return dt
    return None

def main():
    setLogLevel('warning')
    sh(['mn', '-c'])
    ryu = subprocess.Popen([RYU, APP, '--ofp-tcp-listen-port', str(PORT),
                            '--log-file', '/tmp/probe_ctrl.log'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(4)
    net = Mininet(topo=ProbeTopo(), controller=None,
                  switch=partial(OVSSwitch, protocols='OpenFlow13'),
                  link=TCLink, autoSetMacs=True)
    net.start()
    for i in range(1, NSW + 1):
        net.get(f's{i}').cmd(f'ovs-vsctl set-controller s{i} tcp:127.0.0.1:{PORT}')
    print('waiting for switches to connect...')
    time.sleep(8)

    # sanity: baseline connectivity with table-miss present
    sanity = setup_flow_time(net, 'h1', 'h12')
    print(f'sanity h1->h12 setup = {sanity}')

    # --- LOW LOAD baseline ---
    low = []
    for _ in range(10):
        t = setup_flow_time(net, 'h1', 'h12')
        if t is not None:
            low.append(t)
        time.sleep(0.4)

    # --- HIGH LOAD: a few dedicated hosts flood broadcast (one ping -f process
    #     each = sustained high PACKET_IN rate to the single controller, WITHOUT
    #     forking thousands of processes / saturating host CPU). h1/h12 stay free
    #     as probe hosts. ---
    print('starting controlled PACKET_IN flood (ping -f broadcast)...')
    flood = ['h3', 'h4', 'h5', 'h6', 'h7', 'h8']
    pin_before = int(sh(['grep', '-c', 'PACKET_IN', '/tmp/probe_ctrl.log']).stdout.strip() or 0)
    for hn in flood:
        net.get(hn).cmd('timeout 22 ping -f -b -w 22 10.0.0.255 >/dev/null 2>&1 &')
    time.sleep(4)  # let the controller event-loop queue build up
    pin_mid = int(sh(['grep', '-c', 'PACKET_IN', '/tmp/probe_ctrl.log']).stdout.strip() or 0)
    high, fails = [], 0
    for _ in range(10):
        t = setup_flow_time(net, 'h1', 'h12')
        if t is not None:
            high.append(t)
        else:
            fails += 1
        time.sleep(0.4)
    rate = (pin_mid - pin_before) / 4.0
    print(f'flood PACKET_IN rate ~ {rate:.0f}/s; high-load setup failures (timeout): {fails}/10')

    # count PACKET_IN seen by controller
    pin = sh(['grep', '-c', 'PACKET_IN', '/tmp/probe_ctrl.log'])
    print('=' * 50)
    print(f'PACKET_IN logged by controller: {pin.stdout.strip()}')
    if low:
        print(f'LOW  load flow-setup: n={len(low)} mean={statistics.mean(low):.1f}ms '
              f'median={statistics.median(low):.1f}ms max={max(low):.1f}ms')
    if high:
        print(f'HIGH load flow-setup: n={len(high)} mean={statistics.mean(high):.1f}ms '
              f'median={statistics.median(high):.1f}ms max={max(high):.1f}ms')
    if low and high:
        ratio = statistics.mean(high) / statistics.mean(low)
        print(f'>>> HIGH/LOW ratio = {ratio:.2f}x  '
              f'({"FEASIBLE: load affects latency" if ratio > 1.5 else "WEAK: controller not saturated enough"})')
    print('=' * 50)

    net.stop()
    ryu.terminate()
    sh(['mn', '-c'])

if __name__ == '__main__':
    try:
        main()
    finally:
        subprocess.run(['pkill', '-f', 'probe_ctrl'], capture_output=True)
        subprocess.run(['mn', '-c'], capture_output=True)
