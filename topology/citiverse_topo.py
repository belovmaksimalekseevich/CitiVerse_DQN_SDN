# topology/citiverse_topo.py

import sys
import subprocess
import time
from functools import partial

for _p in [
    '/usr/lib/python3/dist-packages',
    '/usr/local/lib/python3/dist-packages',
    '/usr/lib/python3.8/dist-packages',
    '/usr/local/lib/python3.8/dist-packages',
    '/usr/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.10/dist-packages',
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import TCLink
from mininet.log import setLogLevel

from topology.topology_data import (
    ZONES, N_SWITCHES, SWITCHES,
    get_intra_links, get_inter_delay, SW_ZONE,
    MININET_INTER_LINKS, CTRL_PORTS, INTRA_DELAY_MS,
)

OVSSwitch13 = partial(OVSSwitch, protocols='OpenFlow13')


def _make_inter_links():
    """Return 4 loop-free inter-zone links with correct delays from LATENCY_MATRIX."""
    links = []
    for zone_a, zone_b in MININET_INTER_LINKS:
        # Use last switch of zone_a and first switch of zone_b as gateway pair
        gw_a = ZONES[zone_a]['switches'][-1]
        gw_b = ZONES[zone_b]['switches'][0]
        delay_ms = get_inter_delay(zone_a, zone_b)
        links.append({
            'src':      gw_a,
            'dst':      gw_b,
            'delay_ms': delay_ms,
        })
    return links


class CitiverseTopo(Topo):
    def build(self):
        switches = {}
        for i in range(1, N_SWITCHES + 1):
            switches[i] = self.addSwitch(
                f's{i}',
                protocols='OpenFlow13',
                failMode='standalone',
            )

        # Intra-zone: linear chain, 100 Mbps, 3 ms
        for link in get_intra_links():
            self.addLink(
                switches[link['src']], switches[link['dst']],
                delay=f"{link['delay_ms']}ms",
                bw=100,
                loss=0,
                max_queue_size=1000,
            )

        # Inter-zone: 4 spanning-tree links with real zone delays
        for link in _make_inter_links():
            self.addLink(
                switches[link['src']], switches[link['dst']],
                delay=f"{link['delay_ms']}ms",
                bw=50,
                loss=0,
                max_queue_size=1000,
            )

        # 10 hosts: 2 per zone (first and last switch)
        host_num = 1
        for zone_name, zone_data in ZONES.items():
            sws = zone_data['switches']
            for sw_idx in [0, -1]:
                sw = sws[sw_idx]
                h = self.addHost(f'h{host_num}', ip=f'10.0.0.{host_num}/24')
                self.addLink(h, switches[sw])
                host_num += 1


def create_network():
    setLogLevel('warning')
    topo = CitiverseTopo()
    net = Mininet(
        topo=topo,
        controller=None,   # no controller here — assigned by environment.py via ovs-vsctl
        switch=OVSSwitch13,
        link=TCLink,
        autoSetMacs=True,
    )
    return net


def set_all_fail_standalone(net):
    """Ensure fail_mode=standalone on all switches after net.start()."""
    for i in range(1, N_SWITCHES + 1):
        sw = net.get(f's{i}')
        sw.cmd(f'ovs-vsctl set bridge s{i} fail_mode=standalone')
        sw.cmd(f'ovs-vsctl set bridge s{i} protocols=OpenFlow13')


def get_all_switch_names():
    """Returns list of OvS switch names: ['s1', 's2', ..., 's20']"""
    return [f's{i}' for i in range(1, N_SWITCHES + 1)]


def reset_all_flows(net=None):
    """Clear all flow tables across all switches. Call after each episode reset."""
    for i in range(1, N_SWITCHES + 1):
        subprocess.run(
            ['ovs-ofctl', '-O', 'OpenFlow13', 'del-flows', f's{i}'],
            capture_output=True, timeout=5,
        )


def assign_all_controllers(assignments, ctrl_ports=None):
    """
    Bulk-assign all switches to controllers via ovs-vsctl.
    assignments: array-like of ctrl_idx per switch (len=N_SWITCHES, 0-indexed)
    ctrl_ports:  list of Ryu ports, defaults to CTRL_PORTS [6653..6657]
    """
    if ctrl_ports is None:
        ctrl_ports = CTRL_PORTS
    for sw_idx, ctrl_idx in enumerate(assignments):
        sw = f's{sw_idx + 1}'
        port = ctrl_ports[int(ctrl_idx)]
        subprocess.run(
            ['ovs-vsctl', 'set-controller', sw, f'tcp:127.0.0.1:{port}'],
            capture_output=True, timeout=5,
        )


if __name__ == '__main__':
    from mininet.cli import CLI
    setLogLevel('info')
    net = create_network()
    net.start()
    set_all_fail_standalone(net)
    print('Waiting 10s for OvS initialization...')
    time.sleep(10)
    CLI(net)
    net.stop()
