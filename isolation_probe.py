#!/usr/bin/env python3
"""Per-controller ISOLATION probe.
Goal: load ONE controller via a pure-python raw-socket flooder (varying src MAC ->
fresh table-miss each packet -> sustained PACKET_IN to exactly that switch's
controller, staying local). Then measure flow-setup latency on:
  (a) another switch of the SAME (loaded) controller   -> should RISE
  (b) a switch of a DIFFERENT (idle) controller         -> should stay LOW
If (a) rises and (b) doesn't, per-controller load is isolatable -> hybrid works.
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
VENVPY = '/home/maksim/dqn_env/bin/python3'
APP = 'ryu_apps/ctrl_app.py'
PORTS = [6653, 6654]
NSW = 6  # s1..s3 -> ctrlA(6653), s4..s6 -> ctrlB(6654)

FLOODER = r'''
import socket, sys, time, struct, random
iface, dst_mac, rate, dur = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
dstb = bytes(int(x,16) for x in dst_mac.split(':'))
s = socket.socket(socket.AF_PACKET, socket.SOCK_RAW); s.bind((iface,0))
payload = b'x'*46; end = time.time()+dur; iv = 1.0/rate if rate>0 else 0; n=0
while time.time() < end:
    srcb = bytes([0x02]+[random.randint(0,255) for _ in range(5)])
    s.send(dstb+srcb+struct.pack('!H',0x0800)+payload); n+=1
    if iv: time.sleep(iv)
open('/tmp/flood_%s.cnt'%iface,'w').write(str(n))
'''

def sh(cmd):
    return subprocess.run(cmd, capture_output=True, timeout=15, text=True)

class T(Topo):
    def build(self):
        sws = [self.addSwitch(f's{i}', protocols='OpenFlow13', failMode='secure')
               for i in range(1, NSW+1)]
        for i in range(len(sws)-1):
            self.addLink(sws[i], sws[i+1], delay='3ms', bw=100, max_queue_size=1000)
        hn = 1
        for s in sws:
            for _ in range(2):
                self.addHost(f'h{hn}', ip=f'10.0.0.{hn}/24'); self.addLink(f'h{hn}', s); hn += 1

def reset_miss():
    for i in range(1, NSW+1):
        sh(['ovs-ofctl','-O','OpenFlow13','del-flows',f's{i}'])
        sh(['ovs-ofctl','-O','OpenFlow13','add-flow',f's{i}','priority=0,actions=CONTROLLER:65535'])

def setup_latency(net, a, b):
    """flow-setup latency for fresh flow between same-switch hosts a,b (ms)."""
    ha, hb = net.get(a), net.get(b)
    # reinstall miss on the host pair's switch path is covered by reset_miss()
    ha.cmd('ip neigh flush all'); hb.cmd('ip neigh flush all')
    t0 = time.time()
    out = ha.cmd(f'ping -c1 -W3 {hb.IP()}')
    dt = (time.time()-t0)*1000.0
    return dt if ('1 received' in out or '1 packets received' in out) else None

def probe(net, a, b, n=8):
    vals = []
    for _ in range(n):
        reset_miss(); time.sleep(0.2)
        t = setup_latency(net, a, b)
        if t is not None: vals.append(t)
        time.sleep(0.3)
    return vals

def main():
    setLogLevel('warning'); sh(['mn','-c'])
    open('/tmp/flooder.py','w').write(FLOODER)
    ryus = [subprocess.Popen([RYU, APP, '--ofp-tcp-listen-port', str(p),
            '--log-file', f'/tmp/iso_{p}.log'], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL) for p in PORTS]
    time.sleep(4)
    net = Mininet(topo=T(), controller=None,
                  switch=partial(OVSSwitch, protocols='OpenFlow13'),
                  link=TCLink, autoSetMacs=True)
    net.start()
    for i in range(1, NSW+1):
        port = PORTS[0] if i <= 3 else PORTS[1]
        net.get(f's{i}').cmd(f'ovs-vsctl set-controller s{i} tcp:127.0.0.1:{port}')
    print('connecting...'); time.sleep(8)

    # hosts: s1->h1,h2 ; s2->h3,h4 ; s3->h5,h6 (ctrlA) ; s4->h7,h8 ; s5->h9,h10 ; s6->h11,h12 (ctrlB)
    # Probe pairs (same switch): ctrlA probe on s3=(h5,h6); ctrlB probe on s6=(h11,h12)
    baseA = probe(net, 'h5', 'h6')
    baseB = probe(net, 'h11', 'h12')

    # LOAD ctrlA only: flood from s1(h1->h2) and s2(h3->h4) with varying src MAC
    print('loading ctrlA via raw-socket flooders on s1,s2...')
    floods = []
    for src, dst in [('h1','h2'), ('h3','h4')]:
        hs = net.get(src); peer = net.get(dst)
        intf = hs.defaultIntf().name
        hs.cmd(f'{VENVPY} /tmp/flooder.py {intf} {peer.MAC()} 1000 20 >/dev/null 2>&1 &')
    time.sleep(4)
    loadA = probe(net, 'h5', 'h6')     # same controller as the flood -> should rise
    loadB = probe(net, 'h11', 'h12')   # other controller -> should stay low

    pinA = sh(['grep','-c','PACKET_IN','/tmp/iso_6653.log']).stdout.strip()
    pinB = sh(['grep','-c','PACKET_IN','/tmp/iso_6654.log']).stdout.strip()

    def stat(v): return f'n={len(v)} mean={statistics.mean(v):.1f} med={statistics.median(v):.1f} max={max(v):.1f}' if v else 'NONE'
    print('='*60)
    print(f'PACKET_IN: ctrlA(loaded)={pinA}  ctrlB(idle)={pinB}')
    print(f'ctrlA switch s3:  base [{stat(baseA)}]  ->  underLoad [{stat(loadA)}]')
    print(f'ctrlB switch s6:  base [{stat(baseB)}]  ->  underLoad [{stat(loadB)}]')
    if baseA and loadA and baseB and loadB:
        rA = statistics.mean(loadA)/statistics.mean(baseA)
        rB = statistics.mean(loadB)/statistics.mean(baseB)
        print(f'>>> ctrlA ratio={rA:.2f}x (loaded)   ctrlB ratio={rB:.2f}x (should ~1.0)')
        ok = rA > 1.5 and rB < 1.4
        print('>>> ISOLATION', 'CONFIRMED' if ok else 'WEAK/FAILED')
    print('='*60)
    net.stop()
    for r in ryus: r.terminate()
    sh(['mn','-c'])

if __name__ == '__main__':
    try:
        main()
    finally:
        subprocess.run(['pkill','-f','flooder.py'], capture_output=True)
        subprocess.run(['pkill','-f','iso_665'], capture_output=True)
        subprocess.run(['mn','-c'], capture_output=True)
