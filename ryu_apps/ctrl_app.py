# ryu_apps/ctrl_app.py
import time
import logging
from collections import defaultdict

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import (
    CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER, set_ev_cls,
)
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet

LOG = logging.getLogger(__name__)


class CitiCtrlApp(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.packet_in_counts = defaultdict(int)
        self.packet_in_rates = {}
        self.window_start = time.time()
        self.window_size = 5.0
        self._barrier_received = {}
        self._datapaths = {}       # dpid -> datapath (for echo RTT loop)
        self._echo_pending = {}    # dpid -> float send timestamp
        hub.spawn(self._echo_loop)

    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self._datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self._datapaths.pop(dp.id, None)
            self._echo_pending.pop(dp.id, None)

    def _echo_loop(self):
        """Send OFPEchoRequest every 2s per connected switch to measure real RTT."""
        hub.sleep(5.0)
        while True:
            hub.sleep(2.0)
            for dpid, dp in list(self._datapaths.items()):
                try:
                    req = dp.ofproto_parser.OFPEchoRequest(dp, data=b'rtt')
                    self._echo_pending[dpid] = time.time()
                    dp.send_msg(req)
                except Exception as e:
                    LOG.debug(f'EchoRequest failed dpid={dpid}: {e}')

    @set_ev_cls(ofp_event.EventOFPEchoReply, MAIN_DISPATCHER)
    def _echo_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        t_sent = self._echo_pending.pop(dpid, None)
        if t_sent is not None:
            rtt_ms = (time.time() - t_sent) * 1000.0
            LOG.info(f'ECHO_RTT {dpid} {rtt_ms:.3f}')

    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        # Table-miss: send to controller, never expire
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp, priority=0, match=match, instructions=inst,
            idle_timeout=0, hard_timeout=0,
        )
        dp.send_msg(mod)

        barrier = parser.OFPBarrierRequest(dp)
        dp.send_msg(barrier)

    @set_ev_cls(ofp_event.EventOFPBarrierReply, MAIN_DISPATCHER)
    def barrier_reply_handler(self, ev):
        dpid = ev.msg.datapath.id
        LOG.debug(f'Barrier reply from dpid={dpid} — FlowMod processed')
        self._barrier_received[dpid] = True

    def wait_for_barrier(self, dpid, timeout=3.0):
        """Block until barrier reply received or timeout."""
        self._barrier_received.setdefault(dpid, False)
        t0 = time.time()
        while not self._barrier_received.get(dpid, False):
            if time.time() - t0 > timeout:
                LOG.warning(f'Barrier timeout for dpid={dpid}')
                return False
            time.sleep(0.05)
        self._barrier_received[dpid] = False
        return True

    # ------------------------------------------------------------------
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        dpid = dp.id
        ofp = dp.ofproto
        parser = dp.ofproto_parser

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Filter LLDP — do not count in PACKET_IN rate
        from ryu.lib.packet import lldp as lldp_proto
        if eth.ethertype == lldp_proto.LLDP_MAC_NEAREST_BRIDGE:
            return

        # Structured log for ControllerMonitor
        LOG.info(f'PACKET_IN {dpid} {time.time():.6f}')
        self.packet_in_counts[dpid] += 1
        self._update_rates()

        dst = eth.dst
        src = eth.src
        in_port = msg.match['in_port']

        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src] = in_port
        out_port = self.mac_to_port[dpid].get(dst, ofp.OFPP_FLOOD)

        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(dp, 1, match, actions)

        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data,
        )
        dp.send_msg(out)

    # ------------------------------------------------------------------
    def _add_flow(self, dp, priority, match, actions):
        parser = dp.ofproto_parser
        ofp = dp.ofproto
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=dp,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=30,
            hard_timeout=120,
        )
        dp.send_msg(mod)
        LOG.info(f'FLOW_MOD {dp.id} {time.time():.6f}')

    def _update_rates(self):
        now = time.time()
        elapsed = now - self.window_start
        if elapsed >= self.window_size:
            for dpid, cnt in self.packet_in_counts.items():
                self.packet_in_rates[dpid] = cnt / elapsed
            self.packet_in_counts.clear()
            self.window_start = now

    def get_packet_in_rate(self, dpid):
        return self.packet_in_rates.get(dpid, 0.0)
