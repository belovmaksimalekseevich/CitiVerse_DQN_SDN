# -*- coding: utf-8 -*-
"""Generate a draw.io (.drawio) topology using real Cisco stencils.
Open in app.diagrams.net (or desktop draw.io), then export PNG/SVG.
Pure python, no deps."""
import os, math
from xml.sax.saxutils import escape

OUT = [r"C:\Users\Максим\OneDrive\Рабочий стол\бонч\магистратура\статьи\статья dqn_sdn\CitiVerse_topology.drawio",
       r"C:\Users\Максим\Downloads\citiverse_results\CitiVerse_topology.drawio"]

# ---- topology (hardcoded to match topology_data.py) ----
ZONES = {
 'res1': {'sw': list(range(1,7)),  'ctrl':0, 'name':'Residential-1', 'color':'#1B78B3'},
 'res2': {'sw': list(range(7,12)), 'ctrl':1, 'name':'Residential-2', 'color':'#2E8BC0'},
 'com1': {'sw': list(range(12,16)),'ctrl':2, 'name':'Commercial-1',  'color':'#2E8B57'},
 'com2': {'sw': list(range(16,19)),'ctrl':3, 'name':'Commercial-2',  'color':'#3CB371'},
 'ind':  {'sw': list(range(19,21)),'ctrl':4, 'name':'Industrial',    'color':'#C8772E'},
}
PORTS = [6653,6654,6655,6656,6657]
SW_ZONE = {sw:z for z,d in ZONES.items() for sw in d['sw']}
ZONE_CTRL = {z:d['ctrl'] for z,d in ZONES.items()}
INTER = [('res1','com1',15,6,12), ('res1','res2',8,6,7),
         ('res2','com2',12,11,16), ('com1','ind',20,15,19)]  # za,zb,ms,sa,sb

# ---- layout (matplotlib-like coords, y up) ----
zone_center = {'res1':(1.5,9.2),'res2':(9.2,10.2),'com1':(4.7,4.6),
               'com2':(11.4,4.6),'ind':(5.7,0.2)}
ctrl_off = {'res1':(-2.9,0.0),'res2':(3.3,0.6),'com1':(-3.1,0.0),
            'com2':(3.3,0.0),'ind':(0.0,-2.4)}

def grid(center,n,dx=1.25,dy=1.30,cols=3):
    cx,cy=center; rows=math.ceil(n/cols); out=[]
    for i in range(n):
        r,c=divmod(i,cols); nc=min(cols,n-r*cols)
        out.append((cx+(c-(nc-1)/2)*dx, cy-(r-(rows-1)/2)*dy))
    return out

pos={}
for z,d in ZONES.items():
    for sw,p in zip(d['sw'],grid(zone_center[z],len(d['sw']))): pos[sw]=p
cpos={d['ctrl']:(zone_center[z][0]+ctrl_off[z][0], zone_center[z][1]+ctrl_off[z][1])
      for z,d in ZONES.items()}

SCALE=80; OX=5.0; OY=13.0   # x_px=(x+OX)*S ; y_px=(OY-y)*S  (flip y)
def X(x): return int((x+OX)*SCALE)
def Y(y): return int((OY-y)*SCALE)

SW_W, SW_H = 52, 44
CT_W, CT_H = 52, 60

cells=[]
def cell(s): cells.append(s)

# zone background rectangles (behind everything)
for z,d in ZONES.items():
    xs=[pos[s][0] for s in d['sw']]; ys=[pos[s][1] for s in d['sw']]
    x0=X(min(xs))-46; x1=X(max(xs))+46+SW_W
    y1=Y(min(ys))+46+SW_H; y0=Y(max(ys))-58
    w=x1-x0; h=y1-y0
    style=(f"rounded=1;arcSize=8;fillColor={d['color']};opacity=15;"
           f"strokeColor={d['color']};dashed=0;verticalAlign=top;align=center;"
           f"fontStyle=1;fontColor={d['color']};fontSize=14;")
    val=f"{d['name']} ({len(d['sw'])} sw)"
    cell(f'<mxCell id="zone_{z}" value="{escape(val)}" style="{style}" vertex="1" parent="1">'
         f'<mxGeometry x="{x0}" y="{y0}" width="{w}" height="{h}" as="geometry"/></mxCell>')

# control-plane channels (dashed) — draw before nodes
ei=0
for sw,z in SW_ZONE.items():
    c=ZONE_CTRL[z]
    style=(f"endArrow=none;dashed=1;dashPattern=2 3;strokeColor={ZONES[z]['color']};"
           f"strokeWidth=1;opacity=45;html=1;")
    cell(f'<mxCell id="cp{ei}" style="{style}" edge="1" parent="1" '
         f'source="s{sw}" target="c{c}"><mxGeometry relative="1" as="geometry"/></mxCell>'); ei+=1

# intra-zone data links
for z,d in ZONES.items():
    for a,b in zip(d['sw'][:-1],d['sw'][1:]):
        cell(f'<mxCell id="il{ei}" style="endArrow=none;strokeColor=#6F6F6F;strokeWidth=2;html=1;" '
             f'edge="1" parent="1" source="s{a}" target="s{b}">'
             f'<mxGeometry relative="1" as="geometry"/></mxCell>'); ei+=1

# inter-zone backbone with ms labels
for za,zb,ms,sa,sb in INTER:
    cell(f'<mxCell id="bb{ei}" value="{ms} ms" '
         f'style="endArrow=none;strokeColor=#1B1B1B;strokeWidth=3;html=1;fontSize=11;'
         f'labelBackgroundColor=#ffffff;" edge="1" parent="1" source="s{sa}" target="s{sb}">'
         f'<mxGeometry relative="1" as="geometry"/></mxCell>'); ei+=1

# switches (Cisco workgroup switch stencil)
for sw,z in SW_ZONE.items():
    x=X(pos[sw][0]); y=Y(pos[sw][1])
    style=("sketch=0;outlineConnect=0;html=1;whiteSpace=wrap;fillColor="
           f"{ZONES[z]['color']};strokeColor=#ffffff;verticalLabelPosition=bottom;"
           "verticalAlign=top;shape=mxgraph.cisco.switches.workgroup_switch;")
    cell(f'<mxCell id="s{sw}" value="s{sw}" style="{style}" vertex="1" parent="1">'
         f'<mxGeometry x="{x}" y="{y}" width="{SW_W}" height="{SW_H}" as="geometry"/></mxCell>')

# controllers (Cisco standard host / server stencil)
for z,d in ZONES.items():
    c=d['ctrl']; x=X(cpos[c][0]); y=Y(cpos[c][1])
    style=("sketch=0;outlineConnect=0;html=1;whiteSpace=wrap;fillColor=#5D7B9D;"
           "strokeColor=#ffffff;verticalLabelPosition=bottom;verticalAlign=top;"
           "shape=mxgraph.cisco.servers.standard_host;")
    cell(f'<mxCell id="c{c}" value="C{c} :{PORTS[c]}" style="{style}" vertex="1" parent="1">'
         f'<mxGeometry x="{x}" y="{y}" width="{CT_W}" height="{CT_H}" as="geometry"/></mxCell>')

body="\n".join(cells)
xml=('<mxfile host="app.diagrams.net" type="device"><diagram name="CitiVerse-topology">'
     '<mxGraphModel dx="1400" dy="900" grid="1" gridSize="10" guides="1" tooltips="1" '
     'connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" '
     'pageHeight="1200" math="0" shadow="0"><root>'
     '<mxCell id="0"/><mxCell id="1" parent="0"/>'
     f'{body}'
     '</root></mxGraphModel></diagram></mxfile>')

for o in OUT:
    os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(o,'w',encoding='utf-8') as f: f.write(xml)
    print("saved", o)
