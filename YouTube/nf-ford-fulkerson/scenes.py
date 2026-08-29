from manim import *
import json
from pathlib import Path

ROOT=Path(__file__).parent
SHEET=json.loads((ROOT/"beat_sheet.json").read_text())
DURS={b["beat_id"]:float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG,INK,GRAY="#F0EAD6","#000000","#4D4D4D"
TEAL,VERM,YELLOW,BLUE="#009E73","#D55E00","#F0E442","#0072B2"
config.background_color=BG

def tx(s,size=29,color=INK,weight=NORMAL): return Text(s,font="EB Garamond",font_size=size,color=color,weight=weight)
def header(s):
    t=tx(s,39,INK,SEMIBOLD).to_edge(UP,buff=.25)
    return VGroup(t,Line(LEFT*6.4,RIGHT*6.4,color=GRAY,stroke_width=1).next_to(t,DOWN,buff=.12))
def node(s,p):
    c=Circle(.4,color=BLUE,stroke_width=4,fill_color=BLUE,fill_opacity=.1).move_to(p)
    return VGroup(c,tx(s,28,INK,SEMIBOLD).move_to(c))

P={"s":(-4.7,0,0),"a":(-1.8,1.7,0),"b":(-1.8,-1.7,0),"t":(1.3,0,0)}
EDGES=[("s","a"),("s","b"),("a","b"),("a","t"),("b","t")]
def graph(flows=None):
    flows=flows or {}
    ns={k:node(k,v) for k,v in P.items()}
    es={}
    labs={}
    for u,v in EDGES:
        ar=Arrow(ns[u].get_center(),ns[v].get_center(),buff=.46,color=GRAY,stroke_width=4,max_tip_length_to_length_ratio=.12).set_z_index(-1)
        es[(u,v)]=ar
        f=flows.get((u,v),0)
        lab=tx(f"{f}/1",23).move_to(ar.get_center()+UP*.25)
        if (u,v)==("a","b"): lab.next_to(ar,RIGHT,buff=.12)
        labs[(u,v)]=lab
    return ns,es,labs
def table(rows,title="FLOW / CAPACITY"):
    body=VGroup(tx(title,25,INK,SEMIBOLD),*[tx(r,24) for r in rows]).arrange(DOWN,aligned_edge=LEFT,buff=.22).move_to([4.35,.1,0])
    return body
def add_graph(scene,flows=None):
    ns,es,labs=graph(flows)
    scene.add(*ns.values(),*es.values(),*labs.values())
    return ns,es,labs

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        getattr(self,"draw_"+self.BID)()
        left=DURS[self.BID]-self.renderer.time
        if left>0:self.wait(left)
    def draw_B01(self):
        self.add(header("A legal path that traps the forward view")); ns,es,labs=add_graph(self)
        path=[("s","a"),("a","b"),("b","t")]
        self.play(*[es[e].animate.set_color(VERM) for e in path],run_time=2)
        for e in path:self.play(Transform(labs[e],tx("1/1",23).move_to(labs[e])),run_time=.5)
        self.play(Write(table(["PATH   s → a → b → t","BOTTLENECK   1","TOTAL FLOW   1"])),run_time=2)
        self.play(Write(tx("FORWARD  PATHS:  BLOCKED",31,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1.2)
    def draw_B02(self):
        self.add(header("Reset · two outer paths reach two")); ns,es,labs=add_graph(self)
        p1=[("s","a"),("a","t")];p2=[("s","b"),("b","t")]
        self.play(*[es[e].animate.set_color(TEAL) for e in p1],run_time=2)
        self.play(*[es[e].animate.set_color(TEAL) for e in p2],run_time=2)
        for e in p1+p2:self.add(tx("1/1",23).move_to(labs[e]));self.remove(labs[e])
        self.play(Write(table(["PATH 1   s → a → t","PATH 2   s → b → t","TOTAL FLOW   2"])),run_time=2)
    def draw_B03(self):
        self.add(header("Predict the missing permission"));ns,es,labs=add_graph(self,{e:1 for e in [("s","a"),("a","b"),("b","t")]})
        for e in [("s","a"),("a","b"),("b","t")]:es[e].set_color(VERM)
        q=tx("WHAT  EDGE  LETS  THIS  CHOICE  CHANGE?",34,INK,SEMIBOLD).to_edge(DOWN,buff=.4)
        wash=SurroundingRectangle(q,buff=.18,fill_color=YELLOW,fill_opacity=.35,stroke_opacity=0).set_z_index(-1)
        self.play(FadeIn(wash),Write(q),run_time=2)
    def draw_B04(self):
        self.add(header("Residual graph · used flow creates an undo edge"));ns,es,labs=add_graph(self,{e:1 for e in [("s","a"),("a","b"),("b","t")]})
        rev=CurvedArrow(ns["b"].get_center()+RIGHT*.1,ns["a"].get_center()+RIGHT*.1,angle=-TAU/5,color=TEAL,stroke_width=5)
        self.play(Create(rev),run_time=2)
        self.play(Write(tx("b → a   residual capacity 1",27).move_to([4.15,1.0,0])),run_time=1.5)
        self.play(Write(tx("UNDO  1  UNIT  ON  a → b",31,INK,SEMIBOLD).to_edge(DOWN,buff=.35)),run_time=1.5)
    def draw_B05(self):
        self.add(header("Repair path · s → b → a → t"));ns,es,labs=add_graph(self,{e:1 for e in [("s","a"),("a","b"),("b","t")]})
        rev=CurvedArrow(ns["b"].get_center()+RIGHT*.1,ns["a"].get_center()+RIGHT*.1,angle=-TAU/5,color=TEAL,stroke_width=5);self.add(rev)
        self.play(es[("s","b")].animate.set_color(TEAL),rev.animate.set_color(TEAL),es[("a","t")].animate.set_color(TEAL),run_time=2)
        self.play(es[("a","b")].animate.set_color(GRAY).set_stroke(opacity=.3),run_time=1.5)
        rows=table(["ADD   s → b","CANCEL   a → b","ADD   a → t","TOTAL FLOW   2"])
        self.play(Write(rows),run_time=2)
        self.play(Write(tx("MISTAKE  REPAIRED",34,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1)
    def draw_B06(self):
        self.add(header("The characteristic: reversible commitments"));ns,es,labs=add_graph(self)
        cards=VGroup(tx("FORWARD:  ROOM  TO  ADD",22),tx("BACKWARD:  ROOM  TO  UNDO",22),tx("INTEGRAL:  WHOLE-UNIT  STEPS",22)).arrange(DOWN,aligned_edge=LEFT,buff=.72).move_to([4.45,.1,0])
        self.play(Write(cards[0]),run_time=1.5);self.play(Write(cards[1]),run_time=1.5);self.play(Write(cards[2]),run_time=1.5)
    def draw_B07(self):
        self.add(header("No residual path leaves a saturated cut"));ns,es,labs=add_graph(self,{e:1 for e in [("s","a"),("s","b"),("a","t"),("b","t")]})
        for e in [("s","a"),("s","b"),("a","t"),("b","t")]:es[e].set_color(TEAL)
        cut=DashedLine([-.1,3,0],[-.1,-3,0],color=VERM,stroke_width=5)
        self.play(Create(cut),run_time=1.5)
        self.play(Write(table(["REACHABLE FROM s   {s}","CUT EDGES   s→a, s→b","CUT CAPACITY   2","FLOW VALUE   2"])),run_time=2)
        self.play(Write(tx("FLOW  =  CUT  CAPACITY  →  MAXIMUM",30,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1.5)
    def draw_B08(self):
        self.add(header("Payoff · the opening trap is repairable"));ns,es,labs=add_graph(self,{e:1 for e in [("s","a"),("a","b"),("b","t")]})
        bad=VGroup(*[es[e] for e in [("s","a"),("a","b"),("b","t")]])
        self.play(bad.animate.set_color(VERM),run_time=1.5)
        self.play(es[("a","b")].animate.set_color(GRAY).set_stroke(opacity=.25),es[("s","b")].animate.set_color(TEAL),es[("a","t")].animate.set_color(TEAL),es[("s","a")].animate.set_color(TEAL),es[("b","t")].animate.set_color(TEAL),run_time=2.5)
        self.play(Write(tx("BAD  PATH  ≠  BAD  FINAL  FLOW",34,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1.5)

class B01(ReelBeat):BID="B01"
class B02(ReelBeat):BID="B02"
class B03(ReelBeat):BID="B03"
class B04(ReelBeat):BID="B04"
class B05(ReelBeat):BID="B05"
class B06(ReelBeat):BID="B06"
class B07(ReelBeat):BID="B07"
class B08(ReelBeat):BID="B08"
