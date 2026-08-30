from manim import *
import json
from pathlib import Path

ROOT=Path(__file__).parent
SHEET=json.loads((ROOT/"beat_sheet.json").read_text())
DURS={b["beat_id"]:float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG,INK,GRAY="#F0EAD6","#000000","#4D4D4D"
TEAL,VERM,YELLOW,BLUE="#009E73","#D55E00","#F0E442","#0072B2"
config.background_color=BG

def tx(s,size=28,color=INK,weight=NORMAL):
    # Pango on this Mac can collapse isolated single spaces at glyph boundaries.
    return Text(s.replace(" ","  "),font="EB Garamond",font_size=size,color=color,weight=weight)
def header(s):
    t=tx(s,38,INK,SEMIBOLD).to_edge(UP,buff=.25)
    return VGroup(t,Line(LEFT*6.4,RIGHT*6.4,color=GRAY,stroke_width=1).next_to(t,DOWN,buff=.12))
def node(label,pos,color=BLUE):
    c=Circle(.38,color=color,stroke_width=4,fill_color=color,fill_opacity=.08).move_to(pos)
    return VGroup(c,tx(label,24,INK,SEMIBOLD).move_to(c))
def edge(a,b,color=GRAY,width=4): return Line(a.get_center(),b.get_center(),color=color,stroke_width=width).set_z_index(-1)
def ledger(lines,title="LEDGER"):
    return VGroup(tx(title,24,INK,SEMIBOLD),*[tx(x,25) for x in lines]).arrange(DOWN,aligned_edge=LEFT,buff=.3).move_to([4.25,.05,0])
def star():
    ps={"c":(-3.1,0,0),"a":(-5.2,1.8,0),"b":(-1.0,1.8,0),"d":(-5.2,-1.8,0),"e":(-1.0,-1.8,0)}
    ns={k:node(k,p) for k,p in ps.items()}; es={("c",v):edge(ns["c"],ns[v]) for v in "abde"}
    return ns,es

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG; getattr(self,"draw_"+self.BID)()
        left=DURS[self.BID]-self.renderer.time
        if left>0:self.wait(left)
    def draw_B01(self):
        self.add(header("Adversary · a star exposes the full gap"));ns,es=star();self.add(*es.values(),*ns.values())
        chosen=es[("c","a")];self.play(chosen.animate.set_color(VERM).set_stroke(width=7),run_time=1.5)
        self.play(ns["c"][0].animate.set_fill(TEAL,.75),ns["a"][0].animate.set_fill(TEAL,.75),run_time=1.5)
        self.play(Write(ledger(["ALGORITHM   {c, a}   →  2","OPTIMUM      {c}      →  1","GAP                 TWICE"])),run_time=2)
    def draw_B02(self):
        self.add(header("Second instance · the same rule can be exact"))
        ns=[node(str(i+1),[-5+1.7*i,0,0]) for i in range(4)];es=[edge(ns[i],ns[i+1]) for i in range(3)];self.add(*es,*ns)
        self.play(es[1].animate.set_color(TEAL).set_stroke(width=7),run_time=1.4)
        self.play(ns[1][0].animate.set_fill(TEAL,.75),ns[2][0].animate.set_fill(TEAL,.75),es[0].animate.set_stroke(opacity=.18),es[2].animate.set_stroke(opacity=.18),run_time=1.8)
        self.play(Write(ledger(["CHOSEN EDGE   2 — 3","COVER            {2, 3}","OPTIMUM         2","GAP               EXACT"])),run_time=2)
    def draw_B03(self):
        self.add(header("Predict the hidden structure"))
        left=tx("STAR\n2 versus 1",31,INK,SEMIBOLD).move_to([-3.5,.3,0]);right=tx("PATH\n2 versus 2",31,INK,SEMIBOLD).move_to([3.5,.3,0]);self.add(left,right)
        q=tx("WHAT  MUST  ALL  CHOSEN  EDGES  SHARE?",33,INK,SEMIBOLD).to_edge(DOWN,buff=.55);wash=SurroundingRectangle(q,buff=.18,fill_color=YELLOW,fill_opacity=.35,stroke_opacity=0).set_z_index(-1)
        self.play(FadeIn(wash),Write(q),run_time=2)
    def draw_B04(self):
        self.add(header("Chosen edges cannot share endpoints"));ns,es=star();self.add(*es.values(),*ns.values());first=es[("c","a")]
        self.play(first.animate.set_color(TEAL).set_stroke(width=7),run_time=1.4)
        others=VGroup(*[v for k,v in es.items() if k!=("c","a")]);self.play(others.animate.set_stroke(opacity=.12),run_time=1.8)
        self.play(Write(ledger(["SELECT   c — a","DELETE   every incident edge","REMAINING EDGES   0","CHOSEN EDGES   DISJOINT"])),run_time=2)
    def draw_B05(self):
        self.add(header("A matching becomes a lower-bound certificate"))
        pairs=[]
        for i,y in enumerate([1.7,0,-1.7],1):
            a=node(chr(96+2*i-1),[-4.9,y,0]);b=node(chr(96+2*i),[-2.5,y,0]);e=edge(a,b,TEAL,7);pairs.append((a,b,e));self.add(e,a,b)
        self.play(*[p[2].animate.set_color(TEAL) for p in pairs],run_time=1.2)
        self.play(Write(ledger(["3  DISJOINT  CHOSEN  EDGES","ANY COVER PAYS ≥ 1 EACH","PAYMENTS CANNOT BE REUSED","OPTIMUM IS AT LEAST 3"])),run_time=2.2)
    def draw_B06(self):
        self.add(header("The license · maximal matching plus disjointness"))
        left=VGroup(tx("ALGORITHM",24,INK,SEMIBOLD),tx("takes both endpoints",29),tx("2 per chosen edge",30,INK,SEMIBOLD)).arrange(DOWN,buff=.35).move_to([-3.5,.2,0])
        right=VGroup(tx("OPTIMUM",24,INK,SEMIBOLD),tx("must touch each edge",29),tx("at least 1 per edge",30,INK,SEMIBOLD)).arrange(DOWN,buff=.35).move_to([3.5,.2,0])
        self.play(Write(left),run_time=1.8);self.play(Write(right),run_time=1.8)
        self.play(Write(tx("TWO-FOR-ONE  CERTIFICATE",34,INK,SEMIBOLD).to_edge(DOWN,buff=.35)),run_time=1.2)
    def draw_B07(self):
        self.add(header("Payoff · wasteful, but never unbounded"));ns,es=star();self.add(*es.values(),*ns.values());es[("c","a")].set_color(VERM).set_stroke(width=7);ns["c"][0].set_fill(TEAL,.75);ns["a"][0].set_fill(TEAL,.75)
        self.play(Write(ledger(["SELECTED EDGES      1","OPTIMUM ≥             1","ALGORITHM             2","WORST MULTIPLE   TWICE"])),run_time=2)
        self.play(Write(tx("A  GUARANTEE — NOT  A  GUESS",33,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1.4)

class B01(ReelBeat):BID="B01"
class B02(ReelBeat):BID="B02"
class B03(ReelBeat):BID="B03"
class B04(ReelBeat):BID="B04"
class B05(ReelBeat):BID="B05"
class B06(ReelBeat):BID="B06"
class B07(ReelBeat):BID="B07"
