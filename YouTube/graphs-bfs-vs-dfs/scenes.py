from manim import *
import json
from pathlib import Path

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW = "#009E73", "#D55E00", "#F0E442"
BLUE, ORANGE, SKY = "#0072B2", "#E69F00", "#56B4E9"
config.background_color = BG

POS = {"S":(-4.9,0.4), "A":(-3.0,1.7), "B":(-3.0,-1.2), "C":(-1.0,2.1),
       "T":(-1.0,-1.2), "E":(0.8,1.45), "F":(2.25,2.2)}
EDGES = [("S","A"),("S","B"),("A","C"),("B","T"),("C","E"),("E","F")]

def tx(s, size=30, color=INK, weight=NORMAL, font="EB Garamond"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)

def heading(s):
    t = tx(s, 39, INK, SEMIBOLD).to_edge(UP, buff=.25)
    return VGroup(t, Line(LEFT*6.55, RIGHT*6.55, color=GRAY, stroke_width=1).next_to(t, DOWN, buff=.1))

def graph():
    edges = VGroup(*[Line([*POS[a],0],[*POS[b],0],color=GRAY,stroke_width=4) for a,b in EDGES])
    nodes = {}
    vg = VGroup(edges)
    for name,(x,y) in POS.items():
        c = Circle(.34,color=GRAY,fill_color=BG,fill_opacity=1,stroke_width=4).move_to([x,y,0])
        label = tx(name,26,INK,SEMIBOLD).move_to(c)
        nodes[name] = VGroup(c,label)
        vg.add(nodes[name])
    return vg, nodes

def panel(title, kind="QUEUE"):
    box = RoundedRectangle(width=3.5,height=4.4,corner_radius=.12,color=GRAY,stroke_width=2).move_to([4.55,-.15,0])
    hd = tx(title,28,INK,SEMIBOLD).next_to(box,UP,buff=.15)
    rule = tx("oldest  ←" if kind=="QUEUE" else "newest  ←",22,GRAY).move_to([4.55,1.65,0])
    return VGroup(box,hd,rule)

def cells(items, y=.85):
    out=VGroup()
    for i,name in enumerate(items):
        r=RoundedRectangle(width=.62,height=.55,corner_radius=.07,color=BLUE,fill_color=SKY,fill_opacity=.22)
        r.move_to([3.35+i*.72,y,0]); out.add(VGroup(r,tx(name,23,INK,SEMIBOLD).move_to(r)))
    return out

def dot(name, color):
    c=Circle(.31,color=color,fill_color=color,fill_opacity=.18)
    return VGroup(c,tx(name,22,INK,SEMIBOLD).move_to(c))

def visit(nodes, names, color=TEAL):
    return [nodes[n][0].animate.set_color(color).set_fill(color,opacity=.22) for n in names]

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        getattr(self,"draw_"+self.BID)()
        if self.renderer.time < DURS[self.BID]: self.wait(DURS[self.BID]-self.renderer.time)

    def draw_B01(self):
        self.add(heading("Breadth first: oldest frontier first")); g,n=graph(); p=panel("FIFO  QUEUE"); self.play(Create(g),Create(p),run_time=1.4)
        q=cells(["S"]); self.play(FadeIn(q),run_time=.4)
        order=[("S",["A","B"]),("A",["B","C"]),("B",["C","T"]),("T",["C"])]
        seen=[]
        for node,front in order:
            seen.append(node); self.play(*visit(n,[node]),Transform(q,cells(front)),run_time=.8)
        self.play(Write(tx("VISITED  " + "  →  ".join(seen),24,INK,SEMIBOLD).move_to([4.55,-1.55,0])),run_time=.8)
        self.play(Write(tx("TARGET  AT  DISTANCE  2",25,INK,SEMIBOLD).move_to([4.55,-2.05,0])),run_time=.7)

    def draw_B02(self):
        self.add(heading("Depth first: newest frontier first")); g,n=graph(); p=panel("LIFO  STACK","STACK"); self.play(Create(g),Create(p),run_time=1.3)
        q=cells(["S"]); self.play(FadeIn(q),run_time=.3)
        order=[("S",["B","A"]),("A",["B","C"]),("C",["B","E"]),("E",["B","F"]),("F",["B"])]
        seen=[]
        for node,front in order:
            seen.append(node); self.play(*visit(n,[node],ORANGE),Transform(q,cells(front)),run_time=.72)
        self.play(Write(tx("VISITED  " + "  →  ".join(seen),22,INK,SEMIBOLD).move_to([4.55,-1.55,0])),run_time=.7)
        self.play(Write(tx("T  IS  STILL  WAITING",25,INK,SEMIBOLD).move_to([4.55,-2.05,0])),run_time=.6)

    def draw_B03(self):
        self.add(heading("Predict before the next removal")); g,n=graph(); self.add(g); self.play(*visit(n,["S","A"]),run_time=.8)
        q=VGroup(tx("QUEUE   [ B , C ]   →   removes B",26),tx("STACK   [ B , C ]   →   removes C",26)).arrange(DOWN,aligned_edge=LEFT,buff=.3).move_to([3.8,.55,0])
        wash=SurroundingRectangle(q,color=YELLOW,fill_color=YELLOW,fill_opacity=.25,stroke_width=0,buff=.2).set_z_index(-1)
        self.play(AddTextLetterByLetter(q),FadeIn(wash),run_time=2)
        self.play(AddTextLetterByLetter(tx("Which reaches T sooner?  Commit.",30,INK,SEMIBOLD).move_to([3.8,-1.25,0])),run_time=1.4)

    def draw_B04(self):
        self.add(heading("Same graph. Change only the frontier.")); g,n=graph(); self.add(g)
        q=VGroup(tx("FIFO  QUEUE",27,INK,SEMIBOLD),cells(["B","C"],0)).move_to([4.35,1.15,0])
        self.play(FadeIn(q),run_time=.7); self.play(*visit(n,["B","T"]),Transform(q[1],cells(["C","T"],0).move_to(q[1])),run_time=1.8)
        result=tx("B  →  T",29,INK,SEMIBOLD).move_to([4.35,.15,0]); self.play(Write(result),run_time=.6)
        stack=VGroup(tx("LIFO  STACK",27,INK,SEMIBOLD),cells(["B","C"],0)).move_to([4.35,-1.05,0])
        self.play(ReplacementTransform(q,stack),run_time=1); self.play(*visit(n,["C","E","F"],ORANGE),Transform(result,tx("C  →  E  →  F",27,INK,SEMIBOLD).move_to([4.35,-2.0,0])),run_time=2)

    def draw_B05(self):
        self.add(heading("Widen the graph; the disciplines remain"))
        levels=[["S"],["A","B","C"],["D","E","F","G"]]; groups=VGroup()
        for j,lev in enumerate(levels):
            y=1.7-j*1.55; row=VGroup(*[dot(x,[BLUE,ORANGE,SKY][j]) for x in lev]).arrange(RIGHT,buff=.8).move_to([-2.7,y,0]); groups.add(row)
        for a in groups[0]:
            for b in groups[1]: self.add(Line(a.get_center(),b.get_center(),color=GRAY,stroke_width=2).set_z_index(-2))
        for i,a in enumerate(groups[1]):
            for b in groups[2][max(0,i-1):min(4,i+2)]: self.add(Line(a.get_center(),b.get_center(),color=GRAY,stroke_width=2).set_z_index(-2))
        self.play(Create(groups),run_time=1.5)
        notes=VGroup(tx("BFS: finish distance 1 first",27,INK,SEMIBOLD),tx("DFS: follow one branch first",27,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.45).move_to([3.8,.25,0]); self.play(AddTextLetterByLetter(notes),run_time=2)

    def draw_B06(self):
        self.add(heading("The characteristics that license BFS distance"))
        left=VGroup(tx("UNWEIGHTED  EDGES",29,INK,SEMIBOLD),tx("every  edge  costs  one  step",25,GRAY),Line(LEFT*2,RIGHT*2,color=TEAL,stroke_width=7),tx("equal  step  cost",23)).arrange(DOWN,buff=.28).move_to([-3.4,.3,0])
        right=VGroup(tx("FIFO  FRONTIER",29,INK,SEMIBOLD),tx("oldest  layer  leaves  first",25,GRAY),cells(["d","d","d+1"],0),tx("only  adjacent  layers",23)).arrange(DOWN,buff=.28).move_to([3.35,.3,0])
        self.play(Create(left),Create(right),run_time=2)
        cap=tx("INVARIANT:  frontier holds distance d and d + 1",29,INK,SEMIBOLD).to_edge(DOWN,buff=.45)
        self.play(AddTextLetterByLetter(cap),run_time=1.8)

    def draw_B07(self):
        self.add(heading("Why a shorter route cannot arrive later"))
        d=VGroup(*[dot(x,BLUE) for x in ["A","B","C"]]).arrange(RIGHT,buff=.8).move_to([-2.8,1.0,0])
        dp=VGroup(*[dot(x,ORANGE) for x in ["D","E","F"]]).arrange(RIGHT,buff=.8).move_to([-2.8,-1.1,0])
        self.add(tx("layer  d",25).next_to(d,LEFT),tx("layer  d + 1",25).next_to(dp,LEFT)); self.play(Create(d),run_time=.8)
        arrows=VGroup(*[Arrow(d[i].get_bottom(),dp[i].get_top(),color=GRAY,buff=.08) for i in range(3)]); self.play(Create(arrows),Create(dp),run_time=1.4)
        queue=VGroup(tx("QUEUE",25,INK,SEMIBOLD),cells(["A","B","C","D"],0)).arrange(DOWN,buff=.2).move_to([4.35,.45,0]); self.play(FadeIn(queue),run_time=.7)
        self.play(Transform(queue[1],cells(["D","E","F"],0).move_to(queue[1])),run_time=1.5)
        self.play(Write(tx("d drains before d + 1",27,INK,SEMIBOLD).move_to([4.35,-1.2,0])),run_time=.8)

    def draw_B08(self):
        self.add(heading("The opening graph, resolved")); g,n=graph(); self.add(g)
        bfs=VGroup(tx("BFS",27,INK,SEMIBOLD),tx("S  →  A  →  B  →  T",26),tx("target after 4 removals",23,GRAY)).arrange(DOWN,aligned_edge=LEFT,buff=.2).move_to([4.25,1.0,0])
        dfs=VGroup(tx("DFS",27,INK,SEMIBOLD),tx("S  →  A  →  C  →  E  →  F  →  B  →  T",21),tx("target after 7 removals",23,GRAY)).arrange(DOWN,aligned_edge=LEFT,buff=.2).move_to([4.25,-1.2,0])
        self.play(AddTextLetterByLetter(bfs),run_time=1.8); self.play(AddTextLetterByLetter(dfs),run_time=2.2)
        self.play(n["T"][0].animate.set_color(TEAL).set_fill(TEAL,opacity=.28),run_time=.8)

for _bid in [b["beat_id"] for b in SHEET["beats"] if b["render"] == "manim"]:
    globals()[_bid] = type(_bid,(ReelBeat,),{"BID":_bid})
