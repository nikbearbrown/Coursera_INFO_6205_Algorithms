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

def tx(s, size=30, color=INK, weight=NORMAL):
    return Text(s, font="EB Garamond", font_size=size, color=color, weight=weight)

def header(title):
    t = tx(title, 39, INK, SEMIBOLD).to_edge(UP, buff=.25)
    return VGroup(t, Line(LEFT*6.4, RIGHT*6.4, color=GRAY, stroke_width=1).next_to(t, DOWN, buff=.12))

def person(name, x, y, color):
    c = Circle(.38, color=color, stroke_width=4, fill_color=color, fill_opacity=.12).move_to([x,y,0])
    return VGroup(c, tx(name, 28, INK, SEMIBOLD).move_to(c))

def match_stage(names=("A","B","X","Y")):
    a,b,x,y = person(names[0],-4.7,1.15,BLUE), person(names[1],-4.7,-1.15,BLUE), person(names[2],.4,1.15,ORANGE), person(names[3],.4,-1.15,ORANGE)
    prefs = VGroup(
        tx(f"{names[0]}:  {names[3]} > {names[2]}",26), tx(f"{names[1]}:  {names[3]} > {names[2]}",26),
        tx(f"{names[2]}:  {names[0]} > {names[1]}",26), tx(f"{names[3]}:  {names[0]} > {names[1]}",26)
    ).arrange(DOWN, aligned_edge=LEFT, buff=.22).move_to([4.1,.0,0])
    return VGroup(a,b,x,y), prefs

def edge(p, q, color=GRAY, dashed=False):
    cls = DashedLine if dashed else Line
    return cls(p.get_center(), q.get_center(), color=color, stroke_width=5).set_z_index(-1)

def proposal_stage(second=False):
    students = VGroup(*[person(f"S{i}",-4.8,1.65-(i-1)*1.65,BLUE) for i in range(1,4)])
    colleges = VGroup(*[person(f"C{i}",.1,1.65-(i-1)*1.65,ORANGE) for i in range(1,4)])
    if second:
        rows = ["S1:   C1   C2   C3", "S2:   C1   C3   C2", "S3:   C3   C2   C1"]
        receiver = "C1:   S2   S1   S3"
    else:
        rows = ["S1:   C1   C2   C3", "S2:   C2   C1   C3", "S3:   C1   C3   C2"]
        receiver = "C1:   S1   S3   S2"
    upper = VGroup(*[tx(r,25) for r in rows]).arrange(DOWN,aligned_edge=LEFT,buff=.2)
    lower = tx(receiver,25).next_to(upper,DOWN,buff=.65).align_to(upper,LEFT)
    table = VGroup(upper,lower).move_to([4.1,.15,0])
    return students, colleges, table

class ReelBeat(Scene):
    BID = ""
    def construct(self):
        self.camera.background_color = BG
        getattr(self, "draw_" + self.BID)()
        remaining = DURS[self.BID] - self.renderer.time
        if remaining > 0: self.wait(remaining)

    def draw_B01(self):
        self.add(header("A complete matching can still break"))
        people,prefs=match_stage(); self.play(FadeIn(people),Write(prefs),run_time=2)
        fixed=VGroup(edge(people[0],people[2]),edge(people[1],people[3])); self.play(Create(fixed),run_time=2)
        block=edge(people[0],people[3],VERM,True); label=tx("BLOCKING  PAIR",25).next_to(block,DOWN,buff=.1)
        self.play(Create(block),Write(label),run_time=2)
        badge=tx("COMPLETE  ≠  STABLE",34,INK,SEMIBOLD).to_edge(DOWN,buff=.35); self.play(Write(badge),run_time=1.5)

    def draw_B02(self):
        self.add(header("Same matching, one changed preference"))
        people,prefs=match_stage(); fixed=VGroup(edge(people[0],people[2]),edge(people[1],people[3])); self.add(people,prefs,fixed)
        old=prefs[3]; new=tx("Y:  B > A",26).move_to(old,aligned_edge=LEFT)
        self.play(Transform(old,new),run_time=2)
        candidate=edge(people[0],people[3],VERM,True); self.play(Create(candidate),run_time=1.5)
        no=tx("A prefers Y  ·  Y prefers B",29).to_edge(DOWN,buff=.55); self.play(Write(no),candidate.animate.set_color(GRAY).set_stroke(opacity=.35),run_time=2)
        stable=tx("NO  MUTUAL  DEFECTION",31,INK,SEMIBOLD).next_to(no,UP,buff=.25); self.play(Write(stable),run_time=1.3)

    def draw_B03(self):
        self.add(header("Predict before C1 decides")); ss,cs,tab=proposal_stage(); self.add(ss,cs,tab)
        e1=edge(ss[0],cs[0],TEAL); e3=edge(ss[2],cs[0],YELLOW); self.play(Create(e1),Create(e3),run_time=2)
        held=tx("C1  HOLDS  S1",27).next_to(cs[0],DOWN,buff=.42); self.play(Write(held),run_time=1)
        q=tx("PERMANENT  —  OR  TENTATIVE?",37,INK,SEMIBOLD).to_edge(DOWN,buff=.38)
        wash=SurroundingRectangle(q,buff=.18,color=YELLOW,fill_color=YELLOW,fill_opacity=.35,stroke_opacity=0).set_z_index(-1)
        self.play(FadeIn(wash),Write(q),run_time=1.8)

    def draw_B04(self):
        self.add(header("Trace 1 · rejection moves one pointer")); ss,cs,tab=proposal_stage(); self.add(ss,cs,tab)
        keep=edge(ss[0],cs[0],TEAL); reject=edge(ss[2],cs[0],VERM); self.play(Create(keep),Create(reject),run_time=2)
        badges=VGroup(tx("KEEP",22).next_to(keep.get_center(),UP),tx("REJECT",22).next_to(reject.get_center(),DOWN)).set_z_index(3); self.play(Write(badges),run_time=1.4)
        newedge=edge(ss[2],cs[2],TEAL); self.play(FadeOut(reject),Transform(badges[1],tx("NEXT  →  C3",22).move_to([3,-2.55,0])),Create(newedge),run_time=2.3)
        pointer=tx("S3 pointer:  1  →  2",28).to_edge(DOWN,buff=.32); self.play(Write(pointer),run_time=1.2)

    def draw_B05(self):
        self.add(header("Trace 2 · receivers can trade up")); ss,cs,tab=proposal_stage(True); self.add(ss,cs,tab)
        first=edge(ss[0],cs[0],TEAL); self.play(Create(first),Write(tx("HOLD  S1",22).next_to(cs[0],RIGHT,buff=.42)),run_time=2)
        challenger=edge(ss[1],cs[0],YELLOW); self.play(Create(challenger),run_time=1.3)
        trade=edge(ss[1],cs[0],TEAL); nextedge=edge(ss[0],cs[1],TEAL)
        self.play(Transform(first,trade),FadeOut(challenger),Create(nextedge),run_time=2.4)
        states=VGroup(tx("C1:  TRADE  UP  S1 → S2",25),tx("S1:  NEXT  →  C2",25)).arrange(DOWN,aligned_edge=LEFT,buff=.18).to_edge(DOWN,buff=.25)
        self.play(Write(states),run_time=1.5)

    def draw_B06(self):
        self.add(header("The characteristic: monotone progress")); ss,cs,tab=proposal_stage(True); self.add(ss,cs,tab)
        self.play(FadeOut(tab),run_time=.6)
        labels=VGroup(tx("PROPOSERS:   DOWN  THE  LIST",27),tx("RECEIVERS:   UP  IN  RANK",27)).arrange(DOWN,aligned_edge=LEFT,buff=1.35).move_to([3.85,.15,0])
        arrows=VGroup(Arrow(labels[0].get_left()+DOWN*.45,labels[0].get_right()+DOWN*.45,color=GRAY,buff=0),Arrow(labels[1].get_right()+DOWN*.45,labels[1].get_left()+DOWN*.45,color=GRAY,buff=0))
        self.play(Write(labels[0]),GrowArrow(arrows[0]),run_time=2); self.play(Write(labels[1]),GrowArrow(arrows[1]),run_time=2)
        fin=tx("FINITE  LISTS  ·  NO  BACKWARD  MOVE",31,INK,SEMIBOLD).to_edge(DOWN,buff=.3); self.play(Write(fin),run_time=1.8)

    def draw_B07(self):
        self.add(header("Why a blocking pair cannot survive")); people,prefs=match_stage(); self.add(people)
        hyp=edge(people[0],people[3],VERM,True); self.play(Create(hyp),Write(tx("suppose A + Y block",26).next_to(hyp,DOWN,buff=.12)),run_time=2)
        proposal=Arrow(people[0].get_center(),people[3].get_center(),color=BLUE,buff=.45); self.play(GrowArrow(proposal),run_time=1.5)
        ladder=VGroup(tx("Y  held  A",25),tx("Y  rejected  A",25),tx("Y  holds  someone  ranked  higher",25)).arrange(DOWN,aligned_edge=LEFT,buff=.38).move_to([4.05,.05,0])
        self.play(Write(ladder[0]),run_time=1); self.play(Write(ladder[1]),run_time=1); self.play(Write(ladder[2]),run_time=1.4)
        self.play(hyp.animate.set_color(GRAY).set_stroke(opacity=.2),FadeOut(proposal),run_time=1)
        contradiction=tx("Y  CANNOT  PREFER  A  AT  THE  END",30,INK,SEMIBOLD).to_edge(DOWN,buff=.32); self.play(Write(contradiction),run_time=1.5)

    def draw_B08(self):
        self.add(header("Payoff · the hidden defection disappears")); people,prefs=match_stage(); self.add(people,prefs)
        fixed=VGroup(edge(people[0],people[2],GRAY),edge(people[1],people[3],TEAL)); candidate=edge(people[0],people[3],VERM,True); self.play(Create(fixed),Create(candidate),run_time=2)
        rank=tx("Y holds B  ·  Y ranks B above A",28).move_to([4.0,-2.0,0]); self.play(Write(rank),run_time=1.8)
        self.play(candidate.animate.set_color(GRAY).set_stroke(opacity=.15),run_time=1.2)
        answer=tx("A  WANTS  Y  ·  Y  DOES  NOT  WANT  A",31,INK,SEMIBOLD).to_edge(DOWN,buff=.25); self.play(Write(answer),run_time=1.7)

class B01(ReelBeat): BID="B01"
class B02(ReelBeat): BID="B02"
class B03(ReelBeat): BID="B03"
class B04(ReelBeat): BID="B04"
class B05(ReelBeat): BID="B05"
class B06(ReelBeat): BID="B06"
class B07(ReelBeat): BID="B07"
class B08(ReelBeat): BID="B08"
