from manim import *
import json
from pathlib import Path

ROOT=Path(__file__).parent
SHEET=json.loads((ROOT/"beat_sheet.json").read_text())
DURS={b["beat_id"]:float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG,INK,GRAY="#F0EAD6","#000000","#4D4D4D"
TEAL,VERM,YELLOW,BLUE="#009E73","#D55E00","#F0E442","#0072B2"
config.background_color=BG

def tx(s,size=28,color=INK,weight=NORMAL,font="EB Garamond"):
    return Text(s,font=font,font_size=size,color=color,weight=weight)
def header(s):
    t=tx(s,38,INK,SEMIBOLD).to_edge(UP,buff=.24)
    return VGroup(t,Line(LEFT*6.4,RIGHT*6.4,color=GRAY,stroke_width=1).next_to(t,DOWN,buff=.1))
def cells(values, center=(-2.5,.35,0), scale=1.0):
    boxes=VGroup()
    for v in values:
        sq=Square(.72*scale,color=GRAY,stroke_width=2,fill_color=BG,fill_opacity=1)
        n=tx(str(v),27*scale,font="JetBrains Mono").move_to(sq)
        boxes.add(VGroup(sq,n))
    boxes.arrange(RIGHT,buff=.08*scale).move_to(center)
    return boxes
def side(lines,title="TRACE"):
    return VGroup(tx(title,24,INK,SEMIBOLD),*[tx(x,23) for x in lines]).arrange(DOWN,aligned_edge=LEFT,buff=.28).move_to([4.15,.05,0])
def pivot_mark(box,label="PIVOT"):
    wash=box[0].copy().set_fill(YELLOW,.45).set_stroke(INK,3)
    lab=tx(label,19,INK,SEMIBOLD).next_to(box,DOWN,buff=.16)
    return VGroup(wash,lab)

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        getattr(self,"draw_"+self.BID)()
        left=DURS[self.BID]-self.renderer.time
        if left>0:self.wait(left)
    def draw_B01(self):
        self.add(header("Fixed last pivot · sorted input aims at an extreme"))
        a=cells(range(1,10));self.add(a)
        work=side(["active  9","pivot  9","keep  8","work  9"],"FIND RANK 4");self.add(work)
        mark=pivot_mark(a[-1]);self.play(FadeIn(mark),run_time=.8)
        total=9
        for active,p in [(8,8),(7,7),(6,6),(5,5),(4,4)]:
            total+=active
            self.play(a[active:].animate.set_opacity(.16),run_time=.65)
            new=pivot_mark(a[active-1]);self.play(Transform(mark,new),run_time=.45)
            nw=side([f"active  {active}",f"pivot  {p}",f"keep  {active-1 if p!=4 else 0}",f"work  {total}"],"FIND RANK 4")
            self.play(Transform(work,nw),run_time=.45)
        self.play(Write(tx("FOUND  4 · AFTER 39 INSPECTIONS",30,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1)
    def draw_B02(self):
        self.add(header("Same input · pivots 5, 2, 4"));a=cells(range(1,10));self.add(a)
        work=side(["pivot rank  5","k = 4  →  LEFT","active  4","work  9"],"RANDOM TAPE");self.add(work)
        mark=pivot_mark(a[4]);self.play(FadeIn(mark),run_time=.8);self.play(a[4:].animate.set_opacity(.16),run_time=1)
        self.play(Transform(mark,pivot_mark(a[1])),run_time=.7)
        nw=side(["pivot rank  2","k = 4  →  RIGHT","local k = 2","work  13"],"RANDOM TAPE");self.play(Transform(work,nw),a[:2].animate.set_opacity(.16),run_time=1.2)
        self.play(Transform(mark,pivot_mark(a[3])),run_time=.7)
        nw2=side(["pivot rank  2","local k = 2","pivot  4  = target","work  15"],"RANDOM TAPE");self.play(Transform(work,nw2),run_time=1)
        self.play(Write(tx("FOUND  4",32,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=.8)
    def draw_B03(self):
        self.add(header("Predict before the partition disappears"));a=cells(range(1,10));self.add(a)
        mark=pivot_mark(a[5]);self.add(mark)
        q=VGroup(tx("pivot rank  6 · target rank  4",27),tx("LEFT or RIGHT?   Does k change?",31,INK,SEMIBOLD)).arrange(DOWN,buff=.35).move_to([3.7,-.05,0])
        wash=SurroundingRectangle(q,buff=.25,fill_color=YELLOW,fill_opacity=.32,stroke_opacity=0).set_z_index(-1)
        self.play(FadeIn(wash),Write(q),run_time=2)
    def draw_B04(self):
        self.add(header("A third trace · pivots 6, 3, 4"));a=cells(range(1,10));self.add(a)
        mark=pivot_mark(a[5]);self.add(mark);work=side(["6 > 4  →  keep LEFT","k stays 4","work 9"],"TRACE");self.add(work)
        self.play(a[5:].animate.set_opacity(.16),run_time=1)
        self.play(Transform(mark,pivot_mark(a[2])),run_time=.7)
        nw=side(["3 < 4  →  keep RIGHT","local k:  4 − 3 = 1","work 14"],"TRACE");self.play(Transform(work,nw),a[:3].animate.set_opacity(.16),run_time=1.2)
        self.play(Transform(mark,pivot_mark(a[3])),run_time=.7)
        self.play(Write(tx("LOCAL RANK 1  →  FOUND 4",30,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1)
    def draw_B05(self):
        self.add(header("The structural license"));a=cells(range(1,10),(-3.35,.5,0),.82);self.add(a)
        cards=VGroup(
            tx("1   PARTITION  →  pivot rank is final",24),
            tx("2   TARGET k  →  exactly one side survives",24),
            tx("3   RANDOM RANK  →  independent of input order",24)
        ).arrange(DOWN,aligned_edge=LEFT,buff=.62).move_to([3.25,.15,0])
        for c in cards:self.play(Write(c),run_time=1.2)
        self.play(a[4][0].animate.set_fill(YELLOW,.4),a[:4].animate.set_color(TEAL),a[5:].animate.set_color(VERM),run_time=1.3)
    def draw_B06(self):
        self.add(header("Complexity last · count the traced work"))
        left=VGroup(tx("FIXED LAST PIVOT",24,INK,SEMIBOLD),tx("9 + 8 + 7 + 6 + 5 + 4",28,font="JetBrains Mono"),tx("39 inspected cells",29,INK,SEMIBOLD)).arrange(DOWN,buff=.42).move_to([-3.5,.35,0])
        right=VGroup(tx("PIVOTS 5, 2, 4",24,INK,SEMIBOLD),tx("9 + 4 + 2",28,font="JetBrains Mono"),tx("15 inspected cells",29,INK,SEMIBOLD)).arrange(DOWN,buff=.42).move_to([3.5,.35,0])
        self.play(Write(left),run_time=2);self.play(Write(right),run_time=2)
        bad=Rectangle(width=4.8,height=.42,fill_color=VERM,fill_opacity=.45,stroke_opacity=0).move_to([-3.5,-1.65,0])
        good=Rectangle(width=1.85,height=.42,fill_color=TEAL,fill_opacity=.55,stroke_opacity=0).move_to([2.03,-1.65,0]).align_to(bad,LEFT)
        self.play(GrowFromEdge(bad,LEFT),GrowFromEdge(good,LEFT),run_time=1.5)
        self.play(Write(tx("EXPECTED LINEAR · WORST CASE STILL QUADRATIC",28,INK,SEMIBOLD).to_edge(DOWN,buff=.3)),run_time=1)
    def draw_B07(self):
        self.add(header("Payoff · the input loses control of pivot rank"));a=cells(range(1,10),(-2.8,.65,0));self.add(a)
        fixed=VGroup(tx("FIXED RULE",24,INK,SEMIBOLD),tx("input aims at rank 9",25),tx("extreme every time",25)).arrange(DOWN,buff=.3).move_to([3.8,1.15,0])
        coin=VGroup(tx("FRESH RANDOM RULE",24,INK,SEMIBOLD),tx("input cannot aim",25),tx("good cut not guaranteed",25)).arrange(DOWN,buff=.3).move_to([3.8,-1.2,0])
        self.play(Write(fixed),run_time=1.5);self.play(a[-1][0].animate.set_fill(VERM,.35),run_time=1)
        self.play(Write(coin),run_time=1.5);self.play(a[-1][0].animate.set_fill(BG,1),a[4][0].animate.set_fill(YELLOW,.45),run_time=1)

class B01(ReelBeat):BID="B01"
class B02(ReelBeat):BID="B02"
class B03(ReelBeat):BID="B03"
class B04(ReelBeat):BID="B04"
class B05(ReelBeat):BID="B05"
class B06(ReelBeat):BID="B06"
class B07(ReelBeat):BID="B07"
