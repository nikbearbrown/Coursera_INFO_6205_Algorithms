from manim import *
import json
from pathlib import Path

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or 7) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW, BLUE, ORANGE, SKY = "#009E73", "#D55E00", "#F0E442", "#0072B2", "#E69F00", "#56B4E9"
config.background_color = BG

def tx(s, size=28, color=INK, weight=NORMAL, font="EB Garamond"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)

def heading(s):
    title = tx(s, 39, INK, SEMIBOLD).to_edge(UP, buff=.25)
    return VGroup(title, Line(LEFT*6.5, RIGHT*6.5, color=GRAY, stroke_width=1).next_to(title, DOWN, buff=.1))

def cell(v, color=BLUE, scale=.72):
    box = Square(.72, color=GRAY, fill_color=color, fill_opacity=.16, stroke_width=2)
    num = tx(str(v), 28, INK, SEMIBOLD, "JetBrains Mono").move_to(box)
    return VGroup(box, num).scale(scale)

def array(vals, scale=.72):
    return VGroup(*[cell(v, [BLUE, ORANGE, SKY, TEAL][i % 4], scale) for i,v in enumerate(vals)]).arrange(RIGHT, buff=.06)

def tree8():
    levels = [
        [array([8,3,6,2,7,1,5,4], .55)],
        [array([8,3,6,2], .55), array([7,1,5,4], .55)],
        [array([8,3], .55), array([6,2], .55), array([7,1], .55), array([5,4], .55)],
        [array([8], .55),array([3], .55),array([6], .55),array([2], .55),array([7], .55),array([1], .55),array([5], .55),array([4], .55)]
    ]
    ys=[2.25,.8,-.65,-2.05]
    for lev,y in zip(levels,ys): VGroup(*lev).arrange(RIGHT,buff=.45).move_to(UP*y)
    edges=VGroup()
    for li in range(3):
        for i,p in enumerate(levels[li]):
            for c in levels[li+1][2*i:2*i+2]: edges.add(Line(p.get_bottom(),c.get_top(),color=GRAY,stroke_width=1.5))
    return levels,edges

def recurrence(size=45):
    return MathTex(r"T(n)=2T(n/2)+n", color=INK, font_size=size)

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        getattr(self,"draw_"+self.BID)()
        if self.renderer.time < DURS[self.BID]: self.wait(DURS[self.BID]-self.renderer.time)
    def head(self,s): self.add(heading(s))

    def draw_B01(self):
        self.head("Instance one: four values")
        top=array([8,3,6,2],.9).shift(UP*1.75); self.play(Create(top),run_time=2)
        singles=VGroup(*[array([v],.9) for v in [8,3,6,2]]).arrange(RIGHT,buff=.75).shift(UP*.15)
        self.play(TransformFromCopy(top,singles),run_time=2)
        pairs=VGroup(array([3,8],.9),array([2,6],.9)).arrange(RIGHT,buff=1.3).shift(DOWN*1.2)
        self.play(TransformFromCopy(singles,pairs),run_time=2.5)
        out=array([2,3,6,8],.9).shift(DOWN*2.35); count=tx("final merge · 3 comparisons",25,INK,SEMIBOLD).next_to(out,RIGHT,buff=.4)
        self.play(TransformFromCopy(pairs,out),Write(count),run_time=3)

    def draw_B02(self):
        self.head("Two levels — four values at each level")
        bars=VGroup()
        for y,label in [(1.25,"two 2-item merges"),(-.55,"one 4-item merge")]:
            bar=Rectangle(width=7,height=.7,color=GRAY,fill_color=TEAL,fill_opacity=.2).move_to(LEFT*1.6+UP*y)
            marks=VGroup(*[Line(UP*.32,DOWN*.32,color=GRAY) for _ in range(5)]).arrange(RIGHT,buff=1.68).move_to(bar)
            lab=tx(label,28).next_to(bar,RIGHT,buff=.45); total=tx("covers  4",30,INK,SEMIBOLD).next_to(lab,DOWN,buff=.12)
            bars.add(VGroup(bar,marks,lab,total))
        self.play(LaggedStart(*[Create(x) for x in bars],lag_ratio=.35),run_time=4)
        result=tx("2  levels × 4  values",42,INK,SEMIBOLD,"JetBrains Mono").to_edge(DOWN,buff=.55)
        self.play(Write(result),run_time=2)

    def draw_B03(self):
        self.head("Instance two: eight values")
        levels,edges=tree8(); self.play(Create(levels[0][0]),run_time=1.5)
        for i in range(1,4):
            self.play(LaggedStart(*[FadeIn(x,shift=DOWN*.2) for x in levels[i]],lag_ratio=.12),Create(VGroup(*[e for e in edges if abs(e.get_end()[1]-levels[i][0].get_top()[1])<.1])),run_time=2)
        labels=VGroup(*[tx(s,20,GRAY,"NORMAL","JetBrains Mono").to_edge(RIGHT,buff=.25).move_to(RIGHT*5.8+UP*y) for s,y in [("n=8",2.25),("n/2",.8),("n/4",-.65),("1",-2.05)]])
        self.play(Write(labels),run_time=2)

    def draw_B04(self):
        self.head("Predict the work across one level")
        groups=VGroup(array([3,8],.85),array([2,6],.85),array([1,7],.85),array([4,5],.85)).arrange(RIGHT,buff=.45).shift(UP*.75)
        self.add(groups)
        q=tx("How many items can this level inspect?",34,INK,SEMIBOLD).shift(DOWN*.55); self.play(Write(q),run_time=2); self.wait(2)
        counter=tx("0",52,INK,SEMIBOLD,"JetBrains Mono").shift(DOWN*1.65); self.add(counter)
        for i,g in enumerate(groups):
            nxt=tx(str(2*(i+1)),52,INK,SEMIBOLD,"JetBrains Mono").move_to(counter)
            self.play(Indicate(g,color=TEAL),Transform(counter,nxt),run_time=.8)
        answer=tx("8 items · at most 7 key comparisons",29,INK,SEMIBOLD).to_edge(DOWN,buff=.35)
        self.play(Write(answer),run_time=2)

    def draw_B05(self):
        self.head("The recurrence the trace earned")
        levels,edges=tree8(); tree=VGroup(*sum(levels,[]),edges).scale(.65).to_edge(LEFT,buff=.25).shift(DOWN*.1); self.add(tree)
        laws=VGroup(tx("SAME-TYPE  HALVES",25,INK,SEMIBOLD),tx("independent · strictly smaller",22),tx("BASE CASE  AT  ONE",25,INK,SEMIBOLD),tx("LINEAR  COMBINE",25,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.18).move_to(RIGHT*3.6+UP*.9)
        self.play(Write(laws),run_time=3)
        eq=recurrence(46).next_to(laws,DOWN,buff=.55); self.play(Write(eq),run_time=2.5)

    def tangent(self):
        self.add(heading("Merge sort · equation"))
        eq=recurrence(48).move_to(LEFT*3.55+UP*.35); self.add(eq)
        rule=Line(UP*2.2,DOWN*2.55,color=GRAY,stroke_width=1).move_to(LEFT*.25); self.add(rule)
        return eq

    def draw_B05A(self):
        eq=self.tangent(); side=VGroup(tx("LEFT",22,GRAY,SEMIBOLD),tx("all work for size n",28),tx("RIGHT",22,GRAY,SEMIBOLD),tx("two half sorts + one merge",28),tx("=  claims complete accounting",25,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.17).move_to(RIGHT*3.2)
        self.play(LaggedStart(*[Write(x) for x in side],lag_ratio=.2),run_time=5); self.play(Indicate(eq,color=YELLOW),run_time=1)

    def draw_B05B(self):
        self.tangent(); gloss=VGroup(tx("symbol   role   plain meaning   domain",21,INK,SEMIBOLD),tx("T   function   running cost   nonnegative",22),tx("n   variable   input size   positive integer",22),tx("2   constant   recursive calls",22),tx("n/2   argument   each child size",22),tx("+n   term   combine work",22)).arrange(DOWN,aligned_edge=LEFT,buff=.18).move_to(RIGHT*3.15)
        self.play(LaggedStart(*[Write(x) for x in gloss],lag_ratio=.17),run_time=5)

    def draw_B05C(self):
        self.tangent(); calc=VGroup(tx("WORKED  n = 8",23,INK,SEMIBOLD),MathTex(r"T(8)=2T(4)+8",color=INK,font_size=39),MathTex(r"2T(4)=4T(2)+8",color=INK,font_size=39),tx("combine total at each depth = 8",27,INK,SEMIBOLD)).arrange(DOWN,buff=.28).move_to(RIGHT*3.1)
        self.play(Write(calc[0]),Write(calc[1]),run_time=2); self.play(ReplacementTransform(calc[1].copy(),calc[2]),run_time=2); self.play(Write(calc[3]),run_time=2)

    def draw_B05D(self):
        self.tangent(); claims=VGroup(tx("THE  COMMITMENT",23,INK,SEMIBOLD),tx("balanced, independent halves",29),tx("linear merge work",29),tx("change either fact → change the recurrence",25,GRAY),tx("return to the tree: count levels",27,INK,SEMIBOLD)).arrange(DOWN,buff=.25).move_to(RIGHT*3.05)
        self.play(LaggedStart(*[Write(x) for x in claims],lag_ratio=.2),run_time=5)

    def draw_B06(self):
        self.head("Where the n log n work hides")
        levels,edges=tree8(); tree=VGroup(*sum(levels,[]),edges).scale(.68).to_edge(LEFT,buff=.25).shift(DOWN*.15); self.add(tree)
        math=VGroup(tx("work per level",25,GRAY),MathTex(r"n",color=INK,font_size=48),tx("number of levels",25,GRAY),MathTex(r"\log_2 n",color=INK,font_size=48),MathTex(r"n\log_2 n",color=INK,font_size=58),tx("8 × 3 = 24 merge-work units",27,INK,SEMIBOLD,"JetBrains Mono")).arrange(DOWN,buff=.15).move_to(RIGHT*3.65)
        self.play(Write(math[0]),Write(math[1]),run_time=1.5); self.play(Write(math[2]),Write(math[3]),run_time=1.5); self.play(Write(math[4]),run_time=2); self.play(Write(math[5]),run_time=1.5)

for _bid in ["B01","B02","B03","B04","B05","B05A","B05B","B05C","B05D","B06"]:
    globals()[_bid] = type(_bid,(ReelBeat,),{"BID":_bid})
