from manim import *
import json
from pathlib import Path

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW = "#009E73", "#D55E00", "#F0E442"
BLUE, ORANGE = "#0072B2", "#E69F00"
config.background_color = BG

def tx(s, size=30, color=INK, weight=NORMAL):
    return Text(s, font="EB Garamond", font_size=size, color=color, weight=weight)

def heading(s):
    t = tx(s, 40, INK, SEMIBOLD).to_edge(UP, buff=.28)
    return VGroup(t, Line(LEFT*6.5, RIGHT*6.5, color=GRAY, stroke_width=1).next_to(t, DOWN, buff=.12))

def axis():
    line = NumberLine(x_range=[0, 10, 1], length=10.4, include_numbers=True,
                      font_size=22, color=GRAY).shift(DOWN*2.55)
    return line

def bar(a, b, lane, label, color=GRAY, state=None):
    x0, x1 = -5.2 + a*1.04, -5.2 + b*1.04
    y = 1.65 - lane*.72
    rect = RoundedRectangle(width=x1-x0, height=.48, corner_radius=.08,
                            color=color, fill_color=color, fill_opacity=.16,
                            stroke_width=3).move_to([(x0+x1)/2, y, 0])
    lab = tx(label, 23, INK, SEMIBOLD).move_to(rect)
    g = VGroup(rect, lab)
    if state:
        badge = tx(state, 19, INK, SEMIBOLD).next_to(rect, RIGHT, buff=.12)
        g.add(badge)
    return g

def day_one():
    return VGroup(
        bar(1,9,0,"L  [1,9]"), bar(1,3,1,"A  [1,3]"),
        bar(3,5,2,"B  [3,5]"), bar(5,7,3,"C  [5,7]"), bar(7,9,4,"D  [7,9]")
    )

class ReelBeat(Scene):
    BID = ""
    def construct(self):
        self.camera.background_color = BG
        getattr(self, "draw_" + self.BID, self.draw_default)()
        if self.renderer.time < DURS[self.BID]:
            self.wait(DURS[self.BID] - self.renderer.time)
    def draw_default(self):
        self.play(Write(tx(self.BID, 60)))

    def draw_B01(self):
        self.add(heading("The earliest start trap"), axis())
        bars = day_one(); self.play(LaggedStart(*[Create(x) for x in bars], lag_ratio=.16), run_time=2.5)
        chosen = bars[0][0].copy().set_color(VERM).set_fill(VERM, opacity=.22)
        self.play(Transform(bars[0][0], chosen), run_time=1)
        for x in bars[1:]: self.play(x.animate.set_opacity(.22), run_time=.3)
        self.play(Write(tx("KEEP  1  ·  REJECT  4", 31, INK, SEMIBOLD).move_to([4.55,2.45,0])), run_time=1)

    def draw_B02(self):
        self.add(heading("The shortest duration trap"), axis())
        left = bar(0,4,0,"P  [0,4]"); tiny = bar(3.5,4.5,1,"T  [3.5,4.5]"); right = bar(4,8,2,"Q  [4,8]")
        self.play(Create(left), Create(tiny), Create(right), run_time=2)
        self.play(tiny[0].animate.set_color(VERM).set_fill(VERM, opacity=.22), left.animate.set_opacity(.25), right.animate.set_opacity(.25), run_time=1.5)
        bad = tx("SHORTEST:  1", 29, INK, SEMIBOLD).move_to([4.7,2.2,0])
        self.play(Write(bad), run_time=1)
        pair = VGroup(left.copy().set_opacity(1), right.copy().set_opacity(1))
        pair[0][0].set_color(TEAL).set_fill(TEAL, opacity=.2); pair[1][0].set_color(TEAL).set_fill(TEAL, opacity=.2)
        self.play(ReplacementTransform(VGroup(left,right), pair), tiny.animate.set_opacity(.2), run_time=1.5)
        self.play(Transform(bad, tx("COMPATIBLE  PAIR:  2",27,INK,SEMIBOLD).move_to([4.35,2.2,0])), run_time=1)

    def draw_B03(self):
        self.add(heading("Predict the first choice"), axis())
        bars=day_one(); self.play(LaggedStart(*[Create(x) for x in bars],lag_ratio=.12),run_time=2)
        q=VGroup(tx("Which interval goes first?",38,INK,SEMIBOLD),tx("Commit before the scan.",28,GRAY)).arrange(DOWN,buff=.2).to_edge(RIGHT).shift(UP*2.05)
        wash=SurroundingRectangle(q,stroke_width=0,fill_color=YELLOW,fill_opacity=.28,buff=.15).set_z_index(-1)
        self.play(AddTextLetterByLetter(q),FadeIn(wash),run_time=2)

    def draw_B04(self):
        self.add(heading("Finish first, then scan"), axis())
        bars=day_one(); self.add(bars)
        boundary=DashedLine([ -4.16,-2.25,0],[-4.16,2.2,0],color=GRAY)
        self.play(Create(boundary),run_time=.8)
        order=[1,0,2,3,4]; states=["KEEP","REJECT","KEEP","KEEP","KEEP"]
        for idx,state in zip(order,states):
            color=TEAL if state=="KEEP" else VERM
            self.play(bars[idx][0].animate.set_color(color).set_fill(color,opacity=.2),Write(tx(state,18,INK,SEMIBOLD).next_to(bars[idx],RIGHT,buff=.08)),run_time=.7)
            if state=="KEEP":
                end=[3,5,7,9][[1,2,3,4].index(idx)]
                self.play(boundary.animate.shift(RIGHT*((end)-(-4.16+5.2)/1.04)*0),run_time=.01) if False else None
        self.play(Write(tx("4 KEPT",38,INK,SEMIBOLD).to_edge(RIGHT).shift(UP*2.3)),run_time=1)

    def draw_B05(self):
        self.add(heading("Lengths change; finish order still decides"), axis())
        old=VGroup(bar(1,3,0,"A"),bar(3,5,1,"B"),bar(5,7,2,"C"),bar(7,9,3,"D"))
        self.add(old)
        new=VGroup(bar(.2,3,0,"A"),bar(2.2,5,1,"B"),bar(4.4,7,2,"C"),bar(6.3,9,3,"D"))
        self.play(Transform(old,new),run_time=3)
        for x in old: self.play(x[0].animate.set_color(TEAL).set_fill(TEAL,opacity=.2),run_time=.35)
        self.play(Write(tx("same finish order  ·  same four choices",30,INK,SEMIBOLD).to_edge(DOWN,buff=.35)),run_time=1)

    def draw_B06(self):
        self.add(heading("The characteristic that licenses the rule"))
        subject=VGroup(
            RoundedRectangle(width=3.2,height=.58,corner_radius=.08,color=TEAL,fill_color=TEAL,fill_opacity=.18),
            tx("earliest  finisher",25,INK,SEMIBOLD),
            RoundedRectangle(width=4.5,height=.58,corner_radius=.08,color=GRAY,fill_color=GRAY,fill_opacity=.12),
            tx("other  first  choice",25,INK,SEMIBOLD)
        )
        subject[0].move_to([-4.6,1.0,0]); subject[1].move_to(subject[0]); subject[2].move_to([-4.0,-.55,0]); subject[3].move_to(subject[2])
        room=VGroup(tx("FUTURE  ROOM",27,INK,SEMIBOLD),tx("after  finish  3",25),Rectangle(width=4.0,height=.5,color=TEAL,fill_color=TEAL,fill_opacity=.18),tx("after  finish  5",25),Rectangle(width=2.8,height=.5,color=GRAY,fill_color=GRAY,fill_opacity=.12)).arrange(DOWN,aligned_edge=LEFT,buff=.2).move_to([3.65,.25,0])
        self.play(Create(subject),Create(room),run_time=2.5)
        cap=VGroup(tx("GREEDY-CHOICE  PROPERTY",34,INK,SEMIBOLD),tx("earlier  finish  ⇒  no  less  room",28,INK)).arrange(DOWN).to_edge(DOWN,buff=.35)
        self.play(AddTextLetterByLetter(cap),run_time=2)

    def draw_B07(self):
        self.add(heading("Exchange one first choice; preserve the suffix"),axis())
        opt=VGroup(bar(1,4,0,"X",ORANGE),bar(4,6,1,"Y",TEAL),bar(6,8,2,"Z",TEAL))
        self.add(opt)
        earliest=bar(1,3,0,"A",TEAL)
        self.play(ReplacementTransform(opt[0],earliest),run_time=2.5)
        self.play(Indicate(VGroup(opt[1],opt[2]),color=TEAL),run_time=1.5)
        count=VGroup(tx("before: 3",29),tx("after:  3",29),tx("later intervals still fit",31,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT).to_edge(RIGHT).shift(UP*2)
        self.play(Write(count),run_time=2)

    def draw_B08(self):
        self.add(heading("The opening puzzle, resolved"),axis())
        bars=day_one().scale(.72).to_edge(LEFT,buff=.3).shift(DOWN*.15); self.add(bars)
        for i,x in enumerate(bars): x[0].set_color(TEAL if i else VERM).set_fill(TEAL if i else VERM,opacity=.18)
        result=VGroup(tx("EARLIEST  START  →  1",28,INK,SEMIBOLD),tx("EARLIEST  FINISH  →  4",30,INK,SEMIBOLD),tx("sort  once  +  scan  once",27),tx("O(n  log  n)",36,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.2).move_to([4.25,.85,0])
        self.play(LaggedStart(*[Write(x) for x in result],lag_ratio=.3),run_time=4)

for _bid in [b["beat_id"] for b in SHEET["beats"] if b["render"] == "manim"]:
    globals()[_bid] = type(_bid, (ReelBeat,), {"BID": _bid})
