from manim import *
import json
from pathlib import Path
from itertools import permutations

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW = "#009E73", "#D55E00", "#F0E442"
BLUE, ORANGE, SKY = "#0072B2", "#E69F00", "#56B4E9"
config.background_color = BG

def tx(s, size=30, color=INK, weight=NORMAL, font="EB Garamond"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)

def heading(s):
    title = tx(s, 40, INK, SEMIBOLD).to_edge(UP, buff=.28)
    rule = Line(LEFT*6.5, RIGHT*6.5, color=GRAY, stroke_width=1).next_to(title, DOWN, buff=.12)
    return VGroup(title, rule)

def card(rank, pos, state="WAIT", color=GRAY):
    box = RoundedRectangle(width=1.12, height=1.38, corner_radius=.1, color=color,
                           fill_color=color, fill_opacity=.12, stroke_width=3)
    num = tx(str(rank), 42, INK, SEMIBOLD, "JetBrains Mono").move_to(box)
    idx = tx(str(pos), 18, GRAY, NORMAL, "JetBrains Mono").next_to(box, DOWN, buff=.08)
    badge = tx(state, 16, INK, SEMIBOLD).next_to(box, UP, buff=.08)
    return VGroup(box, num, idx, badge)

def row(ranks, states=None, colors=None, y=.3):
    states = states or ["WAIT"] * len(ranks)
    colors = colors or [GRAY] * len(ranks)
    g = VGroup(*[card(r, i+1, states[i], colors[i]) for i, r in enumerate(ranks)])
    g.arrange(RIGHT, buff=.22).move_to([-.7, y, 0])
    return g

def threshold_wins(n, reject):
    wins = 0
    for order in permutations(range(1, n+1)):
        if reject == 0:
            chosen = order[0]
        else:
            benchmark = max(order[:reject])
            chosen = next((x for x in order[reject:] if x > benchmark), None)
        wins += chosen == n
    return wins

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
        self.add(heading("The perfect-information trap"))
        ranks = [2, 4, 1, 3]
        cards = row(ranks, ["ARRIVES"]*4)
        for i, c in enumerate(cards):
            self.play(FadeIn(c, shift=UP*.2), run_time=.55)
            self.play(Transform(c[3], tx("REJECT",16,INK,SEMIBOLD).move_to(c[3])),
                      c[0].animate.set_color(VERM).set_fill(VERM, opacity=.18), run_time=.35)
        best = SurroundingRectangle(cards[1], color=TEAL, buff=.07, stroke_width=5)
        panel = VGroup(tx("INFORMATION",22,GRAY,SEMIBOLD), tx("best = 4",34,INK,SEMIBOLD),
                       tx("ABILITY  TO  ACT",22,GRAY,SEMIBOLD), tx("0 candidates left",31,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.16).to_edge(RIGHT,buff=.5).shift(DOWN*.25)
        self.play(Create(best), LaggedStart(*[AddTextLetterByLetter(x) for x in panel],lag_ratio=.18), run_time=2.4)

    def draw_B02(self):
        self.add(heading("Four candidates · every possible order"))
        dots = VGroup(*[Square(.34, color=GRAY, fill_color=GRAY, fill_opacity=.12) for _ in range(24)]).arrange_in_grid(4,6,buff=.16).to_edge(LEFT,buff=.85)
        self.play(LaggedStart(*[FadeIn(x,scale=.7) for x in dots],lag_ratio=.025),run_time=1.8)
        counts = [threshold_wins(4,r) for r in range(4)]
        bars = VGroup()
        for r,c in enumerate(counts):
            rect=Rectangle(width=.72,height=c*.22,color=TEAL if r==1 else GRAY,fill_color=TEAL if r==1 else GRAY,fill_opacity=.22).align_to(ORIGIN,DOWN)
            lab=tx(f"r={r}",21,INK,NORMAL,"JetBrains Mono").next_to(rect,DOWN,buff=.1)
            val=tx(f"{c}/24",23,INK,SEMIBOLD,"JetBrains Mono").next_to(rect,UP,buff=.08)
            bars.add(VGroup(rect,lab,val))
        bars.arrange(RIGHT,buff=.35,aligned_edge=DOWN).move_to([3.4,-.25,0])
        self.play(LaggedStart(*[GrowFromEdge(x[0],DOWN) for x in bars],lag_ratio=.15),run_time=2)
        self.play(LaggedStart(*[Write(VGroup(x[1],x[2])) for x in bars],lag_ratio=.12),run_time=1.4)
        self.play(*[dots[i].animate.set_color(TEAL).set_fill(TEAL,opacity=.35) for i in range(11)],run_time=1.2)
        cap=tx("reject  1  →  11  wins",30,INK,SEMIBOLD).to_edge(DOWN,buff=.35)
        wash=SurroundingRectangle(cap,stroke_width=0,fill_color=YELLOW,fill_opacity=.25,buff=.12).set_z_index(-1)
        self.play(AddTextLetterByLetter(cap),FadeIn(wash),run_time=1.3)

    def draw_B03(self):
        self.add(heading("Predict before the leap"))
        ranks=[4,7,2,6,8,1,5,3]
        states=["OBSERVE","OBSERVE","OBSERVE","NOW","HIDDEN","HIDDEN","HIDDEN","HIDDEN"]
        colors=[BLUE,BLUE,BLUE,ORANGE,GRAY,GRAY,GRAY,GRAY]
        cards=row(ranks,states,colors)
        self.play(LaggedStart(*[FadeIn(c,shift=UP*.2) for c in cards],lag_ratio=.09),run_time=2.2)
        bench=VGroup(tx("BENCHMARK",22,GRAY,SEMIBOLD),tx("7",46,INK,SEMIBOLD,"JetBrains Mono"),tx("candidate  4  =  6",28,INK,SEMIBOLD)).arrange(DOWN,buff=.16).to_edge(RIGHT,buff=.45).shift(UP*2)
        q=tx("STOP  OR  KEEP  LOOKING?",33,INK,SEMIBOLD).to_edge(DOWN,buff=.45)
        wash=SurroundingRectangle(q,stroke_width=0,fill_color=YELLOW,fill_opacity=.28,buff=.14).set_z_index(-1)
        self.play(AddTextLetterByLetter(bench),run_time=1.5)
        self.play(AddTextLetterByLetter(q),FadeIn(wash),run_time=1.4)

    def draw_B04(self):
        self.add(heading("Look, then leap"))
        ranks=[4,7,2,6,8,1,5,3]
        states=["OBSERVE"]*3+["TEST"]+["HIDDEN"]*4
        colors=[BLUE]*3+[ORANGE]+[GRAY]*4
        cards=row(ranks,states,colors).scale(.78).to_edge(LEFT,buff=.28); self.add(cards)
        panel=VGroup(tx("benchmark",22,GRAY,SEMIBOLD),tx("7",44,INK,SEMIBOLD,"JetBrains Mono"),tx("6 < 7  ·  REJECT",25,INK,SEMIBOLD)).arrange(DOWN,buff=.18).to_edge(RIGHT,buff=.35).shift(UP*2)
        self.play(Write(panel),cards[3][0].animate.set_color(VERM).set_fill(VERM,opacity=.18),Transform(cards[3][3],tx("REJECT",16,INK,SEMIBOLD).move_to(cards[3][3])),run_time=1.8)
        self.play(cards[4].animate.set_opacity(1),cards[4][0].animate.set_color(TEAL).set_fill(TEAL,opacity=.24),Transform(cards[4][3],tx("ACCEPT",16,INK,SEMIBOLD).move_to(cards[4][3])),run_time=1.7)
        self.play(Transform(panel[2],tx("8 > 7  ·  ACCEPT",25,INK,SEMIBOLD).move_to(panel[2])),run_time=1)
        self.play(Write(tx("GLOBAL  BEST  SELECTED",32,INK,SEMIBOLD).to_edge(DOWN,buff=.38)),run_time=1.2)

    def draw_B05(self):
        self.add(heading("Same policy · two outcomes"))
        good=[5,7,2,6,8,4,3,1]
        g=row(good,["OBSERVE"]*3+["REJECT","ACCEPT"]+["—"]*3,[BLUE]*3+[VERM,TEAL]+[GRAY]*3,y=.75).scale(.82)
        self.play(LaggedStart(*[FadeIn(c) for c in g],lag_ratio=.08),run_time=1.8)
        goodlab=tx("best after window  ·  earlier rival inside  →  WIN",27,INK,SEMIBOLD).next_to(g,DOWN,buff=.35)
        self.play(AddTextLetterByLetter(goodlab),run_time=1.4)
        bad=[5,8,2,6,7,4,3,1]
        b=row(bad,["OBSERVE"]*3+["REJECT"]*5,[BLUE]*3+[VERM]*5,y=-1.65).scale(.82)
        self.play(ReplacementTransform(g.copy(),b),run_time=2.2)
        badlab=tx("best inside window  ·  benchmark unbeatable  →  LOSE",27,INK,SEMIBOLD).next_to(b,DOWN,buff=.3)
        self.play(AddTextLetterByLetter(badlab),run_time=1.4)

    def draw_B06(self):
        self.add(heading("What licenses look-then-leap?"))
        labels=["random  arrival  order","known  total  count","irrevocable  choices","comparable  relative  ranks","goal:  the  unique  best"]
        left=VGroup()
        for i,s in enumerate(labels):
            mark=Circle(.12,color=TEAL,fill_color=TEAL,fill_opacity=.7)
            line=VGroup(mark,tx(s,25,INK,SEMIBOLD)).arrange(RIGHT,buff=.18)
            left.add(line)
        left.arrange(DOWN,aligned_edge=LEFT,buff=.24).move_to([-3.8,.15,0])
        self.play(LaggedStart(*[FadeIn(x,shift=RIGHT*.15) for x in left],lag_ratio=.18),run_time=3)
        right=VGroup(tx("WIN  IFF",27,GRAY,SEMIBOLD),tx("best arrives",25),tx("AFTER  the window",30,INK,SEMIBOLD),tx("and earlier rival",25),tx("lies  INSIDE  it",30,INK,SEMIBOLD)).arrange(DOWN,buff=.16).move_to([3.75,.25,0])
        box=RoundedRectangle(width=4.15,height=3.25,corner_radius=.12,color=GRAY,fill_color=YELLOW,fill_opacity=.14).move_to(right)
        self.play(Create(box),LaggedStart(*[AddTextLetterByLetter(x) for x in right],lag_ratio=.16),run_time=2.6)

    def draw_B07(self):
        self.add(heading("Information versus opportunity"))
        axes=Axes(x_range=[0,1,.2],y_range=[0,.5,.1],x_length=7.0,y_length=4.2,axis_config={"color":GRAY,"include_numbers":True,"font_size":20}).to_edge(LEFT,buff=.75).shift(DOWN*.45)
        xl=tx("fraction  observed",24,INK).next_to(axes,DOWN,buff=.25)
        curve=axes.plot(lambda x: (-x*np.log(x)) if x>0.01 else 0,x_range=[.02,.98],color=BLUE,stroke_width=5)
        peak=Dot(axes.c2p(1/np.e,1/np.e),radius=.11,color=TEAL)
        guides=VGroup(DashedLine(axes.c2p(1/np.e,0),axes.c2p(1/np.e,1/np.e),color=GRAY),DashedLine(axes.c2p(0,1/np.e),axes.c2p(1/np.e,1/np.e),color=GRAY))
        self.play(Create(axes),Write(xl),Create(curve),run_time=3)
        self.play(Create(guides),FadeIn(peak),run_time=1.2)
        panel=VGroup(tx("n = 4",23,GRAY,SEMIBOLD,"JetBrains Mono"),tx("reject  1",30,INK,SEMIBOLD),tx("11 / 24  wins",30,INK,SEMIBOLD,"JetBrains Mono"),tx("large  n",23,GRAY,SEMIBOLD),tx("observe  ≈  37%",30,INK,SEMIBOLD),tx("success  ≈  37%",30,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.18).move_to([4.35,.3,0])
        self.play(LaggedStart(*[AddTextLetterByLetter(x) for x in panel],lag_ratio=.13),run_time=2.6)
        cap=tx("BALANCE",31,INK,SEMIBOLD).to_edge(DOWN,buff=.35).shift(RIGHT*3.9)
        wash=SurroundingRectangle(cap,stroke_width=0,fill_color=YELLOW,fill_opacity=.28,buff=.12).set_z_index(-1)
        self.play(Write(cap),FadeIn(wash),run_time=1)

for _bid in [b["beat_id"] for b in SHEET["beats"] if b["render"] == "manim"]:
    globals()[_bid] = type(_bid, (ReelBeat,), {"BID": _bid})
