from manim import *
import json
from pathlib import Path

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or 6) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW, BLUE, ORANGE, SKY = "#009E73", "#D55E00", "#F0E442", "#0072B2", "#E69F00", "#56B4E9"
config.background_color = BG

def tx(s, size=30, color=INK, weight=NORMAL, font="EB Garamond"):
    return Text(s, font=font, font_size=size, color=color, weight=weight)

def heading(s):
    t = tx(s, 40, INK, SEMIBOLD).to_edge(UP, buff=.28)
    return VGroup(t, Line(LEFT*6.5, RIGHT*6.5, color=GRAY, stroke_width=1).next_to(t, DOWN, buff=.12))

ITEMS = [("A",10,60,6.0,BLUE),("B",20,100,5.0,ORANGE),("C",30,120,4.0,SKY)]

def item_card(name, weight, value, density, color):
    box = RoundedRectangle(width=2.5, height=1.25, corner_radius=.12, color=GRAY,
                           fill_color=color, fill_opacity=.18)
    labels = VGroup(tx(name,30,INK,SEMIBOLD), tx(f"weight  {weight}",22),
                    tx(f"value  {value}",22), tx(f"density  {density:g}",20,GRAY)).arrange(DOWN,buff=.05)
    labels.move_to(box)
    return VGroup(box, labels)

def item_stack(scale=.9, shift=LEFT*3.8):
    g = VGroup(*[item_card(*it) for it in ITEMS]).arrange(DOWN,buff=.22).scale(scale).move_to(shift+DOWN*.15)
    return g

def bag(label="capacity  50"):
    outline = Polygon(LEFT*1.25+UP*1.65, RIGHT*1.25+UP*1.65, RIGHT*1.05+DOWN*1.5,
                      LEFT*1.05+DOWN*1.5, color=GRAY, stroke_width=3)
    lab = tx(label,25,INK,SEMIBOLD).next_to(outline,UP,buff=.15)
    return VGroup(outline,lab)

def grid(rows, caps=(0,10,20,30,40,50), scale=.7):
    data = [["items\\cap"]+[str(x) for x in caps]] + rows
    return Table(data, include_outer_lines=True,
                 line_config={"color":GRAY,"stroke_width":1},
                 element_to_mobject=lambda s:tx(str(s),20)).scale(scale)

ROWS = [
    ["0","0","0","0","0","0","0"],
    ["A","0","60","60","60","60","60"],
    ["A,B","0","60","100","160","160","160"],
    ["A,B,C","0","60","100","160","180","220"],
]

def full_stage(include_c=True):
    cards=item_stack(.62,LEFT*4.65+UP*.1)
    use_rows=ROWS if include_c else ROWS[:3]
    tab=grid(use_rows,scale=.63).to_edge(RIGHT,buff=.35).shift(DOWN*.15)
    return cards,tab

def recurrence(size=34):
    return MathTex(r"DP[i,w]=\max\{DP[i-1,w],\ DP[i-1,w-w_i]+v_i\}",
                   color=INK,font_size=size)

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        getattr(self,"draw_"+self.BID)()
        if self.renderer.time < DURS[self.BID]: self.wait(DURS[self.BID]-self.renderer.time)
    def head(self,s): self.add(heading(s))

    def draw_B01(self):
        self.head("Three indivisible items")
        cards=item_stack(); self.play(LaggedStart(*[FadeIn(c,shift=RIGHT*.3) for c in cards],lag_ratio=.25),run_time=3)
        bg=bag().to_edge(RIGHT,buff=1.25).shift(DOWN*.25); self.play(Create(bg),run_time=1.5)
        a,b,c=[x.copy().scale(.6) for x in cards]
        a.move_to(bg[0].get_center()+UP*.65); b.move_to(bg[0].get_center()+DOWN*.15)
        self.play(TransformFromCopy(cards[0],a),run_time=1.5); self.play(TransformFromCopy(cards[1],b),run_time=1.5)
        total=VGroup(tx("GREEDY",24,INK,SEMIBOLD),tx("weight  30 / 50",27),tx("value  160",42,INK,SEMIBOLD)).arrange(DOWN,buff=.14).next_to(bg,DOWN,buff=.25)
        self.play(Write(total),run_time=2)
        cross=Cross(cards[2],stroke_color=VERM,stroke_width=6); reject=tx("C  rejected",25,INK,SEMIBOLD).next_to(cards[2],RIGHT,buff=.2)
        self.play(Create(cross),Write(reject),run_time=2)

    def draw_B02(self):
        self.head("The locally best item blocks the best pair")
        cards=item_stack(); self.add(cards)
        bg=bag().to_edge(RIGHT,buff=1.25).shift(DOWN*.5); self.add(bg)
        b,c=[cards[i].copy().scale(.6) for i in (1,2)]; b.move_to(bg[0].get_center()+UP*.62); c.move_to(bg[0].get_center()+DOWN*.32)
        self.play(TransformFromCopy(cards[1],b),TransformFromCopy(cards[2],c),run_time=3)
        best=VGroup(tx("OPTIMAL",23,INK,SEMIBOLD),tx("weight  20 + 30 = 50",25),tx("value  100 + 120 = 220",32,INK,SEMIBOLD)).arrange(DOWN,buff=.12).next_to(bg,DOWN,buff=.14)
        self.play(Write(best),run_time=2.5)
        license=VGroup(tx("GREEDY  LICENSE",22,INK,SEMIBOLD),tx("greedy-choice  property",21),tx("ABSENT — items  indivisible",22,INK,SEMIBOLD)).arrange(DOWN,buff=.06).move_to(RIGHT*3.55+UP*2.62)
        wash=SurroundingRectangle(license[-1],stroke_width=0,fill_color=YELLOW,fill_opacity=.32,buff=.08).set_z_index(-1)
        self.play(Write(license),FadeIn(wash),run_time=2)

    def draw_B03(self):
        self.head("Instance two: smaller capacities first")
        cards=item_stack(.55,LEFT*4.8); self.add(cards)
        rows=[["0","0","0","0"],["A","0","60","60"],["A,B","0","60","100"]]
        tab=grid(rows,caps=(0,10,20),scale=.88).to_edge(RIGHT,buff=.75).shift(DOWN*.2)
        self.play(Create(tab),run_time=3)
        for r in range(2,5):
            self.play(Indicate(tab.get_rows()[r-1],color=TEAL),run_time=1)
        calc=VGroup(tx("at  (B,20)",27,INK,SEMIBOLD),tx("skip  60",28),tx("take  0 + 100",28),tx("answer  100",34,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.13).next_to(tab,DOWN,buff=.25)
        self.play(Write(calc),run_time=2)

    def draw_B04(self):
        self.head("Finished rows stay finished")
        cards,tab=full_stage(False); self.add(cards); self.play(Create(tab),run_time=4)
        old=SurroundingRectangle(tab.get_rows()[-1],color=TEAL,buff=.03)
        self.play(Create(old),run_time=1)
        c=cards[2].copy(); self.play(c.animate.shift(RIGHT*.35),run_time=1)
        nextrow=VGroup(*[tx(x,20) for x in ROWS[3]]).arrange(RIGHT,buff=.42).next_to(tab,DOWN,buff=.4)
        cap=tx("new  C-row · old  B-row  untouched",25,GRAY).next_to(nextrow,DOWN,buff=.18)
        self.play(LaggedStart(*[FadeIn(x,shift=UP*.15) for x in nextrow],lag_ratio=.12),Write(cap),run_time=3)

    def draw_B05(self):
        self.head("Predict the decisive cell")
        cards,tab=full_stage(); self.add(cards,tab)
        cell=tab.get_cell((5,7)); wash=SurroundingRectangle(cell,color=TEAL,fill_color=YELLOW,fill_opacity=.32,buff=.02)
        self.play(FadeIn(wash),run_time=1)
        q=VGroup(tx("SKIP",25,INK,SEMIBOLD),tx("160",43,INK,SEMIBOLD),tx("or",24,GRAY),tx("TAKE",25,INK,SEMIBOLD),tx("100 + 120",41,INK,SEMIBOLD),tx("commit.",28,INK,SEMIBOLD)).arrange(DOWN,buff=.1).move_to(LEFT*.15+DOWN*.35)
        self.play(Write(q),run_time=4); self.wait(2)
        ans=tx("220  wins",47,INK,SEMIBOLD).move_to(q)
        self.play(ReplacementTransform(q,ans),Indicate(cell,color=TEAL),run_time=2)

    def draw_B06(self):
        self.head("Dynamic programming: each question solved once")
        cards,tab=full_stage(); tab.scale(.82).to_edge(LEFT,buff=.3); self.play(Create(tab),run_time=4)
        for row in tab.get_rows()[1:]: self.play(Indicate(row,color=TEAL),run_time=.65)
        laws=VGroup(tx("1  ORDERING",32,INK,SEMIBOLD),tx("earlier\u2003items\u2003—\u2003smaller\u2003capacity",23),tx("2  PERMANENCE",32,INK,SEMIBOLD),tx("a\u2003solved\u2003(i,w)\u2003answer\u2003never\u2003changes",22),tx("SOLVED  EXACTLY  ONCE",31,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.2).to_edge(RIGHT,buff=.35)
        self.play(Write(laws),run_time=4)
        roll=VGroup(*[tx(x,24) for x in ROWS[-1][1:]]).arrange(RIGHT,buff=.2).next_to(laws,DOWN,buff=.35)
        self.play(TransformFromCopy(tab.get_rows()[-1],roll),run_time=2)
        self.play(Write(tx("rows\u2003share\u2003storage\u2003—\u2003answers\u2003stay\u2003fixed",22,GRAY).next_to(roll,DOWN,buff=.15)),run_time=1.5)

    def draw_B07(self):
        self.head("The recurrence the motion earned")
        cards,tab=full_stage(); cards.scale(.82).to_edge(LEFT,buff=.35); tab.scale(.72).to_edge(RIGHT,buff=.3).shift(DOWN*.55)
        eq=recurrence(38).to_edge(UP,buff=1.2).shift(RIGHT*.9)
        self.play(Write(eq),Create(cards),Create(tab),run_time=4)
        self.play(Indicate(cards[2],color=TEAL),Indicate(tab.get_cell((5,7)),color=TEAL),run_time=2)

    def tangent(self):
        self.add(heading("0/1 knapsack · equation"))
        eq=recurrence(34).to_edge(UP,buff=1.15)
        cards=item_stack(.48,LEFT*4.85+DOWN*1.0)
        tab=grid(ROWS[2:],scale=.55).to_edge(RIGHT,buff=.3).shift(DOWN*.95)
        self.add(eq,cards,tab)
        return eq,cards,tab

    def draw_B07A(self):
        eq,cards,tab=self.tangent()
        lhs=SurroundingRectangle(eq[0][:7],color=GRAY,buff=.06); rhs=SurroundingRectangle(eq[0][8:],color=GRAY,buff=.06)
        a=tx("best value  for this item-count and capacity",24,GRAY).to_edge(DOWN,buff=.35)
        self.play(Create(lhs),Write(a),run_time=2.5)
        b=tx("larger of SKIP  and  TAKE · equals means exact",24,GRAY).move_to(a)
        self.play(ReplacementTransform(lhs,rhs),ReplacementTransform(a,b),run_time=3)

    def draw_B07B(self):
        eq,cards,tab=self.tangent()
        gloss=VGroup(tx("symbol   role   meaning   domain",22,INK,SEMIBOLD),tx("i   index   item count   0…n",22),tx("w   variable   capacity   0…W",22),tx("vᵢ   fixed data   item value   ≥0",22),tx("wᵢ   fixed data   item weight   >0",22),tx("max   operator   larger candidate",22)).arrange(DOWN,aligned_edge=LEFT,buff=.13).move_to(RIGHT*2.8+UP*.15)
        self.play(LaggedStart(*[Write(x) for x in gloss],lag_ratio=.2),run_time=5)

    def draw_B07C(self):
        eq,cards,tab=self.tangent()
        calc=VGroup(tx("CELL  (C,50)",24,INK,SEMIBOLD),tx("skip:  160",29),tx("take:  100 + 120",29),tx("max(160, 220) = 220",34,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.16).move_to(RIGHT*2.65+UP*.05)
        self.play(Write(calc[0]),Write(calc[1]),run_time=2); self.play(Write(calc[2]),Indicate(cards[2],color=TEAL),run_time=2)
        self.play(Write(calc[3]),Indicate(tab.get_cell((3,7)),color=TEAL),run_time=2)
        bagset=VGroup(cards[1].copy(),cards[2].copy()).arrange(DOWN,buff=.08).scale(.72).move_to(LEFT*1.7+DOWN*.7)
        self.play(TransformFromCopy(VGroup(cards[1],cards[2]),bagset),run_time=2)

    def draw_B07D(self):
        eq,cards,tab=self.tangent()
        arrows=VGroup(*[Arrow(tab.get_cell((2,c)).get_bottom(),tab.get_cell((3,c)).get_top(),buff=.06,color=GRAY,stroke_width=2) for c in range(2,8)])
        self.play(LaggedStart(*[GrowArrow(a) for a in arrows],lag_ratio=.12),run_time=2.5)
        lock=SurroundingRectangle(tab.get_rows()[1],color=TEAL,buff=.03)
        cap=VGroup(tx("ORDERING · every arrow points from an earlier row",23),tx("PERMANENCE · finished answers stay finished",23)).arrange(DOWN,buff=.16).to_edge(DOWN,buff=.28)
        self.play(Create(lock),Write(cap),run_time=2.5)

    def draw_B08(self):
        self.head("The best-looking item loses")
        cards=item_stack(.72,LEFT*4.45); self.add(cards)
        greedy=VGroup(tx("DENSITY  GREEDY",21,INK,SEMIBOLD),tx("A + B",38),tx("value  160",45,INK,SEMIBOLD)).arrange(DOWN,buff=.13).move_to(LEFT*.85+DOWN*.1)
        dp=VGroup(tx("DYNAMIC  PROGRAMMING",21,INK,SEMIBOLD),tx("B + C",38),tx("value  220",45,INK,SEMIBOLD)).arrange(DOWN,buff=.13).move_to(RIGHT*3.85+DOWN*.1)
        self.play(Write(greedy),run_time=2); self.play(Write(dp),run_time=2)
        self.play(Circumscribe(dp[-1],color=TEAL),run_time=2)
        complexity=tx("(n+1)(W+1)\u2003cells\u2003·\u2003O(nW)\u2003time\u2003·\u2003O(nW)\u2003full-table\u2003space",23,GRAY).to_edge(DOWN,buff=.35)
        self.play(Write(complexity),run_time=2)

for _bid in ["B01","B02","B03","B04","B05","B06","B07","B07A","B07B","B07C","B07D","B08"]:
    globals()[_bid] = type(_bid,(ReelBeat,),{"BID":_bid})
