from manim import *
import json, os
from pathlib import Path

ROOT = Path(__file__).parent
SHEET = json.loads((ROOT / "beat_sheet.json").read_text())
DURS = {b["beat_id"]: float(b.get("actual_duration_s") or b.get("estimated_duration_s") or 6) for b in SHEET["beats"]}
BG, INK, GRAY = "#F0EAD6", "#000000", "#4D4D4D"
TEAL, VERM, YELLOW = "#009E73", "#D55E00", "#F0E442"
BLUE, ORANGE, SKY, PURPLE = "#0072B2", "#E69F00", "#56B4E9", "#CC79A7"
config.background_color = BG

def tx(s, size=32, color=INK, weight=NORMAL):
    return Text(s, font="EB Garamond", font_size=size, color=color, weight=weight)

def title(s):
    t=tx(s,42,INK,SEMIBOLD).to_edge(UP,buff=.35)
    r=Line(LEFT*6.4,RIGHT*6.4,color=GRAY,stroke_width=1).next_to(t,DOWN,buff=.15)
    return VGroup(t,r)

POS={"S":(-5,0),"E":(-2.6,-2),"D":(.1,-1.45),"A":(-1.8,1.65),"C":(1.5,1.2),"B":(4.4,.1)}
EDGES=[("S","A","10"),("S","E","8"),("E","D","1"),("D","A","−4"),("D","C","−1"),("A","C","2"),("C","B","−2"),("B","A","?")]

def graph(edge_override=None, scale=.78, shift=LEFT*1.9, overrides=None):
    pts={k:np.array([x,y,0])*scale+shift for k,(x,y) in POS.items()}
    es=VGroup(); labels=VGroup()
    for u,v,w in EDGES:
        if edge_override and (u,v)==("D","A"): w=edge_override
        if overrides and (u,v) in overrides: w=overrides[(u,v)]
        a,b=pts[u],pts[v]; vec=b-a; unit=vec/np.linalg.norm(vec)
        ar=Arrow(a+unit*.32,b-unit*.32,buff=0,color=GRAY,stroke_width=2,max_tip_length_to_length_ratio=.08)
        lab=tx(w,24,VERM if w.startswith("−") else GRAY).move_to((a+b)/2+rotate_vector(unit,PI/2)*.22)
        if w=="?": ar.set_opacity(.35); lab.set_opacity(.55)
        es.add(ar); labels.add(lab)
    ns=VGroup(*[VGroup(Circle(.3,color=INK,fill_color=BG,fill_opacity=1),tx(k,25,INK,SEMIBOLD)).move_to(p) for k,p in pts.items()])
    return VGroup(es,labels,ns),pts

def distance_panel(rows, highlight=None):
    heads=["k","S","A","B","C","D","E"]
    data=[heads]+[[str(i)]+r for i,r in enumerate(rows)]
    tab=Table(data,include_outer_lines=True,line_config={"color":GRAY,"stroke_width":1},element_to_mobject=lambda s:tx(s,21)).scale(.64)
    tab.to_edge(RIGHT,buff=.35).shift(DOWN*.2)
    if highlight:
        rr,cc=highlight
        tab.add_to_back(SurroundingRectangle(tab.get_cell((rr+2,cc+2)),color=TEAL,fill_color=YELLOW,fill_opacity=.28,buff=.01))
    return tab

class ReelBeat(Scene):
    BID=""
    def construct(self):
        self.camera.background_color=BG
        fn=getattr(self,"draw_"+self.BID.replace("-","_"),self.draw_default)
        fn(); elapsed=self.renderer.time
        if elapsed < DURS[self.BID]: self.wait(DURS[self.BID]-elapsed)
    def head(self,s): self.add(title(s))
    def draw_default(self):
        self.play(Write(tx(self.BID,70).move_to(ORIGIN)),run_time=1)
    def draw_B01(self):
        self.head("The confident answer")
        g,p=graph(scale=.68,shift=LEFT*3.25); self.play(Create(g),run_time=4)
        direct=Arrow(p["S"]+.3*UP,p["A"]+.2*DOWN,color=VERM,stroke_width=9,buff=.25)
        lock=VGroup(tx("LOCKED",24,INK,SEMIBOLD),tx("10",56,INK,SEMIBOLD)).arrange(DOWN).to_edge(RIGHT,buff=1.45).shift(UP*1.55)
        card=VGroup(tx("DIJKSTRA LICENSE",25,INK,SEMIBOLD),tx("every weight ≥ 0",29,INK),tx("VOID: edge −4",28,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.18).to_edge(RIGHT,buff=.45).shift(DOWN*.45)
        self.play(GrowArrow(direct),Write(lock),run_time=3); self.play(Write(card),run_time=3)
        wash=SurroundingRectangle(card[-1],buff=.06,stroke_width=0,fill_color=YELLOW,fill_opacity=.28).set_z_index(-1)
        self.play(FadeIn(wash),run_time=.5); self.wait(1); self.play(FadeOut(wash),run_time=.5)
    def draw_B02(self):
        self.head("One hop-budget at a time")
        g,p=graph(scale=.64,shift=LEFT*3.1); self.play(Create(g),run_time=3)
        rows=[["0","∞","∞","∞","∞","∞"],["0","10","∞","∞","∞","8"],["0","10","∞","12","9","8"]]
        tab=distance_panel(rows); self.play(Create(tab),run_time=4)
        labs=VGroup(tx("one hop",25,INK),tx("two hops",25,INK)).arrange(DOWN,aligned_edge=LEFT).next_to(tab,DOWN,buff=.25)
        self.play(Write(labs),run_time=2)
    def draw_B03(self):
        self.head("Predict before the relaxation")
        g,p=graph(); self.add(g)
        q=VGroup(tx("D = 9",38),tx("D → A = −4",38,INK),tx("A = 10",38),tx("Does A change?",42,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.2).to_edge(RIGHT,buff=.55)
        self.play(Write(q),run_time=4); self.wait(2)
        ans=tx("9 − 4 = 5",54,INK,SEMIBOLD).move_to(q[-1]); self.play(ReplacementTransform(q[-1],ans),run_time=2)
    def draw_B04(self):
        self.head("The truth travels one edge per round")
        rows=[["0","∞","∞","∞","∞","∞"],["0","10","∞","∞","∞","8"],["0","10","∞","12","9","8"],["0","5","10","8","9","8"],["0","5","6","8","9","8"],["0","5","5","8","9","8"]]
        tab=distance_panel(rows).scale(1.15).move_to(ORIGIN+DOWN*.2); self.play(Create(tab),run_time=5)
        for cell in [(3,1),(3,2),(3,3),(4,2),(5,2)]: self.play(Indicate(tab.get_cell((cell[0]+2,cell[1]+2)),color=TEAL),run_time=.8)
        self.play(Write(tx("FINAL:  S 0 · A 5 · B 5 · C 8 · D 9 · E 8",31,INK,SEMIBOLD).to_edge(DOWN,buff=.35)),run_time=2)
    def draw_B05(self):
        self.head("Dynamic programming: different questions, solved once")
        rows=[["0","∞","∞","∞","∞","∞"],["0","10","∞","∞","∞","8"],["0","10","∞","12","9","8"],["0","5","10","8","9","8"],["0","5","6","8","9","8"],["0","5","5","8","9","8"]]
        tab=distance_panel(rows).scale(.95).to_edge(LEFT,buff=.3); self.play(Create(tab),run_time=6)
        for i in range(6): self.play(Indicate(tab.get_rows()[i+1],color=TEAL),run_time=.6)
        laws=VGroup(tx("1  ORDERING",34,INK,SEMIBOLD),tx("smaller hop-budgets first",28),tx("2  PERMANENCE",34,INK,SEMIBOLD),tx("a solved layer never changes",28),tx("each (v, k) solved exactly once",31,INK,SEMIBOLD)).arrange(DOWN,aligned_edge=LEFT,buff=.2).to_edge(RIGHT,buff=.4)
        self.play(Write(laws),run_time=5)
        roll=VGroup(*[tx(x,27) for x in ["0","5","5","8","9","8"]]).arrange(RIGHT,buff=.28).next_to(laws,DOWN,buff=.45)
        self.play(TransformFromCopy(tab.get_rows()[-1],roll),run_time=3)
        # Doubled spaces: Pango swallows single-space advances at these
        # boundaries on this machine (rendered "layersshare…answersdo notchange").
        # Clamp keeps the line inside the safe area — it ran off frame right.
        cap = tx("layers  share storage — answers  do  not  change",26,GRAY).next_to(roll,DOWN)
        if cap.get_right()[0] > 6.6: cap.shift(LEFT*(cap.get_right()[0]-6.6))
        self.play(Write(cap),run_time=2)
    def equation(self):
        return MathTex(r"d_k(v)=\min\!\left(d_{k-1}(v),\ \min_{(u,v)\in E}[d_{k-1}(u)+w(u,v)]\right)",color=INK,font_size=39).to_edge(LEFT,buff=.35).shift(UP*.75)
    def draw_B06(self):
        self.head("The recurrence the motion earned")
        eq=self.equation(); self.play(Write(eq),run_time=3)
        self.play(Write(tx("keep yesterday  OR  arrive on one final edge",30,GRAY).next_to(eq,DOWN,buff=.55)),run_time=2)
    # STAGE LAYOUT LAW: subject (graph) LEFT, numbers (layer table + calc)
    # RIGHT, equation on top — and the equation OPERATES: its terms drive
    # visible updates on both sides. (Bear, 2026-08-28: no parked equations,
    # no prose panels; Okabe-Ito on marks only, all text ink.)
    K2 = ["0","10","∞","12","9","8"]
    K3 = ["0","5","10","8","9","8"]
    def tangent_stage(self, k3_shown=False):
        self.add(title("Bellman-Ford · equation"))
        eq = MathTex(r"d_k(v)=\min\!\left(d_{k-1}(v),\ \min_{(u,v)\in E}[d_{k-1}(u)+w(u,v)]\right)",
                     color=INK, font_size=34).next_to(title("x"), DOWN, buff=.05).to_edge(UP, buff=1.25)
        g, pts = graph(scale=.5, shift=LEFT*4.1+DOWN*1.55)
        # k=2 layer values ride the graph as badges — the subject shows the numbers too
        vals = dict(zip(["S","A","B","C","D","E"], self.K2))
        badges = VGroup(*[tx(vals[k], 22, INK, SEMIBOLD).move_to(
            pts[k] + UP*.5 + RIGHT*.52) for k in ["S","A","B","C","D","E"]])
        rows = [["k","S","A","B","C","D","E"], ["2"]+self.K2]
        if k3_shown: rows.append(["3"]+self.K3)
        tab = Table(rows, include_outer_lines=True,
                    line_config={"color":GRAY,"stroke_width":1},
                    element_to_mobject=lambda t: tx(t,20)).scale(.52)
        tab.to_edge(RIGHT, buff=.4).shift(UP*.35)
        self.add(eq, g, badges, tab)
        return eq, g, pts, badges, tab
    def draw_B06A(self):
        eq, g, pts, badges, tab = self.tangent_stage()
        lhs = SurroundingRectangle(eq[0][0:6], color=GRAY, buff=.08)
        rhs = SurroundingRectangle(eq[0][7:], color=GRAY, buff=.08)
        self.play(Create(lhs), run_time=1.5)
        ring = Circle(.42, color=TEAL, stroke_width=4).move_to(pts["A"])
        cap1 = tx("best distance to A,  at most k edges", 24, GRAY).to_edge(DOWN, buff=.5)
        self.play(Create(ring), Write(cap1), run_time=2.5)
        self.wait(1.5)
        cap2 = tx("keep the row above  —  or arrive on one final edge", 24, GRAY).move_to(cap1)
        self.play(ReplacementTransform(lhs, rhs), ReplacementTransform(cap1, cap2),
                  Indicate(tab.get_rows()[1], color=TEAL), run_time=3)
    def draw_B06B(self):
        eq, g, pts, badges, tab = self.tangent_stage()
        def callout(text, target, direction=DOWN):
            lab = tx(text, 22, INK).next_to(target, direction, buff=.35)
            return VGroup(lab, Arrow(lab.get_edge_center(-direction), target.get_edge_center(direction),
                          color=GRAY, stroke_width=2, buff=.06, max_tip_length_to_length_ratio=.2))
        seq = [
            callout("k — the row", tab.get_rows()[1][0], LEFT),
            callout("v — the node being asked", g[2][3], UP),
            callout("u — a neighbor", g[2][2], DOWN),
            callout("w — the edge cost", g[1][3], DOWN),
        ]
        for c in seq:
            self.play(FadeIn(c), run_time=1.6); self.wait(.6); self.play(FadeOut(c), run_time=.6)
        m = SurroundingRectangle(eq[0][7:10], color=TEAL, buff=.06)
        self.play(Create(m), Write(tx("min — keep the smaller", 24, GRAY).to_edge(DOWN, buff=.5)), run_time=2)
    def draw_B06C(self):
        eq, g, pts, badges, tab = self.tangent_stage(k3_shown=True)
        row3_cells = tab.get_rows()[2]
        row3_targets = [m.copy() for m in row3_cells]
        for m in row3_cells: m.set_opacity(0)   # empty row waits on the grid
        calc_keep = tx("keep:  10", 27, INK).to_edge(RIGHT, buff=1.7).shift(DOWN*1.0)
        a_cell = tab.get_rows()[1][2]
        self.play(TransformFromCopy(a_cell, calc_keep), run_time=2)
        calc_arr = tx("arrive:  9 + (−4) = 5", 27, INK).next_to(calc_keep, DOWN, aligned_edge=LEFT, buff=.25)
        d_cell = tab.get_rows()[1][5]
        self.play(TransformFromCopy(d_cell, calc_arr), Indicate(g[1][3], color=VERM), run_time=2.5)
        calc_min = tx("min(10, 5) = 5", 30, INK, SEMIBOLD).next_to(calc_arr, DOWN, aligned_edge=LEFT, buff=.3)
        self.play(Write(calc_min), run_time=1.5)
        # the 5 lands in BOTH places at once: k=3 row fills IN the grid,
        # the badge on the graph updates in the same breath
        newbadge = tx("5", 22, INK, SEMIBOLD).move_to(badges[1])
        wash = SurroundingRectangle(row3_targets[2], color=TEAL, fill_color=YELLOW,
                                    fill_opacity=.3, buff=.02)
        self.play(LaggedStart(*[FadeIn(m) for m in row3_targets], lag_ratio=.08), run_time=2)
        self.play(FadeIn(wash), Transform(badges[1], newbadge),
                  Flash(pts["A"], color=TEAL, line_length=.25), run_time=2)
    def draw_B06D(self):
        eq, g, pts, badges, tab = self.tangent_stage(k3_shown=True)
        deps = VGroup()
        for col in [1,2,4,5]:
            src = tab.get_rows()[1][col]; dst = tab.get_rows()[2][col]
            deps.add(Arrow(src.get_bottom(), dst.get_top(), color=GRAY,
                           stroke_width=2, buff=.08, max_tip_length_to_length_ratio=.25))
        self.play(LaggedStart(*[GrowArrow(a) for a in deps], lag_ratio=.2), run_time=3)
        cap = tx("every dependency comes  from the smaller  layer", 24, GRAY).to_edge(DOWN, buff=.5)
        self.play(Write(cap), run_time=2)
        lock = SurroundingRectangle(tab.get_rows()[1], color=TEAL, buff=.04)
        cap2 = tx("every finished  layer stays  finished — back to the graph", 24, GRAY).move_to(cap)
        self.play(Create(lock), ReplacementTransform(cap, cap2),
                  Indicate(VGroup(*[m for m in g[2]]), color=TEAL), run_time=3)
    def draw_B07(self):
        # Informal proof, plain language (Bear 2026-08-28) — and the honest
        # cycle: the graph's ONLY cycle is A→C→B→A, so trace 2's one changed
        # weight is B→A = −1 (sum 2 − 2 − 1 = −1). The earlier D→A = −9
        # version was WRONG: that edge closes no loop, round 6 would not fire.
        self.head("The sixth round is a smoke detector")
        g,p=graph(scale=.56,shift=LEFT*3.35+DOWN*.35,overrides={("B","A"):"−1"})
        # B→A straight is nearly collinear with A→C→B — draw it as an arc so
        # the cycle reads as a loop, not a doubled edge. Hide the straight one.
        g[0][7].set_opacity(0); g[1][7].set_opacity(0)
        back=CurvedArrow(p["B"]+DOWN*.28,p["A"]+DOWN*.3+LEFT*.05,angle=1.05,
                         color=GRAY,stroke_width=2,tip_length=.16)
        backlab=tx("−1",22,VERM).move_to(back.point_from_proportion(.5)+DOWN*.3)
        self.play(Create(g),Create(back),FadeIn(backlab),run_time=3)
        proof=VGroup(
            tx("a  route  that  repeats  no  node",24,INK),
            tx("uses  at  most  5  edges   (6  nodes)",24,INK),
            tx("invariant:  5  rounds  price  every  such  route",24,INK),
            tx("round  6  improves  ⇒  the  route  repeats  a  node",24,INK),
            tx("cut  the  loop  out  —  the  rest  is  already  priced",24,INK),
            tx("so  the  gain  is  the  loop's  own  total:   <  0",24,INK),
            tx("NEGATIVE  CYCLE — no finite  shortest path",24,INK,SEMIBOLD),
        ).arrange(DOWN,aligned_edge=LEFT,buff=.24).to_edge(RIGHT,buff=.4).shift(UP*.3)
        for i in [0,1]: self.play(Write(proof[i]),run_time=1.4)
        self.play(Write(proof[2]),run_time=1.6)
        self.play(Write(proof[3]),run_time=1.4)
        # the cycle mark = the actual cycle edges (A→C idx5, C→B idx6, B→A idx7)
        cyc=VGroup(g[0][5].copy(),g[0][6].copy(),back.copy())
        cyc.set_color(VERM).set_stroke(width=4).set_opacity(1)
        self.play(Create(cyc),Write(proof[4]),run_time=2)
        self.add(g[2])   # nodes back on top — the loop threads them, never buries them
        lap=tx("lap  cost:   2 − 2 − 1  =  −1",25,INK).next_to(proof,DOWN,aligned_edge=LEFT,buff=.35)
        self.play(Write(proof[5]),Write(proof[6]),Write(lap),run_time=2.5)
    def draw_B08(self):
        self.head("The road marked 10 was not shortest")
        g,p=graph(); self.add(g)
        path=VGroup(Arrow(p["S"],p["E"],buff=.35,color=TEAL,stroke_width=9),Arrow(p["E"],p["D"],buff=.35,color=TEAL,stroke_width=9),Arrow(p["D"],p["A"],buff=.35,color=TEAL,stroke_width=9))
        self.play(Create(path),run_time=4)
        answer=VGroup(tx("S → E → D → A",38,INK,SEMIBOLD),tx("8 + 1 − 4 = 5",55,INK,SEMIBOLD),tx("Dijkstra license: non-negative edges",27,INK),tx("DP license: ordering + permanence",27,INK)).arrange(DOWN,buff=.2).to_edge(RIGHT,buff=.35)
        self.play(Write(answer),run_time=5)
    def draw_B09(self):
        self.head("YOUR TURN · from Module 6 quiz")
        q=tx("Which algorithms use dynamic programming?",38,INK,SEMIBOLD).shift(UP*2.1)
        opts=VGroup(*[tx(x,30) for x in ["□ QuickSort","□ Floyd–Warshall","□ Bellman–Ford","□ Dijkstra’s algorithm","□ Longest Increasing Subsequence"]]).arrange(DOWN,aligned_edge=LEFT,buff=.18).shift(LEFT*2+DOWN*.25)
        prompt=VGroup(tx("PLAY WITH IT",27,INK,SEMIBOLD),tx("Change one edge weight.",30),tx("Predict before every relaxation.",30),tx("Then run the trace.",30)).arrange(DOWN,aligned_edge=LEFT,buff=.22).to_edge(RIGHT,buff=.65)
        self.play(AddTextLetterByLetter(q),run_time=2); self.play(LaggedStart(*[AddTextLetterByLetter(x) for x in opts],lag_ratio=.3),run_time=5); self.play(Write(prompt),run_time=3)
    def draw_BOUT(self):
        self.head("BELLMAN–FORD")
        lines=VGroup(tx("one hop-budget layer",42,INK,SEMIBOLD),tx("solved once",55,INK,SEMIBOLD),tx("one extra pass",42,INK,SEMIBOLD),tx("detects a negative cycle",47,INK,SEMIBOLD),tx("THE ROAD THAT PAYS YOU",29,INK,SEMIBOLD)).arrange(DOWN,buff=.28).move_to(ORIGIN+DOWN*.2)
        self.play(LaggedStart(*[AddTextLetterByLetter(x) for x in lines],lag_ratio=.35),run_time=6)

for _bid in [b["beat_id"] for b in SHEET["beats"] if b["beat_id"]!="B00"]:
    globals()[_bid] = type(_bid,(ReelBeat,),{"BID":_bid})
