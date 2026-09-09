"""Manim scenes — claude-liam-algorithmic-foundations (INFO 6205 course intro)
Claude stage: cream #FAF9F5, ink #3D3929, terracotta #D97757 (fill washes
only), deep terracotta #A64A24 (accent TEXT/borders near text — WCAG 4.66:1).
LAYOUT LAW: subject LEFT, ledger/numbers RIGHT, causally synced.
4K-safe at authoring: serif >= 28 (caps or 34+ for x-height-only words),
mono >= 24 uppercase/digits, horizontal Arrow tip_length >= 0.22,
everything clamped inside |x| <= 6.1.
"""
from manim import *
import numpy as np, os

CREAM="#FAF9F5"; INK="#3D3929"; TERRA="#D97757"; DEEP="#A64A24"; MUTE="#8B7355"; GRAY="#A89F91"
config.background_color = CREAM
config.frame_rate = 24
config.pixel_height = int(os.environ.get("ART_MANIM_H","1080"))
config.pixel_width  = int(os.environ.get("ART_MANIM_W","1920"))

def _t(s,size=30,color=INK,**kw):
    return Text(s.replace(" ","  "),color=color,font_size=size,font="EB Garamond",**kw)
def _mono(s,size=24,color=MUTE,**kw):
    return Text(s,color=color,font_size=size,font="Menlo",**kw)
def _box(c,h_pad=0.28,v_pad=0.2,stroke=INK,sw=2.0,fill=CREAM,corner=0.12):
    r=RoundedRectangle(corner_radius=corner,width=c.width+2*h_pad,height=c.height+2*v_pad,
                       color=stroke,stroke_width=sw,fill_color=fill,fill_opacity=1)
    r.move_to(c); return VGroup(r,c)
def _rule(scene):
    r=Line(LEFT*6,RIGHT*6,color=INK,stroke_width=2).set_stroke(opacity=0.3)
    r.to_edge(DOWN,buff=0.53); scene.add(r); return r
def _clamp(m,lim=6.1):
    if m.get_right()[0]>lim: m.shift(LEFT*(m.get_right()[0]-lim))
    if m.get_left()[0]<-lim: m.shift(RIGHT*(-lim-m.get_left()[0]))
    return m
def check_overlaps(*mobs,margin=0.12,label=""):
    bb=lambda m:(m.get_left()[0],m.get_bottom()[1],m.get_right()[0],m.get_top()[1])
    v=[]
    for i,a in enumerate(mobs):
        la,ba,ra,ta=bb(a)
        for j,b in enumerate(mobs):
            if j<=i: continue
            lb,bb_,rb,tb=bb(b)
            if la<rb+margin and ra>lb-margin and ba<tb+margin and ta>bb_-margin:
                v.append(f"  {i}x{j}")
    print(f"[BBOX {label}] "+("OK" if not v else f"{len(v)} overlap(s):\n"+"\n".join(v)))
def _ledger(rows,size=28,buff=0.42):
    out=VGroup()
    for k,v in rows:
        key=_t(k,size=size,color=MUTE)
        val=v if isinstance(v,Mobject) else _t(str(v),size=size)
        out.add(VGroup(key,val).arrange(RIGHT,buff=0.32))
    out.arrange(DOWN,buff=buff,aligned_edge=LEFT); return out
def _foot(scene,text,under=True):
    f=_t(text,size=28); f.to_edge(DOWN,buff=0.95); _clamp(f)
    u=Line(f.get_left()+DOWN*0.22,f.get_right()+DOWN*0.22,color=DEEP,stroke_width=2.5)
    if under: scene.play(FadeIn(f),Create(u),run_time=0.8)
    else: scene.play(FadeIn(f),run_time=0.8)
    return f
def chip(label,size=28):
    return _box(_mono(label,size=size,color=INK),h_pad=0.2,v_pad=0.14,sw=1.8)


class B04_FiveProperties(Scene):
    """16.9s — the five slots light up; SUGGESTION strikes once all are lit."""
    def construct(self):
        _rule(self)
        inp=chip("INPUT",size=24)
        slots=["OUTPUT","FINITE","DEFINITE","EFFECTIVE"]
        boxes=VGroup(*[RoundedRectangle(corner_radius=0.09,width=1.55,height=0.62,
                       color=INK,stroke_width=2,fill_color=CREAM,fill_opacity=1) for _ in slots])
        row=VGroup(inp,*boxes).arrange(RIGHT,buff=0.22)
        row.move_to(UP*1.5); _clamp(row,lim=5.8)
        labels=VGroup(*[_mono(s,size=19,color=MUTE) for s in slots])
        for l,b in zip(labels,boxes): l.move_to(b)
        arrow=Arrow(inp.get_right()+RIGHT*0.05,boxes[0].get_left()+LEFT*0.05,
                    color=INK,stroke_width=3,tip_length=0.22,buff=0.05)
        self.play(FadeIn(inp),run_time=0.6)
        self.play(GrowArrow(arrow),FadeIn(boxes),run_time=0.8)
        for l,b in zip(labels,boxes):
            wash=b.copy().set_fill(TERRA,opacity=0.28).set_stroke(width=0)
            self.play(FadeIn(wash),FadeIn(l),l.animate.set_color(INK),run_time=0.55)
        sugg=_t("a suggestion",size=32,color=MUTE)
        sugg.move_to(DOWN*1.3)
        strike=Line(sugg.get_left()+LEFT*0.1,sugg.get_right()+RIGHT*0.1,
                    color=DEEP,stroke_width=3)
        self.play(FadeIn(sugg),run_time=0.7)
        self.play(Create(strike),run_time=0.6)
        _foot(self,"skip one, and it is not an algorithm")
        self.wait(3.4)
        check_overlaps(row,labels,sugg,label="B04")


class B05_GrowthCurves(Scene):
    """16.5s — five curves bunch near the origin, then quadratic tears away."""
    def construct(self):
        _rule(self)
        ax=Axes(x_range=[0,4,1],y_range=[0,4,1],x_length=6.0,y_length=3.4,
                axis_config={"color":INK,"stroke_width":2,"include_tip":False})
        ax.move_to(LEFT*3.0+UP*0.3)
        self.play(Create(ax),run_time=0.9)
        curves=[
            ("O(1)",   lambda x: 0.5,               MUTE),
            ("O(logn)",lambda x: 0.5*np.log(x+1)+0.3, INK),
            ("O(n)",   lambda x: 0.75*x,             INK),
            ("O(nlogn)",lambda x: 0.75*x*np.log(x+1.3)/1.6, INK),
            ("O(n^2)", lambda x: 0.28*x*x,           DEEP),
        ]
        graphs=VGroup()
        for name,f,col in curves:
            g=ax.plot(f,x_range=[0,3.0],color=col,stroke_width=3)
            graphs.add(g)
        self.play(LaggedStart(*[Create(g) for g in graphs],lag_ratio=0.15),run_time=1.6)
        ledger_rows=[(n,"") for n,_,_ in curves]
        led=VGroup(*[_mono(n,size=24,color=(DEEP if n=="O(n^2)" else MUTE)) for n,_,_ in curves])
        led.arrange(DOWN,buff=0.3,aligned_edge=LEFT)
        led.move_to(RIGHT*4.3+UP*0.3); _clamp(led)
        self.play(FadeIn(led),run_time=0.7)
        self.wait(1.0)
        ax2=Axes(x_range=[0,4,1],y_range=[0,4,1],x_length=6.0,y_length=3.4,
                 axis_config={"color":INK,"stroke_width":2,"include_tip":False})
        ax2.move_to(ax.get_center())
        quad=ax2.plot(lambda x: 0.28*x*x, x_range=[0,4.0], color=DEEP, stroke_width=4)
        self.play(Transform(ax,ax2),Transform(graphs[4],quad),run_time=1.4,rate_func=rush_into)
        note=_t("quadratic tears away",size=28,color=DEEP)
        note.next_to(ax,DOWN,buff=0.5); _clamp(note)
        self.play(FadeIn(note),run_time=0.7)
        self.wait(2.9)
        check_overlaps(ax,led,note,label="B05")


class B08_Receipts(Scene):
    """31.1s — two bars grow with a bracket; the caveat fades in below."""
    def construct(self):
        _rule(self)
        base=LEFT*3.6+DOWN*0.6
        hw=Rectangle(width=1.0,height=0.6,color=INK,stroke_width=2,
                     fill_color=INK,fill_opacity=0.45).move_to(base+UP*0.9)
        hw.stretch_to_fit_width(0.6,about_point=hw.get_left())
        alg=Rectangle(width=1.0,height=0.6,color=INK,stroke_width=2,
                      fill_color=DEEP,fill_opacity=0.55).move_to(base+DOWN*0.0)
        alg.stretch_to_fit_width(0.6,about_point=alg.get_left())
        hwl=_t("hardware",size=26,color=MUTE).next_to(hw,LEFT,buff=0.3)
        algl=_t("algorithms",size=26,color=MUTE).next_to(alg,LEFT,buff=0.3)
        self.play(FadeIn(hwl),FadeIn(algl),FadeIn(hw),FadeIn(alg),run_time=1.0)
        hwtag=_mono("~1,000x",size=26,color=INK).next_to(hw,RIGHT,buff=0.25)
        self.play(hw.animate.stretch_to_fit_width(2.4,about_point=hw.get_left()),
                   FadeIn(hwtag),run_time=1.3)
        hwtag.add_updater(lambda m: m.next_to(hw,RIGHT,buff=0.25))
        algtag=_mono("~43,000x",size=26,color=DEEP).next_to(alg,RIGHT,buff=0.25)
        self.play(alg.animate.stretch_to_fit_width(4.6,about_point=alg.get_left()),
                   FadeIn(algtag),run_time=1.6)
        algtag.add_updater(lambda m: m.next_to(alg,RIGHT,buff=0.25))
        self.wait(0.3)
        hwtag.clear_updaters(); algtag.clear_updaters()
        bracket=Brace(VGroup(hw,alg),RIGHT,color=INK)
        sumtag=VGroup(_t("82 years",size=26),_t("-> ~1 minute",size=26)).arrange(DOWN,buff=0.1,aligned_edge=LEFT)
        sumtag.next_to(bracket,RIGHT,buff=0.25); _clamp(sumtag)
        self.play(GrowFromCenter(bracket),FadeIn(sumtag),run_time=1.0)
        cite=_mono("PCAST 2010, p.71",size=22,color=MUTE)
        cite.move_to(LEFT*3.6+UP*2.3); _clamp(cite)
        self.play(FadeIn(cite),run_time=0.6)
        self.wait(1.2)
        caveat=VGroup(
            _t("one caveat:",size=26,color=DEEP),
            _t("the solver vendor's own numbers",size=26,color=DEEP),
            _t("differ for the same era.",size=26,color=DEEP),
        ).arrange(DOWN,buff=0.12,aligned_edge=LEFT)
        caveat.move_to(RIGHT*3.2+UP*1.7); _clamp(caveat)
        self.play(FadeIn(caveat,shift=UP*0.2),run_time=1.0)
        _foot(self,"order of magnitude, not a measurement")
        self.wait(6.0)
        check_overlaps(hwl,algl,hw,alg,bracket,sumtag,cite,caveat,label="B08")


class B09_UnevenProgress(Scene):
    """21.9s — 113 dots sort into three shaded bands."""
    def construct(self):
        _rule(self)
        n=113
        cols=15
        dots=VGroup(*[Dot(radius=0.05,color=GRAY) for _ in range(n)])
        for i,d in enumerate(dots):
            r,c=divmod(i,cols)
            d.move_to(LEFT*5.2+RIGHT*0.3*c+UP*(1.6-0.3*r))
        self.play(LaggedStart(*[FadeIn(d) for d in dots],lag_ratio=0.02),run_time=1.4)
        title=_t("113 algorithm families",size=28,color=MUTE)
        title.move_to(LEFT*3.3+UP*2.35); _clamp(title)
        self.play(FadeIn(title),run_time=0.6)
        half=dots[:63]; trans=dots[63:78]; moore=dots[78:]
        self.play(*[d.animate.set_color(MUTE).set_opacity(0.55) for d in half],run_time=1.0)
        self.play(*[d.animate.set_color(DEEP) for d in trans],run_time=1.0)
        self.play(*[d.animate.set_color(INK) for d in moore],run_time=1.0)
        rows=[("BARELY IMPROVED","~half",MUTE),("TRANSFORMATIVE","13%",DEEP),
              ("MATCHED MOORE'S LAW","30-45%",INK)]
        led=VGroup()
        for name,pct,col in rows:
            r=VGroup(_mono(name,size=22,color=col),_t(pct,size=28)).arrange(RIGHT,buff=0.4)
            led.add(r)
        led.arrange(DOWN,buff=0.42,aligned_edge=LEFT)
        led.move_to(RIGHT*2.15+DOWN*0.3); _clamp(led,lim=6.0)
        self.play(FadeIn(led),run_time=0.8)
        src=_mono("57 textbooks · 1,137+ papers",size=20,color=MUTE)
        src.next_to(led,DOWN,buff=0.55); _clamp(src,lim=6.0)
        self.play(FadeIn(src),run_time=0.6)
        _foot(self,"not evenly, and not on schedule")
        self.wait(3.5)
        check_overlaps(dots,title,led,src,label="B09")


class B11_MatMulWaterfall(Scene):
    """22.0s — four bars shrink; the multiplier ledger climbs to 60,000x+."""
    def construct(self):
        _rule(self)
        names=["PYTHON","JAVA","C","HARDWARE-\nTAILORED"]
        heights=[3.0,3.0/11,3.0/47,0.05]
        tags=["~7 hrs","11x","47x vs Python","<1 sec"]
        bars=VGroup()
        xs=[-4.4,-2.3,-0.2,1.9]
        for h,x in zip(heights,xs):
            b=Rectangle(width=1.1,height=max(h,0.06),color=INK,stroke_width=2,
                        fill_color=INK,fill_opacity=0.5)
            b.move_to(RIGHT*x+DOWN*(1.3-max(h,0.06)/2))
            bars.add(b)
        labels=VGroup(*[_mono(n,size=20,color=MUTE) for n in names])
        for l,b,x in zip(labels,bars,xs):
            l.move_to(RIGHT*x+DOWN*2.0); _clamp(l)
        self.play(FadeIn(bars[0]),FadeIn(labels[0]),run_time=0.7)
        cnt=Integer(1,color=INK,font_size=48)
        cntlbl=_t("x faster than Python",size=26,color=MUTE)
        anchor=RIGHT*1.55+UP*2.0
        cnt.move_to(anchor,aligned_edge=RIGHT)
        cntlbl.next_to(cnt,RIGHT,buff=0.25)
        cntlbl.add_updater(lambda m: m.next_to(cnt,RIGHT,buff=0.25))
        self.play(FadeIn(cnt),FadeIn(cntlbl),run_time=0.5)
        tagmobs=VGroup()
        for i in range(4):
            tg=_mono(tags[i],size=22,color=(DEEP if i==3 else INK))
            tg.next_to(bars[i],UP,buff=0.2); _clamp(tg)
            tagmobs.add(tg)
            if i>0:
                self.play(FadeIn(bars[i]),FadeIn(labels[i]),run_time=0.6)
            self.play(FadeIn(tagmobs[i]),run_time=0.4)
            newval=[1,11,47,63000][i]
            self.play(cnt.animate.set_value(newval).move_to(anchor,aligned_edge=RIGHT),run_time=0.7)
        cntlbl.clear_updaters()
        self.play(cnt.animate.set_color(DEEP),cntlbl.animate.set_color(DEEP),run_time=0.4)
        _foot(self,"more than sixty thousand times faster")
        self.wait(4.0)
        check_overlaps(bars,labels,tagmobs,cnt,cntlbl,label="B11")


class B14_ThreeMoves(Scene):
    """17.0s — three tiny code fragments, one per house move."""
    def construct(self):
        _rule(self)
        frags = [
            ("SEQUENCE", "sum an array", ["sum = 0", "sum += A[i]", "return sum"], "->"),
            ("SELECTION", "max of two", ["if a > b:", "  return a", "return b"], "fork"),
            ("ITERATION", "factorial", ["r = 1", "while i <= n:", "  r *= i"], "loop"),
        ]
        groups=VGroup()
        xs=[-4.2,0.0,4.2]
        for (label,sub,lines,icon),x in zip(frags,xs):
            code=VGroup(*[_mono(l,size=22,color=INK) for l in lines])
            code.arrange(DOWN,buff=0.14,aligned_edge=LEFT)
            box=SurroundingRectangle(code,color=INK,stroke_width=2,buff=0.26,corner_radius=0.1)
            box.set_fill(CREAM,opacity=1)
            lab=_mono(label,size=22,color=DEEP)
            subl=_t(sub,size=24,color=MUTE)
            grp=VGroup(lab,subl,VGroup(box,code)).arrange(DOWN,buff=0.2)
            grp.move_to(RIGHT*x); _clamp(grp)
            groups.add(grp)
        for g in groups:
            self.play(FadeIn(g),run_time=1.1)
        _foot(self,"three moves, every algorithm")
        self.wait(4.6)
        check_overlaps(groups,label="B14")


class B15_FrontierRipple(Scene):
    """16.2s — grid rings fill one full ring at a time; cross-reference note."""
    def construct(self):
        _rule(self)
        n=5
        cells=[[None]*n for _ in range(n)]
        grid=VGroup()
        for r in range(n):
            for c in range(n):
                cell=Rectangle(width=0.62,height=0.62,color=INK,stroke_width=1.6,
                               fill_color=CREAM,fill_opacity=1)
                cell.move_to(LEFT*2.9+RIGHT*0.66*c+UP*(1.3-0.66*r))
                cells[r][c]=cell; grid.add(cell)
        self.play(LaggedStart(*[FadeIn(c) for c in grid],lag_ratio=0.02),run_time=1.0)
        src=(2,2)
        rings={}
        for r in range(n):
            for c in range(n):
                d=abs(r-src[0])+abs(c-src[1])
                rings.setdefault(d,[]).append(cells[r][c])
        qtag=VGroup(_mono("QUEUE",size=24,color=MUTE),_t("FIFO",size=24,color=MUTE)).arrange(RIGHT,buff=0.3)
        qtag.move_to(RIGHT*4.0+UP*1.6); _clamp(qtag)
        self.play(FadeIn(qtag),run_time=0.5)
        for d in sorted(rings):
            fills=[c.copy().set_fill(TERRA if d>0 else DEEP,opacity=0.55) for c in rings[d]]
            self.play(*[FadeIn(f) for f in fills],run_time=0.55)
        note=_t("full proof: the BFS vs DFS film",size=24,color=DEEP)
        note.move_to(RIGHT*3.8+DOWN*0.6); _clamp(note)
        under=Line(note.get_left()+DOWN*0.2,note.get_right()+DOWN*0.2,color=DEEP,stroke_width=2)
        self.play(FadeIn(note),Create(under),run_time=0.8)
        _foot(self,"never skips a layer")
        self.wait(3.4)
        check_overlaps(grid,qtag,note,label="B15")


class B19_DijkstraTimeline(Scene):
    """17.8s — 1959 chip lands; the line holds flat; 2025 chip lands."""
    def construct(self):
        _rule(self)
        line=Line(LEFT*5.0+DOWN*0.2,RIGHT*5.0+DOWN*0.2,color=INK,stroke_width=2)
        self.play(Create(line),run_time=0.8)
        p1959=Dot(LEFT*4.6+DOWN*0.2,color=INK,radius=0.09)
        l1959=VGroup(_mono("1959",size=26,color=INK),_t("Dijkstra",size=28))
        l1959.arrange(DOWN,buff=0.1).next_to(p1959,UP,buff=0.3)
        b1=_mono("O(m + n log n)",size=22,color=MUTE).next_to(l1959,UP,buff=0.2)
        self.play(FadeIn(p1959),FadeIn(l1959),FadeIn(b1),run_time=1.0)
        dash=DashedLine(p1959.get_center(),RIGHT*3.6+DOWN*0.2,color=INK,stroke_width=1.5,dash_length=0.15)
        dash.set_stroke(opacity=0.4)
        self.play(Create(dash),run_time=1.6)
        p2025=Dot(RIGHT*3.9+DOWN*0.2,color=DEEP,radius=0.11)
        l2025=VGroup(_mono("2025",size=26,color=DEEP),_t("Duan et al.",size=28))
        l2025.arrange(DOWN,buff=0.1).next_to(p2025,UP,buff=0.3); _clamp(l2025)
        b2=_mono("NEW BOUND",size=22,color=DEEP).next_to(l2025,UP,buff=0.2); _clamp(b2)
        tag=_mono("STOC BEST PAPER",size=20,color=MUTE).next_to(p2025,DOWN,buff=0.3); _clamp(tag)
        self.play(FadeIn(p2025),FadeIn(l2025),FadeIn(b2),FadeIn(tag),run_time=1.1)
        _foot(self,"first time this bound moved in 60+ years")
        self.wait(4.6)
        check_overlaps(line,l1959,b1,l2025,b2,tag,label="B19")


class B20_ProofVsPractice(Scene):
    """18.3s — the proof checkmark lands; the benchmark race runs."""
    def construct(self):
        _rule(self)
        proof_box=RoundedRectangle(corner_radius=0.12,width=3.2,height=1.6,
                                   color=INK,stroke_width=2,fill_color=CREAM,fill_opacity=1)
        proof_box.move_to(LEFT*3.6+UP*0.6)
        check=_t("THE PROOF",size=28,color=INK)
        checkmark=Text("✓",color=DEEP,font_size=54)
        pgrp=VGroup(checkmark,check).arrange(DOWN,buff=0.2).move_to(proof_box)
        self.play(FadeIn(proof_box),FadeIn(pgrp),run_time=1.0)
        bound=_mono("beats Dijkstra's bound",size=22,color=MUTE)
        bound.next_to(proof_box,DOWN,buff=0.3); _clamp(bound)
        self.play(FadeIn(bound),run_time=0.6)
        track=Line(RIGHT*0.8+DOWN*0.6,RIGHT*5.6+DOWN*0.6,color=INK,stroke_width=2)
        finish=Line(RIGHT*5.6+DOWN*1.1,RIGHT*5.6+UP*0.1,color=INK,stroke_width=2)
        self.play(Create(track),Create(finish),run_time=0.7)
        d_chip=chip("DIJKSTRA",size=20); d_chip.move_to(RIGHT*0.8+DOWN*0.6)
        n_chip=chip("NEW ALGO",size=20); n_chip.move_to(RIGHT*0.8+UP*0.95)
        self.play(FadeIn(d_chip),FadeIn(n_chip),run_time=0.6)
        self.play(d_chip.animate.move_to(RIGHT*5.3+DOWN*0.6),
                   n_chip.animate.move_to(RIGHT*4.2+UP*0.95),run_time=1.6)
        wins=_t("WINS",size=30,color=DEEP)
        wins.next_to(d_chip,DOWN,buff=0.3); _clamp(wins)
        under=Line(wins.get_left()+DOWN*0.15,wins.get_right()+DOWN*0.15,color=DEEP,stroke_width=2.5)
        self.play(FadeIn(wins),Create(under),run_time=0.7)
        _foot(self,"real proof. real benchmark. two different answers.")
        self.wait(4.4)
        check_overlaps(proof_box,pgrp,bound,track,d_chip,n_chip,wins,label="B20")
