# dp-bellman-ford — PILOT script draft (algo-explainer Phase 2)
Skill: brutalist-art/skills/make/algo-explainer/SKILL.md · Drafted 2026-08-28.
Status: GATE 1 SELF-AUDIT PASSED — awaiting build (P3 beat sheet onward).

## Gate-1 audit table (pedagogy.md §8)

| Check | Status | Evidence |
|---|---|---|
| Key exercise named | ✅ | The module's own 6-vertex graph (Bellman_Ford.md): shortest S→A. The confident wrong answer: 10. The truth: 5, through an edge that SUBTRACTS. |
| Opens with it, unsolved | ✅ | Hesitant-writer B00 types: "shortest path from S to A?" then types "10 — obviously" … hesitates … deletes "obviously". No vocabulary before the problem is felt. |
| ≥2 concrete instances moving | ✅ | Trace 1: the module's exact 5-iteration run (numbers verbatim from Bellman_Ford.md). Trace 2: same graph, one weight changed (B→A set to −1, closing the graph's ONLY cycle A→C→B→A at total −1) so iteration 6 CATCHES a negative cycle. [CORRECTED 2026-08-28: the draft's D→A→−9 was WRONG — that edge closes no loop, round 6 would never fire.] |
| Definitions are endpoints | ✅ | "Relaxation" named only after the viewer has watched three updates happen; the invariant stated only after both traces. |
| No premature completeness | ✅ | Deferred: Dijkstra's correctness conditions, SPFA, Johnson's algorithm, path reconstruction detail (list below). |
| Mystery framing | ✅ | Gap between what greedy predicts (freeze A at 10) and what the graph pays (5). Utility framing nowhere in the opener. |
| Characteristics named (law 0) | ✅ | Dijkstra's license: non-negative edges — violated by this graph (B01/B08). DP's license: ordered subproblems + permanence of solutions (B05). Structure, not recipe. |

**Length derivation (pedagogy §5):** HOOK 25s · greedy-fail INSTANCE 45s ·
TRACE-1 (5 iterations, module numbers) 90s · PREDICT beat inside trace-1 15s ·
INVARIANT/DP-REVEAL (grid collapse) 50s · recurrence + TANGENT 45s · TRACE-2
(negative-cycle variant) 45s · PAYOFF (hook resolved, path lit) 25s ·
YOUR TURN + outro 40s (total re-derived after DP-reveal grew B05) → **≈ 6:20**. Length is an output; do not pad or trim
to a round number.

## The graph (FACTCHECK anchors — module file is the source)

Vertices S,A,B,C,D,E. Edges used by the trace, weights verbatim:
S→A 10 · S→E 8 · E→D 1 · D→A −4 · D→C −1 · A→C 2 · C→B −2 · B→A (weight
NOT stated in Bellman_Ford.md — it appears only in the iteration-6 edge
check, line 107).
**FACTCHECK row F1:** B→A's weight must be resolved from the module figure or
Bear before the beat sheet locks; until then the animation draws the seven
weighted edges and shows B→A greyed with its check in iteration 6. Any value
≥ 0 is consistent with the trace's final distances.
Final distances (verbatim): S 0 · A 5 · B 5 · C 8 · D 9 · E 8.
Shortest S→A path: S→E→D→A = 8+1−4 = 5.

## Beat plan (roles per pedagogy §4)

| # | Role | What moves | Palette notes |
|---|---|---|---|
| B00 | HOOK | HesitantWriter: "Shortest path, S to A?" → "10 — obviously" → deletes "— obviously" → " …right?" | Claude cream skin (bookend) |
| B01 | INSTANCE | The graph appears (eggshell ground, gray edges). Greedy/Dijkstra runs: frontier commits S→A=10, LOCKS it (vermillion padlock). A small "license" card appears beside Dijkstra: 'all weights ≥ 0' — and the −4 edge stamps it VOID. Camera holds on the locked 10 while the E→D→A corridor glows faintly. | states: visited=blue, frontier=orange, locked=vermillion |
| B02 | INSTANCE/TRACE-1 | Bellman-Ford: the distance table (6 rows) beside the graph. Iterations 1–2 verbatim: A 10, E 8, then D 9, C 12. Edges flash teal as they relax. | table = the second protagonist |
| B03 | PREDICT | Freeze. "D is 9. The edge D→A weighs minus four. A currently says 10. Commit: does A change, and to what?" — held beat, then the relax: A → 5 (teal flash). | the video's one mandated commit |
| B04 | TRANSFORM | Iterations 3–5 verbatim to the final table (B: 10 → 6 → 5). The SAME table cells morph; no cuts. Then all six values settle black. | |
| B05 | ABSTRACTION | THE DP REVEAL (Bear's note, BUILD-LOG 2026-08-28): "distance to A" was never ONE question — it is six: d_k(A) for k=0…5. The full 6×6 grid draws, row per k, each cell written ONCE, ink never erased. Then the rows physically collapse onto a single row → the in-place table from the trace. The "changing" cell was different subproblems sharing one slot of storage. Invariant lands here: every (v,k) solved exactly once, never re-solved — THAT is dynamic programming, not memoization trivia. "Relaxation" named here, not earlier. | grid collapse is the TRANSFORM of the video |
| B06 | ABSTRACTION+TANGENT | The recurrence lands: d_k(v) = min(d_{k−1}(v), min_{(u,v)∈E} d_{k−1}(u)+w(u,v)) → equation tangent fires (equations.md 5-zone, ≤45s, explain-never-derive). | |
| B07 | INSTANCE/TRACE-2 | Same graph, B→A set to −1 (the only real cycle, lap cost 2−2−1=−1). Right panel: the informal proof in plain language, real math — no-repeat route ≤ 5 edges; invariant prices all of them in 5 rounds; round-6 improvement ⇒ a node repeated; cut the loop out, the rest is priced, so the gain is the loop's own total < 0. Cycle marked by its actual edges. | proof panel, stage law |
| B08 | PAYOFF | Hook resolved: the locked 10 shatters; S→E→D→A lights teal end-to-end = 5. Same objects as B01 — persistent, not redrawn. | |
| B09 | BOUNDARY/YOUR TURN | Two-handed: (a) quiz question from Dynamic_Programming/Quiz_Questions.md quoted unanswered; (b) AI prompt (short form on screen, full block in description). | |
| BOUT | OUTRO | your-turn skill closing: recap → title re-read. | Claude cream skin |

## Narration draft (Kokoro am_onyx; Teardown restraint, discovery register)

B00 (over hesitant writer): —silent, the typing is the narration—
B01: "Here's a graph. Six places, eight roads, and one question: shortest
route from S to A. There's a road that goes straight there. Ten. Any
sensible algorithm grabs it, locks it, moves on. Dijkstra — the fast one,
the famous one — does exactly that. Locked. Ten. And it is wrong.
Not broken — misapplied. Dijkstra comes with a license printed on the
box: every road must cost something. This graph has a road that PAYS.
The license is void, and the algorithm has no way to notice."
B02: "So here's an algorithm that asks a more careful question. Not
'how far is A?' — but 'how far is A if I'm allowed at most one hop? At
most two?' Round one answers every one-hop question: A ten, E eight.
Round two, every two-hop question: D nine, C twelve. Each answer is
final the moment it's written — for ITS question."
B03: "Round three. Look at D — nine. And that edge from D to A weighs
MINUS four. A's guess says ten. Before I run it: what happens to A?
Commit to an answer." (hold) "Nine minus four. Five. The confident ten
just lost to a detour through a road that pays YOU."
B04: "And the corrections cascade. C rides the new path down to eight.
B: twelve becomes ten becomes six becomes — round five — five. Watch
what's happening: each round, the truth reaches one hop further."
B05: "It LOOKS like the algorithm keeps changing its mind. It never
does. Lay the rounds out as their own rows and watch: every cell is
written once and never touched again. 'Distance to A' was six different
questions — at most zero hops, one, two — each solved exactly once from
the row above. That discipline has a name: dynamic programming. Not
'saving answers for later' — solving each subproblem ONCE, and never,
ever re-solving it. The table that seemed to flicker? Six rows sharing
one line of storage. And notice what made the whole scheme LEGAL —
two things this problem has that not every problem does. The questions
come in an order: smaller hop-budgets before bigger ones. And answers
are permanent: letting yourself take MORE hops can never change what
the best three-hop route was. No ordering, no DP. No permanence,
nothing safe to store. Five vertices minus one rounds, and no shortest
path needs more. The edge-by-edge step finally deserves its name:
relaxation."
B06: (tangent, over the recurrence) "Each new row asks every node: keep
the answer from the row above, or take a neighbor's row-above answer
plus one road. That min is the entire algorithm — every value built
purely from the previous layer, no cell ever recomputed. No priority
queue, no cleverness. One layer at a time, each layer final."
B07: "One catch. Make that helpful edge TOO helpful — minus nine — and
watch round six. It relaxed again. After five rounds nothing should
move. If it moves, you've built a loop that pays you every lap — a
negative cycle — and 'shortest path' stops meaning anything. The extra
round isn't waste. It's the smoke detector."
B08: "So: S to A. Not the road that says ten. Eight, plus one, minus
four. Five. The greedy answer wasn't unlucky — the graph broke the
rule Dijkstra's correctness depends on. That's the real first question
for ANY algorithm — not 'what are the steps?' but 'what must be true
of my problem for those steps to be legal?' Dijkstra asks for
non-negative roads. Bellman-Ford asks only for ordered questions and
answers that stay answered."
B09: "Your turn. [quiz question — verbatim from module]. And if you want
to actually feel it: the full prompt's in the description — paste it into
Claude or ChatGPT, hand it your own graph, and predict every relaxation
before it happens. Change one weight. Find the cycle."

## Your Turn — full prompt block (goes in dp-bellman-ford-description.md)

> I just watched a video on Bellman-Ford. Act as my trace partner, not a
> lecturer. Here is a directed graph: S→A 10, S→E 8, E→D 1, D→A −4,
> D→C −1, A→C 2, C→B −2. (1) Run Bellman-Ford from S but STOP before every
> relaxation and make me predict the new value first — tell me right/wrong,
> keep score. (2) Then let me change exactly one edge weight and predict
> what breaks before you re-run. (3) Then generate a small visualization
> (Python/matplotlib or an animation if you can) of the distance table
> evolving per round on MY modified graph.

## Deferred (pedagogy §1.5 — logged, not included)

Dijkstra's correctness conditions (own video) · path reconstruction detail ·
SPFA/queue optimizations · Johnson's algorithm · knapsack (card #5).
**Course-content gap, for Bear:** no Bellman-Ford code exists in
Introduction_to_Algorithms_Python_Notebooks/ (checked Ch_4, Ch_9) — the
README's Lesson 2 promises "Code and detect cycles with Bellman-Ford."
This video therefore ships WITHOUT a code beat (skill law: never freehand
code). If a notebook cell is added later, a code-endpoint beat can slot
after B06 in a revision.
