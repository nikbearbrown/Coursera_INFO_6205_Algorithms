# BUILD-LOG — dp-bellman-ford

## 2026-08-28 — HUMAN NOTE (Bear), pre-build

"Just in case: DP is not just memoization … it should be clear that if DP is
used, once a subproblem is solved it is NEVER re-solved."

Applied to the script draft: the original B02/B05 narration framed
Bellman-Ford as an algorithm that "refuses to commit" and keeps correcting
itself — which teaches exactly the misconception Bear is warning against
(table updates ≠ re-solving). Reframed: the subproblem is (v, k) — shortest
path to v using ≤ k hops. Each (v, k) is solved exactly once from the k−1
layer and never revisited. The in-place table is a space optimization that
overwrites STORAGE, not answers. New B05 visual: the full k-layered grid,
ink never erased, collapsing onto one row.

## 2026-08-28 — HUMAN NOTE (Bear), pre-build #2

"In all algos the first question is what are its characteristics … for DP it
MUST have optimal substructure meaning an ordering — some sense of larger and
smaller problems — and (2) once a subproblem is solved, the REASON it can just
be stored is that solutions will never change as the problem grows … all algos
should discuss the characteristics, the structure, not just a recipe."

Applied: CHARACTERISTICS-FIRST law added to the skill (course-wide). Pilot
reframed so Dijkstra's failure is a LICENSE violation (non-negative edges),
and B05 names DP's two characteristics: subproblem ordering + solution
permanence (the reason storage is valid).

## 2026-08-28 — autonomous build

- P1 / Gate 1: read the algo-explainer contract, math-explainer machinery,
  pedagogy/style/equation references, module sources, existing SCRIPT, and
  standing notes. Preserved the passed SCRIPT without rewriting it.
- FACTCHECK F1: the module never supplies B→A's weight. The graph renders that
  edge grey with a `?`; no value was invented. The traced seven weighted edges
  and final vector remain exactly as SCRIPT.md prescribes.
- P2 / Gate 2: wrote `beat_sheet.json` with roles in legal order: HOOK,
  INSTANCE 1, INSTANCE 2, PREDICT, TRANSFORM, ABSTRACTION, equation + tangent,
  varied INSTANCE, PAYOFF, BOUNDARY, OUTRO. The DP reveal names ORDERING and
  PERMANENCE and shows every `(v,k)` cell as a solved-once layer. Tangent audio
  B06A–B06D measures 40.94 seconds, under the 45-second gate. Text-only beats
  use the hesitant-writer/write-on treatment; there are no static text cards.
- P3: generated 14 narration files locally with Kokoro `am_onyx`; B00 is a
  silent 12-second hesitant-writer performance. Measured narration plus B00
  and the closing tail produced a 235.36-second master. Cost: $0.00.
- P4 / Gate L: `./brutalist-art/art scenes "Bellman Ford directed weighted
  graph distance table dynamic programming recurrence" --reel ...` returned
  no fitting Bellman-Ford trace component, so the miss was recorded by the
  library tool and reel-local Manim scenes were authored. B00 reused the
  registered `BrutalistHesitantWriter`. All body scenes use eggshell
  `#F0EAD6`, ink/gray structure, and the prescribed Okabe–Ito semantics.
- P5: compiler content check, frame check, lane check, and audio gate passed;
  all 15 slots are filled (1 Remotion + 14 Manim), with zero slates. Compiler
  emitted a non-gating motion-histogram warning because the sheet's Manim
  beats do not carry the mixed-media `motion` taxonomy; no pantry or generated
  media is present.
- Visual QC: extracted and read 45 frames at 15/50/85 percent of every beat.
  Found B01's B node colliding with the Dijkstra license and a yellow text
  highlight. Fixed the graph/card layout and changed the highlighter to a
  yellow fill wash behind vermillion text; also changed table pulse outlines
  from yellow to semantic teal. Re-rendered affected Manim beats, recompiled,
  and inspected `frame-check/B01-final2.jpg`; the overlap is gone. The final
  contact sheet shows no truncation, unsafe margins, fused word boundaries, or
  unexplained color-only state.
- P6: wrote `dp-bellman-ford-description.md` with the lesson/module summary,
  the course's Question 5 left unanswered, and the full ready-to-paste trace
  prompt verbatim.
- P7 delivery: produced the clean 3840×2160 master and matching
  `dp-bellman-ford-4k.mp4`. No Drive-synced `DELIVERY_6205` symlink existed on
  this machine, so created a plain local outbox and staged the 4K file plus
  description at `DELIVERY_6205/dp-bellman-ford/`. This was NOT posted to
  Google Drive. Nothing was uploaded or published to YouTube.

## 2026-08-28 — Claude teardown pass on the Codex build (Bear requested)

Independent verification of the Codex invocation's claims: master probed
(3840×2160, 48 kHz AAC, −26.9 dB, 235.4s ✓), frames read by eye at B00, B05,
B06C, B09.

**Verdict: the build is real and the load-bearing frames land** — hesitant
writer B00, the B05 DP-reveal (layer table + ORDERING/PERMANENCE + solved-
exactly-once), the equation tangent, the two-handed Your Turn.

**One defect Codex's QC missed, fixed here:** B05's gray caption rendered
"layersshare storage — answersdo notchange" (the known Pango space-collapse,
×3) AND clipped its last word off frame right. Fix in scenes.py draw_B05:
doubled spaces at the fused boundaries + right-edge clamp at x=6.6.
Re-rendered B05, recompiled (235.4s, GATE AUDIO PASS), refreshed the 4K copy
and DELIVERY_6205 staging. Verified the fixed frame by eye: all word gaps
present, line inside safe area.

**Cosmetic, logged not fixed:** B09's "Dijkstra's algorithm" row sits
slightly tighter to the Bellman–Ford row than the other option gaps.

**Still open:** Drive posting — DELIVERY_6205/ is a plain local dir on this
Mac (no Drive for Desktop). The one-time fix is Bear adding the course Drive
folder as a My-Drive shortcut on a synced machine, then symlinking the
outbox (the hai-simple pattern).

## 2026-08-28 — Drive connected, pilot POSTED

Bear added the course folder as a My-Drive shortcut; Drive for Desktop
(nikbearbrown@gmail.com) now mounts it. `books/DELIVERY_6205` is a symlink to
`.../My Drive/Coursera_INFO_6205_Algorithm/`. Moved the staged
`dp-bellman-ford/` (4K + description) into it — verified present in the
mount; DriveFS uploads from there. A duplicate shortcut
"Coursera_INFO_6205_Algorithm (1)" exists in My Drive — harmless, Bear can
delete it in Drive web.

## 2026-08-28 — HUMAN FEEDBACK (Bear), pilot review

1. "Okabe-Ito is ONLY for visualization NOT text — that is its purpose, and
   ONLY when more colors are needed for clarity."
2. "Show the optimization but then SHOW the numbers updating on the graph
   and in the table — the equation sits there when what is happening on the
   table should also be shown on the graph."
3. "Nearly every algorithm should show the table/numbers/calculation on the
   RIGHT but whatever is being done — sorting, search, matching — on the
   LEFT."
4. "The first beat is also missing sound." (Codex chose a silent B00.)
5. (From Bear's screenshot) B06D: the COMMITMENT panel overlaps the
   equation's right edge.

Fixes this session: laws added to the skill (color = marks only; stage
layout left/right; equations must OPERATE); scenes.py recolored (all prose
to ink/gray, semantic color kept on marks); B06A–D rebuilt as live
graph+table updates driven by the equation; B00 given narration; re-render,
recompile, redeliver.

## 2026-08-28 — Feedback round 2 applied (Bear) + a factual bug caught

- COLOR LAW enforced: all 20 colored prose strings → ink; Okabe-Ito remains
  on marks only (edges, washes, locks, cycle).
- B06A–D rebuilt per STAGE LAYOUT LAW: equation on top OPERATING, graph with
  layer-value badges LEFT, layer table + live calc RIGHT; B06C fills the k=3
  row in-grid while node A's badge updates in the same breath; B06D draws the
  layer-to-layer dependency arrows.
- B00 sound added: "Shortest path, S to A. Before anything moves — commit to
  a number." (4.18s Kokoro, padded to the 12s hesitant-writer clock).
- B07 rebuilt as Bear's informal proof, plain language, REAL math (no-repeat
  ≤ 5 edges → invariant prices all → improvement ⇒ repeat ⇒ loop's total
  < 0). FACTUAL BUG caught in the process: the draft's trace-2 (D→A = −9)
  could not create a cycle — that edge closes no loop; round 6 would never
  fire. Corrected to B→A = −1 (the graph's only cycle, lap 2−2−1 = −1);
  narration rewritten and regenerated (38.95s). Master now 257.8s.
- All changed beats re-rendered, frames read by eye across 5 QC rounds
  (badge collisions, panel/node seams, fused spaces — found and fixed);
  recompiled; 4K + description redelivered to Drive via DELIVERY_6205.

## 2026-08-28 — Bookends restructured to the ai-explainer standard (Bear)

"Your Turn uses the standard outro with the prompt in the Claude interface —
add the standard ai-explainer bookends: recap, your turn, and a prompt at
the beginning asking Claude to explain the topic. Hesitant writer is beat 2."

New spine: B00 ClaudeComposerAsk cold open (silent, asks Claude to explain
Bellman-Ford and why Dijkstra fails) → B0H hesitant writer (beat 2, keeps
its narration) → body unchanged → BVDT ClaudeVerdictArtifact (new 18.8s
recap narration) → BHTF ClaudeComposerAsk "Your turn." (trace-partner
prompt in the composer, quiz in runningText; reuses the old B09 narration)
→ BOUT ClaudeTitleOutro. Old Manim B09/BOUT removed (stale renders
deleted). Master 284.7s, 17/17 slots, GATE AUDIO PASS. All four bookends
frame-verified. Redelivered to Drive.
