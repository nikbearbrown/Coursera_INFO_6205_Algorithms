# dq-mergesort-recurrence — algo-explainer script

Status: GATE 1 SELF-AUDIT PASSED — autonomous factory build.

## Scope and source

One insight: halving creates logarithmically many levels, but merge sort still
does linear comparison work across every level, so its recurrence resolves to
`Theta(n log n)` rather than `Theta(log n)`.

Key case: sort `[8,3,6,2,7,1,5,4]`; count merge comparisons level by level.
The module supplies merge sort, its `2T(n/2)+O(n)` recurrence, and the
`O(n log n)` conclusion. Exact comparison counts below are derived on screen.

Source: `Divide_and_Conquer_Strategies/Divide_and_Conquer_Algorithms.md`,
sections “Merging and Merge Sort” and “Complexity Analysis.”

## Gate-1 audit

| Check | Result | Evidence |
|---|---|---|
| Key case named | PASS | Eight-item merge-sort comparison trace. |
| Opens unsolved | PASS | “If we halve repeatedly, why is the work not just logarithmic?” |
| Two moving instances before abstraction | PASS | Four-item and eight-item split/merge traces precede the recurrence. |
| Definitions are endpoints | PASS | Divide-and-conquer characteristics and recurrence are named after the traces. |
| Equation tangent present | PASS | `T(n)=2T(n/2)+n` immediately receives zones 2–5 and re-entry. |
| Characteristics named | PASS | Same-type independent halves, shrinking base case, linear combine. |
| Mystery framing | PASS | Logarithmic depth coexists with linear work per level. |
| Predict beat | PASS | Viewer predicts the eight-item level cost before merges run. |
| Boundary + exercise | PASS | A six-item recurrence/level-count prompt closes the reel. |

Length derivation: one hook, two concrete instances, prediction, one
abstraction, recurrence tangent, payoff, verdict, handoff, outro. Expected
3–5 minutes: **Single-insight** tier; no padding.

## Beat plan

| Beat | Role | Action |
|---|---|---|
| B00 | COLD OPEN | Claude composer asks why halving is not merely logarithmic work. |
| B0H | HOOK | Hesitant writer revises “halving means log time.” |
| B01 | INSTANCE 1 | Four values split to singletons, then merge; comparisons are counted. |
| B02 | INSTANCE 1 / TRANSFORM | The same four values reveal two merge levels and their costs. |
| B03 | INSTANCE 2 | Eight values unfold into a three-level recursion tree. |
| B04 | PREDICT / TRANSFORM | Commit to the cost of one eight-item merge level; then count it. |
| B05 | ABSTRACTION | The traces name the divide-and-conquer license and land the recurrence. |
| B05A–D | TANGENT | Claim, symbol roles, worked `n=8`, commitment and re-entry. |
| B06 | PAYOFF | Depth times level work resolves the opening: `Theta(n log n)`. |
| BVDT | VERDICT | Claude artifact recap. |
| BHTF | BOUNDARY / YOUR TURN | Six-item trace-partner prompt. |
| BOUT | OUTRO | Title re-read. |

## FACTCHECK

| ID | Claim | Status | Anchor |
|---|---|---|---|
| F1 | Merge sort recursively splits, sorts halves, and merges them. | VERIFIED | Module “Merge Sort Algorithm.” |
| F2 | Its recurrence is `T(n)=2T(n/2)+O(n)`. | VERIFIED | Module “Complexity Analysis.” |
| F3 | Solving the recurrence yields `O(n log n)`. | VERIFIED | Same module section. |
| F4 | Each balanced recursion level handles all `n` items during merging. | DERIVED | Partition accounting shown in both traces. |
| F5 | The concrete comparison totals shown are upper bounds, not exact for every input. | DERIVED | A merge of total length `m` uses at most `m-1` key comparisons. |

## Full Your Turn prompt

> I just watched a video about the merge-sort recurrence. Act as my trace
> partner, not a lecturer. Give me the array [9, 2, 8, 1, 7, 3]. Make me draw
> every split before you reveal it. Then make me merge bottom-up, asking me to
> predict each next comparison and keeping a comparison count per level. Do
> not state the recurrence until I have explained why one level touches all
> six items. Then help me write T(n)=2T(n/2)+Theta(n), identify what every
> symbol does, and ask me why logarithmic depth does not imply logarithmic
> total time. End by generating a small Python visualization of my recursion
> tree and level counts.

## Deferred

Master theorem cases, unbalanced splits, auxiliary-space analysis, stability,
and implementation code are separate lessons.
