# random-quickselect — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-30).

## Key case and scope

Find the fourth-smallest value in the sorted array `[1,2,3,4,5,6,7,8,9]`. A deterministic last-element pivot repeatedly keeps almost the whole array, while a pivot chosen independently of the input usually cuts away substantial work. The single insight is that randomization removes the input designer's control over the pivot rank; it does not remove Quickselect's worst case.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| Quickselect partitions around a pivot and recurses only into the side containing rank k | `Divide_and_Conquer_Strategies/Divide_and_Conquer_Algorithms.md`, Quickselect | VERIFIED |
| After partition, the pivot is in its final sorted position | Same Quickselect partition description and pseudocode | VERIFIED |
| A random pivot index is selected uniformly from the current subarray | `Randomized_Algorithms/Randomized_Algorithms.md`, `randomized_partition` | VERIFIED |
| Quickselect has average-case O(n) time | `Randomized_Algorithms/Quiz_Questions_Randomized_Algorithms.md`, QuickSelect question | VERIFIED |
| Worst-case work is quadratic | `Divide_and_Conquer_Strategies/Divide_and_Conquer_Algorithms.md`, Quickselect complexity discussion | VERIFIED |

The displayed random pivot sequences are authored deterministic examples for an inspectable animation; they are not empirical probability estimates.

## Length derivation

One cold open, one hesitant hook, two complete concrete traces, one prediction hold, one additional random trace, one characteristics abstraction, counted-work generalization, payoff, verdict, Your Turn, and outro. This is a single-insight 2–4 minute film. No finished recurrence lands, so no equation tangent fires.

## Beat plan

| Beat | Role | Narration / action |
|---|---|---|
| B00 | COLD OPEN | Claude composer asks why the obvious fixed-pivot rule fails. |
| B0H | HOOK | Hesitant writer asks whether a sorted array should be easiest. |
| B01 | INSTANCE 1 | Last pivots 9, 8, 7, 6, 5 repeatedly retain nearly everything before finding 4. |
| B02 | INSTANCE 2 | Pivot 5 permanently lands; only `[1,2,3,4]` remains, then pivot 2 leaves `[3,4]`, and 4 lands. |
| B03 | PREDICT | Hold on pivot 6 with target rank four; viewer predicts which side survives. |
| B04 | INSTANCE 3 / TRANSFORM | Pivot 6 discards the right side, pivot 3 discards the left side, and 4 lands. |
| B05 | ABSTRACTION | Name the license: correct partition, one-sided rank recursion, and pivot rank independent of input arrangement. |
| B06 | COMPLEXITY | Compare counted inspected cells across the traces; balanced random pivots shrink the retained work, but unlucky pivots remain possible. |
| B07 | PAYOFF | Return to the sorted input: a fixed rule is adversary-controlled; a fresh random rank is not. |
| BVDT | VERDICT | Recap the invariant, the discarded side, and the expected—not guaranteed—speed. |
| BHTF | BOUNDARY / YOUR TURN | Trace a specified pivot tape and distinguish correctness from expected runtime. |
| BOUT | OUTRO | Title restatement and thanks. |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00/B0H pose the sorted-input paradox. |
| At least two moving instances before abstraction | PASS | B01 and B02 are full traces before B05; B04 adds a third. |
| Definition as endpoint | PASS | Randomization's structural license is named only at B05. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | Partition finality, one-sided rank recursion, and input-independent pivot rank are explicit. |
| Predict-before-step | PASS | B03 holds before discarding a partition. |
| Complexity last | PASS | B06 counts concrete inspected cells before naming average linear and worst quadratic. |
| Equation tangent | PASS / N.A. | No finished equation lands on screen. |
| Boundary and exercise | PASS | BHTF supplies a concrete trace and asks expected-vs-guaranteed. |
| Scope discipline | PASS | Probability proof, code, and deterministic median-of-medians are deferred. |

## Full Your Turn prompt

> Act as my Quickselect trace partner, not a lecturer. Use the array `[9,1,8,2,7,3,6,4,5]` and ask me to find the fourth-smallest element using this pivot-value tape in order: `7, 2, 4`. At each partition, make me write the less-than, pivot, and greater-than regions; state the pivot's final rank; adjust k when the retained side is on the right; and cross out the discarded side. Before revealing which side survives, ask me to predict it. Keep a running total of how many active elements were inspected. At the end, ask me to explain separately why the answer is always correct and why random pivots give expected linear rather than guaranteed linear time. Correct one mistake at a time without immediately giving the answer.

Deferred: the expected-time probability proof, implementation code, duplicate-key partition variants, and median-of-medians.
