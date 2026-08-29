# greedy-interval-scheduling — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-28).

## Key case and scope

On one interval set, “start earliest” keeps one job while “finish earliest” keeps four. The single insight is why earliest finish has the greedy-choice property: replacing an optimal schedule’s first interval with the earliest-finishing interval never reduces the room left for all later choices.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| Objective is the maximum number of non-overlapping intervals | `Greedy_Algorithms/Greedy_Algorithms.md:375-392` | VERIFIED |
| Sort by non-decreasing finish time; accept compatible intervals | `Greedy_Algorithms/Greedy_Algorithms.md:394-429` | VERIFIED |
| Runtime is dominated by sorting | `Greedy_Algorithms/Greedy_Algorithms.md:1954-1986` and direct operation count | VERIFIED |
| Scout-card quiz source | `Greedy_Algorithms/Quiz_Questions.md` | MISSING — do not quote or invent; Your Turn is an authored trace exercise |

The concrete adversary sets below are authored demonstrations of the module’s stated rule, not attributed as module examples.

## Length derivation

One hook, two failing concrete strategies, one predict hold, one successful trace, one exchange transform, one characteristics abstraction, payoff, verdict, Your Turn, outro. At measured Kokoro pace this is a single-insight film, approximately 2–4 minutes; no padding.

## Beat plan and narration

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer types: “Pick the most meetings?” then tries “start earliest,” crosses it out, and asks “which meeting first?” |
| B01 | INSTANCE 1 | “Try the meeting that starts earliest. It spans one to nine, so every later candidate collides. One meeting survives. Starting early sounded productive; it consumed the entire day.” |
| B02 | INSTANCE 2 | “Maybe choose the shortest meeting. On this second day, the tiny middle meeting blocks two compatible meetings on either side. Shortest keeps one. The pair keeps two.” |
| B03 | PREDICT | “Back to the first day. Four short meetings finish at three, five, seven, and nine. The long meeting starts first. Before anything moves, choose the first interval you would keep.” |
| B04 | TRANSFORM / TRACE | “Now sort by finish time. Keep one to three. Reject the long overlap. Then keep three to five, five to seven, and seven to nine. Four meetings survive because each choice leaves the widest possible suffix.” |
| B05 | INSTANCE 3 | “Change every start time but preserve those finish choices. The same scan still keeps the four compatible intervals. The exact lengths did not license the choice; the remaining timeline did.” |
| B06 | ABSTRACTION | “That structure finally deserves its name: the greedy-choice property. Among compatible intervals, earliest finish leaves at least as much future room as any other first choice.” |
| B07 | TRANSFORM / PROOF | “Take any optimal schedule whose first interval is different. Swap that first interval for the earliest finisher. It ends no later, so every later interval still fits. The count cannot fall.” |
| B08 | PAYOFF | “The opening puzzle is settled: earliest start kept one; earliest finish kept four. The algorithm is sort once, then scan once—sorting dominates, so the runtime is O of n log n.” |
| BVDT | VERDICT | “The verdict: a local rule is not greedy magic. Earliest finish works because an exchange preserves every later choice; earliest start and shortest duration have no such guarantee.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. For intervals zero to six, one to four, three to five, five to seven, five to nine, and eight to ten, predict every keep or reject before the scan. The full trace-partner prompt is in the description.” |
| BOUT | OUTRO | “Earliest finish wins because it leaves the most future room. Thanks for watching Greedy Interval Scheduling.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00 asks which interval should go first. |
| At least two moving instances before abstraction | PASS | B01 earliest-start failure, B02 shortest-duration failure, B04/B05 successful traces precede B06. |
| Definition is an endpoint | PASS | “greedy-choice property” is named only in B06. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | B06 names exchangeability / future-room preservation; B07 proves it. |
| Predict-before-step | PASS | B03 holds before the first selection. |
| Equations tangent | PASS / N.A. | No finished equation lands; complexity appears as a terminal label, not an equation derivation. |
| Boundary and exercise | PASS | BHTF gives a concrete unanswered trace; absent module quiz is explicitly logged. |
| Scope discipline | PASS | Weighted scheduling, interval partitioning, and implementation details are deferred. |

## Full Your Turn prompt

> Act as my interval-scheduling trace partner, not a lecturer. Use these intervals: [0,6], [1,4], [3,5], [5,7], [5,9], [8,10]. First sort them by finish time, but stop before considering each interval and make me predict KEEP or REJECT. Tell me whether I am right, keep score, and maintain the current finish boundary. After the trace, ask me to construct one counterexample where “earliest start” fails and one where “shortest duration” fails. Finally, ask me to explain the exchange argument in my own words, then challenge any gap in my explanation without giving the answer immediately.

Deferred: weighted interval scheduling (dynamic programming), interval partitioning, tie-policy engineering details, and code walkthrough.
