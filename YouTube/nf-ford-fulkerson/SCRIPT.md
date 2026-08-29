# nf-ford-fulkerson — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-28).

## Key case and scope

A four-node network has maximum flow two, but choosing the tempting middle path first sends only one unit and appears to block every forward route. The single insight is that Ford–Fulkerson is safe because the residual graph makes earlier path choices reversible.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| Ford–Fulkerson repeatedly augments along source-to-sink residual paths | `Network_Flow/Network_Flow.md` | VERIFIED |
| Forward residual capacity is capacity minus flow | `Network_Flow/Network_Flow.md` | VERIFIED |
| A reverse residual edge has capacity equal to the current forward flow | `Network_Flow/Network_Flow.md` | VERIFIED |
| The module example reports a maximum flow of 11 | `Network_Flow/Network_Flow.md` | VERIFIED, NOT USED |
| Quiz asks the purpose of Ford–Fulkerson | `Network_Flow/Quiz_Questions_Network_Flow.md`, Question 6 | VERIFIED |

The four-node adversary is an authored demonstration. Capacities are shown explicitly; no missing module value is invented.

## Length derivation

One hook, two concrete path traces, a prediction hold, a residual undo trace, characteristics abstraction, invariant proof, payoff, verdict, Your Turn, and outro. This is a single-insight 2–4 minute film. No finished equation lands, so no equation tangent fires.

## Beat plan

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer asks why a legal path can trap a greedy flow. |
| B01 | INSTANCE 1 | “This network can carry two units from s to t. But choose s–a–b–t first, and its bottleneck sends one. Every visible forward route now looks blocked.” |
| B02 | INSTANCE 2 | “Reset. Send one along s–a–t, then one along s–b–t. The same capacities deliver two. So legal augmenting paths are not equally helpful in the moment.” |
| B03 | PREDICT | “Return to the bad first path. Before we quit at flow one, predict what edge the residual graph must add if the earlier choice is allowed to change.” |
| B04 | TRACE 1 | “The used edge a–b creates a backward residual edge b–a with capacity one. That edge does not send negative cargo; it cancels one unit of an earlier commitment.” |
| B05 | TRACE 2 | “Now the residual path s–b–a–t uses that backward edge. Augmenting once removes flow from a–b and reroutes it onto the two outer paths. Total flow becomes two.” |
| B06 | ABSTRACTION | “The license is reversibility through residual edges. Forward capacity records what we may still add; backward capacity records what we may undo. Integrality makes each augmentation visible in whole units here.” |
| B07 | PROOF | “Each augmentation preserves capacity limits and flow conservation. If no residual s-to-t path remains, the vertices still reachable from s define a cut with every crossing edge full, so the flow equals that cut’s capacity and cannot be improved.” |
| B08 | PAYOFF | “Back at the opening trap, greedy path choice was never promised to be locally wise. Ford–Fulkerson survives because residual edges can repair it before the no-path stopping test.” |
| BVDT | VERDICT | “The verdict: an augmenting path is a reversible commitment. Forward edges add room; backward edges return room. Stop only when the residual graph has no source-to-sink path.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. Trace the bad path, construct every residual edge, then find the repair path and explain why the final no-path condition certifies maximum flow.” |
| BOUT | OUTRO | “Ford–Fulkerson is not safe because every path is smart; it is safe because residual paths can undo mistakes. Thanks for watching.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00/B01 show the greedy trap. |
| At least two moving instances before abstraction | PASS | B01 and B02 run distinct full path choices; B04/B05 run the undo. |
| Definition as endpoint | PASS | Residual reversibility is named only at B06. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | Reversibility and integrality are explicit at B06. |
| Predict-before-step | PASS | B03 holds before revealing b–a. |
| Plain-language proof | PASS | B07 gives the reachable-set cut certificate. |
| Equation tangent | PASS / N.A. | No finished equation lands. |
| Boundary and exercise | PASS | BHTF traces residual construction and stopping certificate. |
| Scope discipline | PASS | Irrational-capacity termination, Edmonds–Karp, and implementation are deferred. |

## Full Your Turn prompt

> Act as my Ford–Fulkerson trace partner, not a lecturer. Use vertices s, a, b, t and directed capacities s→a=1, s→b=1, a→b=1, a→t=1, b→t=1. First make me augment along s→a→b→t. Keep a table with flow/capacity on every original edge and both forward and backward residual capacities. Before revealing any new residual edge, ask me to predict its direction and capacity. Then make me find a residual s-to-t path that repairs the first choice and reaches flow two. Finally, ask me to identify the vertices reachable from s in the final residual graph, compute the crossing cut capacity, and explain in my own words why equality with the flow certifies optimality. Challenge mistakes one at a time without immediately giving the answer.

Deferred: irrational-capacity nontermination, Edmonds–Karp path selection, runtime analysis, and code.
