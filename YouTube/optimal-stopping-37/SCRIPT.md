# optimal-stopping-37 — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-30).

## Key case and scope

Four candidates arrive in random order and each rejection is permanent. Waiting to see all four gives perfect information but zero ability to hire; rejecting one candidate and then taking the next record wins in 11 of the 24 possible orders. The single insight is why optimal stopping needs a deliberate observation window: it creates a benchmark, then preserves enough unseen candidates to beat it.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| Decisions are immediate and the objective is to select the unique best candidate using relative ranks | `Optimal_Stopping/Optimal_Stopping.md:192-205` | VERIFIED |
| The classic policy rejects about `n/e` candidates, then accepts the next candidate better than every observed candidate | `Optimal_Stopping/Optimal_Stopping.md:207-230` | VERIFIED |
| For `n=4`, rejection counts `r=0,1,2,3` win in `6,11,10,6` of 24 permutations | Direct exhaustive enumeration of all 24 rank orders | VERIFIED |
| The large-`n` optimal observation fraction approaches `1/e`, about 36.8%, and the success probability approaches `1/e` | Direct standard secretary-problem calculation; consistent with the module's `n/e` rule | VERIFIED |
| Module later says `e·n` and uses `ceil(e·n)` | `Optimal_Stopping/Optimal_Stopping.md:1331-1369` | CONFLICT — mathematically impossible as an initial sample size; excluded from film |
| Scout-card quiz source | `Optimal_Stopping` folder | MISSING — no `Quiz_Questions.md`; Your Turn is an authored trace exercise |

## Length derivation

One hook, two complete four-candidate instances, one predict hold, two eight-candidate traces, one characteristics abstraction, one threshold-sweep payoff, verdict, Your Turn, and outro. This is a single-insight film, derived at roughly 2–4 minutes with no padding.

## Beat plan and narration

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer types: “See everyone first?” then crosses it out and asks, “How long should I look before I leap?” |
| B01 | INSTANCE 1 | “Four candidates arrive in random order, and every rejection is permanent. If you wait to see all four, you identify the best perfectly—one decision too late. Perfect information gives zero chance to hire.” |
| B02 | INSTANCE 2 | “Now reject only the first candidate as a benchmark, then take the next new record. Across all twenty-four orders, that rule hires the best eleven times. Looking briefly beats looking completely.” |
| B03 | PREDICT | “Try eight candidates ranked from one to eight, where eight is best. We observe the first three: four, seven, then two. Candidate four is a six. Would you stop?” |
| B04 | TRANSFORM / TRACE | “Do not stop: six fails to beat the benchmark seven. Candidate five is eight, a new record, so we leap and win. The observation window was not wasted; it manufactured the threshold.” |
| B05 | INSTANCE 3 | “Run the same rule on five, seven, two, six, eight, four, three, one. We reject six, accept eight, and win again. But if eight hid inside the first three, the benchmark would become unbeatable and we would lose.” |
| B06 | ABSTRACTION | “Now the license is visible. The order must be random, the total count known, choices irrevocable, ranks comparable, and the goal the single best. Under look-then-leap, we win exactly when the best arrives after the window and the best earlier rival lies inside it.” |
| B07 | PAYOFF | “Sweep the observation window. With four candidates, rejecting one wins eleven of twenty-four orders, the peak. As the crowd grows, the peak slides toward thirty-seven percent observed and thirty-seven percent success—not because thirty-seven is magic, but because information and opportunity balance there.” |
| BVDT | VERDICT | “The verdict: more information is not always more useful. Observe too little and the benchmark is weak; observe too long and the best may already be gone. Look, then leap.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. Use ranks three, one, six, two, eight, five, seven, four with an observation window of three. Predict every reject or accept before revealing the rule’s move. The full trace-partner prompt is in the description.” |
| BOUT | OUTRO | “The thirty-seven percent rule balances learning against the chance to act. Thanks for watching Optimal Stopping: Look, Then Leap.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00 asks how long to observe before acting. |
| At least two moving instances before abstraction | PASS | B01 and B02 fully compare policies; B04 and B05 add two visible traces before B06. |
| Definition is an endpoint | PASS | The policy's licensing characteristics and invariant are named only after the traces. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | B06 names random order, known `n`, irrevocability, relative comparability, and the unique-best objective. |
| Predict-before-step | PASS | B03 holds on candidate four before the accept/reject move. |
| Equations tangent | PASS / N.A. | No finished symbolic equation lands; the four-case counts and asymptotic percentage are plotted data, not a parked equation. |
| Boundary and exercise | PASS | BHTF gives an unanswered trace; the missing module quiz is disclosed. |
| Scope discipline | PASS | Formal asymptotic derivation, unknown horizons, ties, and payoff-maximization variants are deferred. |

## Full Your Turn prompt

> Act as my optimal-stopping trace partner, not a lecturer. Use candidate ranks [3, 1, 6, 2, 8, 5, 7, 4], where 8 is best, with an observation window of r = 3. Show one candidate at a time. Before revealing the policy's action, make me predict OBSERVE, REJECT, or ACCEPT and explain which earlier rank sets the benchmark. Keep the decisions irrevocable. After the trace, ask whether the policy selected the global best and make me state the exact look-then-leap winning condition in my own words. Then change only the arrival order so the best candidate falls inside the observation window, and ask me to predict why the same policy must fail. Do not give an answer until I commit.

Deferred: the harmonic-sum derivation of the asymptotic optimum, unknown numbers of arrivals, ties, recall, non-random order, and objectives based on expected value rather than selecting the unique best.
