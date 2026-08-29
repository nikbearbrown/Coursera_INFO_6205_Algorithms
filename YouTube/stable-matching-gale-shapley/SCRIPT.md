# stable-matching-gale-shapley — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-28).

## Key case and scope

The opening assignment pairs A–X and B–Y, yet A and Y both prefer each other: a quiet blocking pair can overturn a complete-looking matching. The single insight is why deferred acceptance prevents such a pair: proposers move monotonically down finite preference lists, while receivers only trade upward.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| Stability means no unmatched pair mutually prefers each other | `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md:8427-8446` | VERIFIED |
| A free proposer approaches the next receiver; an engaged receiver keeps the preferred proposal | `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md:8484-8525` | VERIFIED |
| Gale–Shapley terminates with a stable matching | `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md:8520-8539` | VERIFIED |
| Proposer-optimality depends on which side proposes | `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md:8544-8570` | VERIFIED |
| Your Turn preference lists and blocking-pair task | `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md:8919-8928` | VERIFIED |

The two-person adversary and compact traces are authored demonstrations of these definitions; they are not attributed as module examples.

## Length derivation

One hook, two concrete moving instances, one prediction hold, two proposal traces, one characteristics abstraction, one plain-language stability proof, payoff, verdict, Your Turn, and outro. This is a single-insight film in the 2–4 minute tier; no equation lands and no tangent is required.

## Beat plan and narration

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer asks why a complete matching can still fall apart. |
| B01 | INSTANCE 1 | “Start with A paired to X and B paired to Y. Every seat is filled, yet A prefers Y, and Y prefers A. That unmatched pair would defect together, so complete does not mean stable.” |
| B02 | INSTANCE 2 | “Change just Y’s ranking so Y prefers B. The same drawn matching now has no mutual defection between A and Y. Stability depends on both preference lists, not on crossed or uncrossed lines.” |
| B03 | PREDICT | “Now three students propose. S one and S three both choose C one first. C one currently holds S one. Before the next step, predict whether that engagement is permanent.” |
| B04 | TRACE 1 | “It is only tentative. C one compares the new proposal, keeps S one, and rejects S three. S three moves down exactly one place to C three; rejected choices never return.” |
| B05 | TRACE 2 | “Run a second preference profile. C one first holds S one, then trades upward to S two. S one becomes free and proposes next to C two. Every rejection advances one pointer.” |
| B06 | ABSTRACTION | “Those traces reveal the license: progress is monotone. Proposers move only downward through finite lists; receivers keep only their best offer so far. No decision moves backward.” |
| B07 | PROOF | “Suppose the final matching had a blocking pair, A and Y. A must have proposed to Y before reaching a worse partner. Y rejected A only for someone Y preferred, and never traded downward. So Y cannot prefer A at the end—a contradiction.” |
| B08 | PAYOFF | “Return to the opening mismatch. Deferred acceptance cannot leave A and Y both wanting to defect. If A prefers Y, Y must already hold someone Y ranks above A. The hidden blocking pair is gone.” |
| BVDT | VERDICT | “The verdict: engagements are deferred, not promises. One side moves down; the other trades up. Those two monotone directions force termination and rule out every blocking pair.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. Test the module’s proposed matching for a blocking pair, then trace deferred acceptance and explain every rejection. The full trace-partner prompt is in the description.” |
| BOUT | OUTRO | “Stable does not mean everyone gets first choice; it means no unmatched pair would leave together. Thanks for watching Gale–Shapley Stable Matching.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00 asks why a complete matching can fail. |
| At least two moving instances before abstraction | PASS | B01 and B02 alter the same matching and preference state; B04 and B05 add two full traces before B06. |
| Definition is an endpoint | PASS | Monotone progress is named only after the proposal pointers move. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | B06 names finite one-way proposer pointers and receiver trade-up behavior. |
| Predict-before-step | PASS | B03 holds before a receiver decides. |
| Plain-language proof | PASS | B07 derives the contradiction from an earlier proposal and irreversible receiver improvement. |
| Equation tangent | PASS / N.A. | No finished equation lands. |
| Boundary and exercise | PASS | BHTF uses the book’s blocking-pair exercise. |
| Scope discipline | PASS | Many-to-one matching, ties, unequal sets, implementation, and full proposer-optimality proof are deferred. |

## Full Your Turn prompt

> Act as my Gale–Shapley trace partner, not a lecturer. Use proposers M1: [W2, W3, W1], M2: [W3, W1, W2], M3: [W1, W2, W3], and receivers W1: [M1, M2, M3], W2: [M2, M3, M1], W3: [M3, M1, M2]. First test the proposed matching (M1,W2), (M2,W3), (M3,W1) for blocking pairs. Make me check one unmatched pair at a time and require both preference comparisons before deciding. Then reset and run proposer-side deferred acceptance. Before each receiver decision, ask me to predict HOLD, TRADE UP, or REJECT. Keep the proposal pointers and tentative engagements visible in a compact table. At the end, ask me to explain why a rejected proposer never needs to revisit that receiver, then challenge any gap without giving the answer immediately.

Deferred: many-to-one capacities, ties or incomplete lists, unequal set sizes, implementation details, and the complete proposer-optimality theorem.
