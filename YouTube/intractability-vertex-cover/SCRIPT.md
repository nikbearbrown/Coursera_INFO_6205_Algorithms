# intractability-vertex-cover — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-30).

## Key case and scope

A four-leaf star has an optimum vertex cover of one center vertex. The arbitrary-edge two-approximation takes the center and one leaf, exactly twice optimum. The single insight is the certificate: selected edges form a matching, so every cover must pay at least one vertex per selected edge while the algorithm pays exactly two.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| A vertex cover touches every edge | `Approximation_Algorithms/Approximation_Algorithms.md` | VERIFIED |
| The algorithm picks an edge, adds both endpoints, and deletes incident edges | `Introduction_to_Algorithms_Python_Notebooks/Ch_11_Intractability.ipynb` | VERIFIED |
| The returned cover is at most twice optimum | `INFO6205_Assingments/Beary_Approximation_Algorithms_assignment.ipynb` | VERIFIED |
| Intractability can be extended with approximation algorithms | `Intractability/README.md`, Lesson 5 | VERIFIED |

The star and four-node path are authored demonstrations with every edge visible; no unresolved value is invented.

## Length derivation

One hook, two complete instances, one prediction hold, one certificate trace, characteristics, payoff, verdict, Your Turn, and outro. Single-insight tier, about two minutes after measured audio. No finished equation lands; the factor-two claim is explained through paired objects and a numeric ledger, so no equation tangent fires.

## Beat plan

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer asks how a visibly wasteful choice can still carry a guarantee. |
| B01 | INSTANCE 1 | “On this four-leaf star, one center vertex covers every edge. Now choose any edge and take both endpoints. We return two vertices where optimum needs one.” |
| B02 | INSTANCE 2 | “Try a four-vertex path. Choose the middle edge, take both endpoints, and every edge disappears. This time two vertices are also optimal.” |
| B03 | PREDICT | “The same rule was exact once and twice optimal once. Before the proof, predict what property the edges chosen by the algorithm must share.” |
| B04 | TRACE / TRANSFORM | “Run the star again. After choosing one edge, deleting every incident edge leaves nothing. On a larger graph, every later chosen edge is disjoint from the earlier ones.” |
| B05 | INSTANCE / CERTIFICATE | “Each chosen edge forces every possible cover to include at least one of its endpoints. Because chosen edges share no endpoints, those payments cannot be reused.” |
| B06 | ABSTRACTION | “That is the license: the chosen edges form a maximal matching, and their disjointness is a lower-bound certificate. The algorithm pays two vertices for every one vertex optimum must pay.” |
| B07 | PAYOFF | “Back at the star, the algorithm’s two vertices look wasteful, but the selected edge certifies optimum cannot use zero. Two is therefore the worst permitted multiple, not a guess.” |
| BVDT | VERDICT | “The verdict: approximation replaces an unreachable exact promise with a checkable bound. Disjoint chosen edges certify the lower bound; taking both endpoints certifies coverage.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. Trace the algorithm on a six-vertex graph, mark the chosen matching edges, then use them to prove both coverage and the factor-two guarantee.” |
| BOUT | OUTRO | “Vertex cover’s two-approximation is useful because every extra vertex is charged to a disjoint edge. Thanks for watching.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00/B01 use the star’s exact two-versus-one gap. |
| At least two moving instances before abstraction | PASS | B01 star and B02 path run completely. |
| Definition as endpoint | PASS | Maximal matching and certificate arrive at B06. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | Disjoint selected edges, maximal matching, coverage, lower bound. |
| Predict-before-step | PASS | B03 holds before disjointness is revealed. |
| Plain-language proof | PASS | B04–B06 show why payments cannot be reused. |
| Equation tangent | PASS / N.A. | No finished symbolic equation lands. |
| Boundary and exercise | PASS | BHTF hands off a six-vertex trace and proof. |
| Scope discipline | PASS | NP-hardness proof, weighted cover, and better special-case algorithms are deferred. |

## Full Your Turn prompt

> Act as my vertex-cover approximation trace partner, not a lecturer. Give me an undirected graph on six labeled vertices with at least seven edges. Ask me to choose an uncovered edge, add both endpoints to the cover, and delete every incident edge; repeat until no edges remain. Keep a visible ledger with the chosen edges, chosen vertices, and remaining edges. Before each step, ask me to predict whether the next chosen edge can share an endpoint with an earlier chosen edge. After the trace, make me explain why the chosen edges form a matching, why every vertex cover must contain at least one endpoint from each chosen edge, why those lower-bound payments cannot be reused, and why taking two endpoints per chosen edge gives a factor-two guarantee. Challenge one mistake at a time without immediately revealing the answer.

Deferred: the NP-hardness reduction, weighted vertex cover, bipartite exact algorithms, and implementation details.
