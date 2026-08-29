# Gale–Shapley Stable Matching

This INFO 6205 film shows why a complete matching can still fail, then runs deferred acceptance proposal by proposal. Two concrete matchings and two full traces reveal the structural license behind Gale–Shapley: proposers move only downward through finite preference lists, while receivers only trade upward. A short contradiction argument then shows why no blocking pair can survive.

Module: `Stable_Matching_and_the_Gale-Shapley_Algorithm` (course book chapter and exercises: Stable Matching / Gale–Shapley Algorithm).

## Your Turn

Test the proposed matching `(M1,W2), (M2,W3), (M3,W1)` for blocking pairs, then reset and trace proposer-side deferred acceptance. Predict each receiver decision before it happens and explain why every rejection is permanent.

## Paste this full prompt into an AI assistant

```text
Act as my Gale–Shapley trace partner, not a lecturer. Use proposers M1: [W2, W3, W1], M2: [W3, W1, W2], M3: [W1, W2, W3], and receivers W1: [M1, M2, M3], W2: [M2, M3, M1], W3: [M3, M1, M2]. First test the proposed matching (M1,W2), (M2,W3), (M3,W1) for blocking pairs. Make me check one unmatched pair at a time and require both preference comparisons before deciding. Then reset and run proposer-side deferred acceptance. Before each receiver decision, ask me to predict HOLD, TRADE UP, or REJECT. Keep the proposal pointers and tentative engagements visible in a compact table. At the end, ask me to explain why a rejected proposer never needs to revisit that receiver, then challenge any gap without giving the answer immediately.
```

Scope boundary: this film does not cover many-to-one capacities, ties, incomplete lists, unequal sets, implementation details, or the full proposer-optimality theorem.
