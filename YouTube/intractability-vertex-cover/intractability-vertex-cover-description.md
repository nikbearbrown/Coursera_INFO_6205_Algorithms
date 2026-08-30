# Vertex Cover: Why Twice Optimal Is a Promise

This video shows why the standard vertex-cover approximation can make a visibly wasteful choice and still guarantee a cover no larger than twice optimum. A four-leaf star exposes the exact two-versus-one gap; a path shows the same rule can be exact. The traces then earn the structural license: chosen edges form a maximal matching, every cover must pay at least one endpoint per chosen edge, and disjointness prevents reusing those payments.

Module: INFO 6205, Intractability (Lesson 5: Extending Tractability), with the Vertex Cover material from Approximation Algorithms.

## Your Turn

Trace the algorithm on a six-vertex graph, mark the chosen matching edges, then use them to prove both coverage and the factor-two guarantee.

## Paste this prompt into Claude

```text
Act as my vertex-cover approximation trace partner, not a lecturer. Give me an undirected graph on six labeled vertices with at least seven edges. Ask me to choose an uncovered edge, add both endpoints to the cover, and delete every incident edge; repeat until no edges remain. Keep a visible ledger with the chosen edges, chosen vertices, and remaining edges. Before each step, ask me to predict whether the next chosen edge can share an endpoint with an earlier chosen edge. After the trace, make me explain why the chosen edges form a matching, why every vertex cover must contain at least one endpoint from each chosen edge, why those lower-bound payments cannot be reused, and why taking two endpoints per chosen edge gives a factor-two guarantee. Challenge one mistake at a time without immediately revealing the answer.
```
