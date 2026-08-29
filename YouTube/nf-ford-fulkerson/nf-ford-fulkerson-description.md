# Ford–Fulkerson: The Edge That Undoes a Mistake

This INFO 6205 Network Flow video shows why a poor augmenting-path choice does not doom Ford–Fulkerson. Two concrete traces expose the apparent trap, then the residual graph’s backward edge visibly cancels and reroutes the earlier commitment. The final reachable-set cut explains why “no residual source-to-sink path” certifies maximum flow.

Module: `Network_Flow`

## Your Turn

Trace the bad path, construct every residual edge, find the repair path, and explain why the final no-path condition certifies maximum flow.

## Paste this full prompt into an AI assistant

```text
Act as my Ford–Fulkerson trace partner, not a lecturer. Use vertices s, a, b, t and directed capacities s→a=1, s→b=1, a→b=1, a→t=1, b→t=1. First make me augment along s→a→b→t. Keep a table with flow/capacity on every original edge and both forward and backward residual capacities. Before revealing any new residual edge, ask me to predict its direction and capacity. Then make me find a residual s-to-t path that repairs the first choice and reaches flow two. Finally, ask me to identify the vertices reachable from s in the final residual graph, compute the crossing cut capacity, and explain in my own words why equality with the flow certifies optimality. Challenge mistakes one at a time without immediately giving the answer.
```
