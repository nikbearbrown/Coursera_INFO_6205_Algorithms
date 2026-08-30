# BFS vs DFS: The Frontier Is the Algorithm

This INFO 6205 explainer uses one unweighted graph to show why breadth-first search and depth-first search can visit the same nodes in radically different orders. The graph stays fixed while the frontier changes: a FIFO queue preserves distance layers, while a LIFO stack follows the current branch. The film names the conditions behind BFS's shortest-path guarantee—equal-cost edges and a FIFO frontier—and proves the two-layer invariant in plain language.

Module: `Graphs_and_Graph_Search_Algorithms`

## Your Turn

Trace both searches from S on the graph with edges S-A, S-B, A-C, B-T, C-E, and E-F. Visit neighbors alphabetically. Before every queue removal or stack pop, predict the next node and write the full frontier. Which traversal reaches T first, and why?

The module folder contains no `Quiz_Questions.md`, so this is an authored practice trace rather than a quoted module quiz.

## Paste this prompt into an AI assistant

```text
Act as my BFS-and-DFS trace partner, not a lecturer. Use the undirected graph with edges S-A, S-B, A-C, B-T, C-E, and E-F. Visit neighbors alphabetically. First run BFS with a FIFO queue, but stop before every removal and ask me which node comes out next. After I answer, show the visited set and the complete queue. Then reset and run iterative DFS with a LIFO stack, pushing neighbors in reverse alphabetical order so alphabetical order is visited first; again stop before every pop. Keep score. At the end, ask me which traversal first reaches T, ask me to state the BFS frontier invariant in my own words, and challenge any gap without revealing the answer immediately.
```
