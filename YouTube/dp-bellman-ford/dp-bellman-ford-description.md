# Bellman-Ford: The Road That Pays You

This video shows why a direct edge of weight 10 can lose to a longer route of total weight 5, and why that failure is a structural warning rather than bad luck. It traces Bellman-Ford on the course module’s six-vertex graph, reveals its dynamic-programming layers, names the two characteristics that license the DP formulation—ordered subproblems and permanent solved layers—and uses the extra pass to expose a negative cycle.

Module: INFO 6205, Module 6 — Dynamic Programming, Lesson 2: Bellman-Ford Algorithm.

## Your Turn — course quiz question (unanswered)

Which of the following algorithms use dynamic programming?

- QuickSort
- Floyd-Warshall algorithm
- Bellman-Ford algorithm
- Dijkstra's algorithm
- Longest Increasing Subsequence

## Ready-to-paste AI prompt

> I just watched a video on Bellman-Ford. Act as my trace partner, not a
> lecturer. Here is a directed graph: S→A 10, S→E 8, E→D 1, D→A −4,
> D→C −1, A→C 2, C→B −2. (1) Run Bellman-Ford from S but STOP before every
> relaxation and make me predict the new value first — tell me right/wrong,
> keep score. (2) Then let me change exactly one edge weight and predict
> what breaks before you re-run. (3) Then generate a small visualization
> (Python/matplotlib or an animation if you can) of the distance table
> evolving per round on MY modified graph.

Source: `Coursera_INFO_6205_Algorithms/Dynamic_Programming/Bellman_Ford.md` and `Quiz_Questions.md`.
