# INFO 6205 — algo-explainer scout cards
Generated 2026-08-28 by the algo-explainer scout lane (cards only — a human
selects before any build). Sources: module .md files under
`Coursera_INFO_6205_Algorithms/<Module>/`. Scored /10: adversary strength ·
Manim traceability · module centrality. Card format now includes
CHARACTERISTICS (skill law 0) — filled for the top 5; builders derive it
from the module text for the rest before scripting.

---

## 1. `dp-bellman-ford` — Score 10
**Characteristics:** DP license — ordered subproblems (hop budget k) + permanent solutions; adversary = Dijkstra's non-negative-weights license violated.
**Module:** Dynamic_Programming (`Bellman_Ford.md` — contains a COMPLETE worked
trace, Step 0 → Iteration 6 incl. negative-cycle check; `Dynamic_Programming.md`)
**Adversary instance:** Dijkstra's greedy on a graph with one negative edge —
it confidently commits to a shortest path that isn't. The module's own trace
graph works as-is.
**Trace:** the distance table filling per iteration, edges relaxing, one value
falling AFTER the greedy answer would have frozen it.
**Invariant:** after k rounds, every shortest path using ≤ k edges is final.
**Quiz hook:** Dynamic_Programming/Quiz_Questions.md (negative-cycle item).

## 2. `greedy-interval-scheduling` — Score 9
**Characteristics:** greedy-choice property (an exchange argument exists) — the adversary strategies fail because THEIR greedy choice lacks it.
**Module:** Greedy_Algorithms (Interval Scheduling + proof-of-optimality section)
**Adversary instance:** "pick the shortest job first" and "pick the earliest
start" both fail on concrete interval sets — the module's own limitation
discussion. Earliest-FINISH wins, which nobody guesses first.
**Trace:** intervals as bars on a timeline; three strategies race on the same
instance; exchange argument shown as literal swap of intervals.
**Invariant:** the greedy choice's finish time is ≤ any optimal schedule's.
**Quiz hook:** Greedy_Algorithms/Quiz_Questions.md.

## 3. `stable-matching-gale-shapley` — Score 9
**Characteristics:** monotone progress — proposals only move down preference lists, so termination + stability are structural.
**Module:** Stable_Matching_and_the_Gale-Shapley_Algorithm
**Adversary instance:** an "obvious" matching that looks fine until one
unstable pair quietly agrees to defect — instability is invisible until you
check every pair.
**Trace:** proposals/rejections round by round; tentative engagements breaking;
the same instance re-run with roles swapped (proposer advantage — the sting).
**Invariant:** once rejected by X, you can never do better than your next
choice below X — rejections only move down.
**Quiz hook:** module quiz.

## 4. `nf-ford-fulkerson` — Score 9
**Characteristics:** augmenting-path property + integrality — greedy path-picking is safe ONLY because residual edges make commitments reversible.
**Module:** Network_Flow (Ford-Fulkerson + residual graph + Python impl)
**Adversary instance:** the greedy path that BLOCKS the max flow — augmenting
down the middle edge first caps flow at 1 less than max until the residual
graph "undoes" it. Backward edges feel like cheating; they're the whole idea.
**Trace:** flow pushed along paths, residual graph updating beside the
original, an undo step visibly reversing an earlier commitment.
**Invariant:** flow value = capacity of some cut, always ≤ min cut.
**Quiz hook:** Network_Flow/Quiz_Questions.md.

## 5. `dp-knapsack` — Score 8
**Characteristics:** DP license — ordering (items × capacity) + permanence; greedy fails for lack of the greedy-choice property.
**Module:** Dynamic_Programming (0/1 Knapsack section)
**Adversary instance:** greedy-by-value-density fails on a concrete 3-item
instance (module's variant discussion).
**Trace:** the DP table filling row by row; the take/skip fork animated at
one decisive cell; backtracking the chosen set.
**Invariant:** cell (i, w) = best value using only the first i items in
capacity w.
**Quiz hook:** module quiz.

## 6. `dq-mergesort-recurrence` — Score 8
**Module:** Divide_and_Conquer_Strategies (Merge Sort + complexity section)
**Adversary instance:** "halving must mean log work" — but naive recursion on
Fibonacci also halves-ish and explodes; the gap is WHERE the work happens
(merge, not split).
**Trace:** the recursion tree unfolding; counted comparisons per level;
levels × work collapsing into n log n. Recurrence T(n)=2T(n/2)+n lands →
equation tangent fires.
**Invariant:** each level does ≤ n comparisons; there are ⌈log n⌉ levels.
**Quiz hook:** module quiz.

## 7. `sort-caching-lru` — Score 7
**Module:** Sorting_and_Caching ("Caching: Forget About It", eviction policies)
**Adversary instance:** LRU vs the clairvoyant optimum on a concrete access
string — LRU evicts exactly the wrong page on a looping pattern (thrash).
**Trace:** cache slots as boxes; the access string marching; hits green
(teal), evictions vermillion; same string under two policies.
**Invariant:** LRU's contents = the k most recently used distinct pages.
**Quiz hook:** module quiz.

## 8. `optimal-stopping-37` — Score 7
**Module:** Optimal_Stopping (Secretary Problem)
**Adversary instance:** "more information is always better" — waiting to see
everyone guarantees you can't go back; the concrete 4-candidate enumeration
shows early commitment beating full information.
**Trace:** candidate sequence dealt as cards; look-then-leap threshold
sweeping; success probability curve peaking at 1/e.
**Invariant:** under look-then-leap(r), you win iff the best is after r AND
the best-before-it is in the looked window.
**Quiz hook:** module quiz.

## 9. `graphs-bfs-vs-dfs` — Score 7
**Module:** Graphs_and_Graph_Search_Algorithms (BFS/DFS worked examples + code)
**Adversary instance:** same graph, same start, two traversals reach a target
in different order — "search is search" until the frontier data structure
(queue vs stack) is the only thing you change and everything changes.
**Trace:** frontier visualized as an actual queue/stack beside the graph;
Okabe-Ito states: unvisited / frontier / visited.
**Invariant:** BFS's frontier only ever holds nodes at distance d and d+1.
**Quiz hook:** module quiz.

## 10. `intractability-vertex-cover` — Score 6
**Module:** Intractability + Approximation_Algorithms (greedy vertex cover +
approximation ratio + TSP worst case)
**Adversary instance:** the greedy cover that's 2× optimal on a concrete
graph — and the proof that 2× is GUARANTEED, which flips "greedy failed"
into "greedy certified."
**Trace:** edges picked, both endpoints added, optimum shown beside it.
**Invariant:** every chosen edge forces ≥1 endpoint into ANY cover.
**Quiz hook:** Approximation_Algorithms/Quiz_Questions.md.

## 11. `random-quickselect` — Score 6
**Module:** Randomized_Algorithms (Randomized QuickSort/Quickselect) +
Divide_and_Conquer (Quickselect)
**Adversary instance:** deterministic pivot on sorted input → n²; the
adversary can always beat a fixed rule but cannot beat a coin.
**Trace:** pivot choices partitioning; the bad case animated; then 5 random
runs on the SAME adversarial input all finishing fast.
**Invariant:** after partition, the pivot is in its final sorted position.
**Quiz hook:** module quiz.

## 12. `bayes-free-throw` — Score 6
**Module:** Bayes_Rule (has a free-throw example already)
**Adversary instance:** base-rate neglect on the module's own numbers — the
"obvious" posterior read off the likelihood is wrong by a factor.
**Trace:** population squares re-weighting as evidence arrives; the module's
free-throw numbers computed on screen.
**Invariant:** posterior ∝ prior × likelihood, renormalized each step.
**Quiz hook:** module quiz.

## 13. `explore-exploit-bandit` — Score 6
**Module:** Explore_Exploit (Multi-Armed Bandit)
**Adversary instance:** always-exploit locks onto a worse arm forever on a
concrete 2-arm run — one unlucky early sample poisons the estimate.
**Trace:** two slot machines, running means, regret accumulating; ε-greedy
vs pure exploit on the same reward tape.
**Invariant:** under pure exploitation, an arm's estimate never updates once
abandoned.
**Quiz hook:** module quiz.

## 14. `scheduling-sjf` — Score 5
**Module:** Scheduling (FCFS vs SJF)
**Adversary instance:** FCFS on the module's job set — one long job first
wrecks average wait; the concrete count makes the abstraction (exchange
argument again) land.
**Trace:** jobs as blocks queueing on a timeline; total wait counted live;
swap animated.
**Invariant:** swapping any adjacent long-before-short pair never increases
total wait.
**Quiz hook:** module quiz.

---
Deferred (thin adversary or weak trace as videos): Linear_Programming
(graphical method traces well but simplex needs its own multi-video arc),
Social_Networks, Randomness (PRNG/LCG is traceable but low centrality),
Game_Theory (better as a two-video arc: best response → Nash), Overfitting/
Relaxation (course-adjacent, weak algorithm trace), LLM/GenAI/RL modules
(not algorithm-trace material for this format).
