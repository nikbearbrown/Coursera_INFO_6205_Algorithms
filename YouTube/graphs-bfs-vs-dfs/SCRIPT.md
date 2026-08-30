# graphs-bfs-vs-dfs — production script

Status: GATE 1 SELF-AUDIT PASSED (2026-08-30).

## Key case and scope

On the same unweighted graph and start node, a target is two edges away while a tempting branch is four edges deep. DFS reaches the deep leaf first; BFS reaches the nearby target first. The single insight is that the frontier discipline creates the guarantee: FIFO preserves distance layers, while LIFO preserves a current path but offers no shortest-path guarantee.

## Factcheck

| Claim | Source | Status |
|---|---|---|
| BFS explores level by level using a queue | `Graphs_and_Graph_Search_Algorithms/Graph_Algorithms.md:617-660` and `README.md:617-660` | VERIFIED |
| BFS finds shortest paths in unweighted graphs | `Graphs_and_Graph_Search_Algorithms/README.md:619-624` | VERIFIED |
| DFS explores a branch deeply using a stack or recursion | `Graphs_and_Graph_Search_Algorithms/Graph_Algorithms.md:704-742` and `README.md:704-742` | VERIFIED |
| Both traversals run in O(V+E) with adjacency lists | `Graphs_and_Graph_Search_Algorithms/Graph_Algorithms.md:649-660,744-753` | VERIFIED |
| Scout-card quiz source | module folder | MISSING — no `Quiz_Questions.md`; do not quote or invent a module quiz |

The labeled graph and deterministic alphabetical neighbor order below are authored demonstrations of the module's rules.

## Length derivation

One hook, two complete traces, one predict hold, one frontier transform, one characteristics abstraction, one payoff, verdict, Your Turn, and outro. This is a single-insight film, approximately 2–4 minutes at measured Kokoro pace; no padding.

## Beat plan and narration

| Beat | Role | Narration / action |
|---|---|---|
| B00 | HOOK | Silent hesitant writer types: “Search is search?” then “same graph, same start…” and asks “why does the frontier change the answer?” |
| B01 | INSTANCE 1 | “Start at S. Breadth first removes the oldest frontier node, A, then B. The target T waits only two edges away, so it is reached before either deep branch.” |
| B02 | INSTANCE 2 | “Reset the same graph. Depth first removes the newest node, A, then C, then E, then F. It commits to one branch while the nearby target still waits.” |
| B03 | PREDICT | “Freeze here: the frontier holds B and C. A queue removes B; a stack removes C. Which node reaches the target sooner? Commit before the containers move.” |
| B04 | TRANSFORM / TRACE | “Change only the container. FIFO pulls from the old end and visits B, then T. LIFO pulls from the new end and follows C, E, F before returning.” |
| B05 | INSTANCE 3 | “Try a wider graph. Breadth first empties every distance-one node before any distance-two node. Depth first still follows whichever branch was pushed last.” |
| B06 | ABSTRACTION | “Now the licensing characteristic is visible: unweighted edges make each step equal, and FIFO keeps the frontier in just two adjacent distance layers, d and d plus one.” |
| B07 | INVARIANT / PROOF | “Why can no shorter route appear later? Every node entering layer d plus one came from layer d, and the queue finishes layer d before removing anything deeper.” |
| B08 | PAYOFF | “Return to the opening graph. BFS reaches T after S, A, B; DFS reaches it after S, A, C, E, F, B. Same graph, different frontier, different promise.” |
| BVDT | VERDICT | “The verdict: queue means breadth and shortest unweighted distance. Stack means depth and a current path. Both traverse; only FIFO protects layers.” |
| BHTF | BOUNDARY / YOUR TURN | “Your turn. Trace both searches from S with alphabetical neighbors. Predict every removal and record the frontier after each step. The full trace-partner prompt is in the description.” |
| BOUT | OUTRO | “The frontier is the algorithm. Thanks for watching BFS versus DFS.” |

## Gate-1 audit

| Check | Pass | Evidence |
|---|---|---|
| Key case named; opens unsolved | PASS | B00 asks why one container changes target order. |
| At least two moving instances before abstraction | PASS | Full BFS and DFS traces plus a container comparison precede B06. |
| Definition is an endpoint | PASS | FIFO layer invariant is named only after the traces. |
| Mystery framing | PASS | No utility preamble. |
| Characteristics named | PASS | B06 names unweighted equal-cost edges and FIFO layer preservation. |
| Predict-before-step | PASS | B03 holds with B and C visible in the frontier. |
| Equations tangent | PASS / N.A. | `d` and `d+1` are layer labels, not a landed equation. |
| Boundary and exercise | PASS | BHTF gives a concrete unanswered trace; missing module quiz is disclosed. |
| Scope discipline | PASS | Weighted shortest paths, recursive implementation, and applications are deferred. |

## Full Your Turn prompt

> Act as my BFS-and-DFS trace partner, not a lecturer. Use the undirected graph with edges S-A, S-B, A-C, B-T, C-E, and E-F. Visit neighbors alphabetically. First run BFS with a FIFO queue, but stop before every removal and ask me which node comes out next. After I answer, show the visited set and the complete queue. Then reset and run iterative DFS with a LIFO stack, pushing neighbors in reverse alphabetical order so alphabetical order is visited first; again stop before every pop. Keep score. At the end, ask me which traversal first reaches T, ask me to state the BFS frontier invariant in my own words, and challenge any gap without revealing the answer immediately.

Deferred: weighted shortest paths, recursive DFS implementation details, cycle detection, and topological sorting.
