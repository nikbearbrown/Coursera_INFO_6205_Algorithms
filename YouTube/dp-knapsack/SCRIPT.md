# dp-knapsack — algo-explainer script

Status: GATE 1 SELF-AUDIT PASSED — autonomous factory build.

## Scope and source

One insight: 0/1 knapsack needs dynamic programming because the locally best
value-per-weight item can block the globally best indivisible pair.

Key case: capacity 50 with items `(weight,value) = (10,60), (20,100),
(30,120)`. Density-greedy takes the first two for value 160; the best legal
set is the last two for value 220. This is the module/notebook instance.

Sources: `Dynamic_Programming/README.md` lines 369–438;
`Introduction_to_Algorithms_Python_Notebooks/Ch_8_Dynammic_Programming.ipynb`
knapsack cell; `Dynamic_Programming/Quiz_Questions.md` Question 4.

## Gate-1 audit

| Check | Result | Evidence |
|---|---|---|
| Key case named | PASS | Three notebook items, capacity 50; greedy 160 versus optimum 220. |
| Opens unsolved | PASS | Composer and hesitant writer ask whether density-greedy can be trusted. |
| Two moving instances before abstraction | PASS | Full greedy failure, then two row-by-row DP traces at capacities 20 and 50. |
| Definitions are endpoints | PASS | `dp[i][w]` is named only after both traces expose the repeated take/skip question. |
| Equation tangent present | PASS | Recurrence is followed immediately by zones 2–5 and re-entry. |
| Characteristics named | PASS | Ordered optimal substructure and permanence are explicit. |
| Mystery framing | PASS | The locally denser choice loses by 60. |
| Predict beat | PASS | Viewer commits at the decisive `(item 3, capacity 50)` cell. |
| Boundary + exercise | PASS | Module Question 4 and a fresh trace handoff close the reel. |

Length derivation: one hook, two concrete traces, one prediction, one DP
abstraction, recurrence plus tangent, payoff, verdict, handoff, outro.
Expected 4–6 minutes: **Standard** tier; no padding.

## Beat plan

| Beat | Role | Action |
|---|---|---|
| B00 | COLD OPEN | Claude composer asks why density-greedy loses. |
| B0H | HOOK | Hesitant writer types `take the densest first… right?` |
| B01 | INSTANCE 1 | Three item cards enter; density-greedy packs A+B = 160. |
| B02 | INSTANCE 1 | B+C replace them and total 220; greedy license is marked absent. |
| B03 | INSTANCE 2 | Full immutable DP rows fill for capacities 0–20. |
| B04 | INSTANCE 2 | Same process extends through capacity 50. |
| B05 | PREDICT | At item C, capacity 50: keep 160 or take 120+100? Hold, then 220. |
| B06 | ABSTRACTION | Every `(i,w)` question is solved once; ordering and permanence are named; grid collapses to rolling storage. |
| B07 | ABSTRACTION | The take/skip recurrence lands and operates on item C plus the table. |
| B07A | TANGENT | LHS/RHS and equality as a claim. |
| B07B | TANGENT | Symbol roles and domains. |
| B07C | TANGENT | Worked cell: max(160,100+120)=220. |
| B07D | TANGENT | Ordering and permanence commitment; return to the bag. |
| B08 | PAYOFF | Greedy 160 yields to optimal B+C = 220; complexity counted from grid cells. |
| BVDT | VERDICT | Claude artifact recap. |
| BHTF | BOUNDARY/YOUR TURN | Module Question 4 plus full trace-partner prompt. |
| BOUT | OUTRO | Title re-read. |

## FACTCHECK

| ID | Claim | Status | Anchor |
|---|---|---|---|
| F1 | Notebook example uses values 60,100,120; weights 10,20,30; capacity 50. | VERIFIED | Ch_8 notebook knapsack cell. |
| F2 | DP cell means best value using first i items and capacity j. | VERIFIED | Dynamic_Programming/README.md 403–415. |
| F3 | Recurrence is keep versus take from the previous item row. | VERIFIED | Dynamic_Programming/README.md 410–415 and notebook code. |
| F4 | Time and full-table space are O(nW). | VERIFIED | Dynamic_Programming/README.md 433–438. |
| F5 | Density-greedy result 160 and optimum 220. | DERIVED | Direct arithmetic from verified notebook instance; recomputed in the trace. |

## Full Your Turn prompt

> I just watched a video on 0/1 knapsack. Act as my trace partner, not a
> lecturer. Use capacity 50 and items A=(weight 10,value 60), B=(20,100),
> C=(30,120). First make me predict what value-density greedy chooses and its
> final value. Then build the dynamic-programming table one cell at a time,
> stopping before every take-or-skip decision so I must predict it; tell me
> right or wrong and keep score. Do not reveal the final chosen set until we
> backtrack together. Finally change exactly one item value, ask me whether
> greedy will now be optimal, and generate a small Python visualization of the
> table filling row by row.

## Deferred

Fractional and bounded knapsack, formal NP-hardness, approximation schemes,
and reconstruction code are separate lessons.
