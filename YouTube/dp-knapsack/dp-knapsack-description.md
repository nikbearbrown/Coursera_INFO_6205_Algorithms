# 0/1 Knapsack: When the Best-Looking Item Loses

This INFO 6205 explainer shows why value-density greedy fails for indivisible
items, then builds the 0/1 knapsack dynamic-programming table from concrete
take-or-skip decisions. It names the two structural licenses for DP—ordered
optimal substructure and permanence—and makes solved-exactly-once visible
before collapsing the full table to rolling-row storage.

Module: **Dynamic Programming**, Lesson 3 — Dynamic Programming for
Optimization. The worked instance comes from the course notebook: capacity
50 with items `(10,60)`, `(20,100)`, and `(30,120)`.

## Your Turn quiz

From `Dynamic_Programming/Quiz_Questions.md`, Question 4:

Which characteristics make a problem suitable for dynamic programming?

- Optimal substructure
- Overlapping sub-problems
- Local optimal choices
- Divide and conquer approach
- Sub-problem solutions are stored and reused

The video intentionally leaves the choices unanswered.

## Paste this prompt into Claude or ChatGPT

```text
I just watched a video on 0/1 knapsack. Act as my trace partner, not a
lecturer. Use capacity 50 and items A=(weight 10,value 60), B=(20,100),
C=(30,120). First make me predict what value-density greedy chooses and its
final value. Then build the dynamic-programming table one cell at a time,
stopping before every take-or-skip decision so I must predict it; tell me
right or wrong and keep score. Do not reveal the final chosen set until we
backtrack together. Finally change exactly one item value, ask me whether
greedy will now be optimal, and generate a small Python visualization of the
table filling row by row.
```

The implementation shown conceptually matches
`Introduction_to_Algorithms_Python_Notebooks/Ch_8_Dynammic_Programming.ipynb`.
