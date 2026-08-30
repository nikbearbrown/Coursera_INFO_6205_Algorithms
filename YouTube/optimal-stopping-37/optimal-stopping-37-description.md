# Optimal Stopping: Look, Then Leap

This INFO 6205 video shows why “more information” can make a sequential decision worse. Using the classic secretary problem, it compares waiting for certainty with a look-then-leap policy, exhaustively counts all 24 orders for four candidates, traces two eight-candidate arrivals, and names the assumptions that license the 37% rule.

Module: `Optimal_Stopping`

## Your Turn

Use candidate ranks `[3, 1, 6, 2, 8, 5, 7, 4]`, where 8 is best, with an observation window of `r = 3`. Before each move, predict `OBSERVE`, `REJECT`, or `ACCEPT`. Track the benchmark, keep every decision irrevocable, and decide whether the policy selects the global best.

The module folder contains no `Quiz_Questions.md`, so this is an authored trace exercise rather than a quoted module quiz.

## Full paste-able AI prompt

```text
Act as my optimal-stopping trace partner, not a lecturer. Use candidate ranks [3, 1, 6, 2, 8, 5, 7, 4], where 8 is best, with an observation window of r = 3. Show one candidate at a time. Before revealing the policy's action, make me predict OBSERVE, REJECT, or ACCEPT and explain which earlier rank sets the benchmark. Keep the decisions irrevocable. After the trace, ask whether the policy selected the global best and make me state the exact look-then-leap winning condition in my own words. Then change only the arrival order so the best candidate falls inside the observation window, and ask me to predict why the same policy must fail. Do not give an answer until I commit.
```

Source note: `Optimal_Stopping/Optimal_Stopping.md` correctly states the initial observation size as approximately `n/e` in its main secretary-problem section. A later example incorrectly prints `e·n`; this video excludes that conflicting claim.
