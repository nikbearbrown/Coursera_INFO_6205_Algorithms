# Greedy Interval Scheduling: Finish First

This INFO 6205 course video shows why interval scheduling chooses the compatible interval that finishes earliest. Two tempting rules—earliest start and shortest duration—fail on concrete timelines. The successful trace then earns the greedy-choice property and a visual exchange argument: replacing an optimal schedule’s first interval with the earliest finisher cannot remove any later compatible interval.

Module: **Greedy Algorithms**, interval scheduling (`Greedy_Algorithms/Greedy_Algorithms.md`).

## Your Turn

For `[0,6], [1,4], [3,5], [5,7], [5,9], [8,10]`, sort by finish time and predict every KEEP or REJECT before the scan advances. The scout card referenced `Greedy_Algorithms/Quiz_Questions.md`, but that file is absent from the module, so this trace exercise is authored for the film rather than quoted from a course quiz.

## Paste-able AI trace-partner prompt

```text
Act as my interval-scheduling trace partner, not a lecturer. Use these intervals: [0,6], [1,4], [3,5], [5,7], [5,9], [8,10]. First sort them by finish time, but stop before considering each interval and make me predict KEEP or REJECT. Tell me whether I am right, keep score, and maintain the current finish boundary. After the trace, ask me to construct one counterexample where “earliest start” fails and one where “shortest duration” fails. Finally, ask me to explain the exchange argument in my own words, then challenge any gap in my explanation without giving the answer immediately.
```
