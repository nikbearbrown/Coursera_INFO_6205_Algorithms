# Merge Sort: Where the n log n Work Hides

This INFO 6205 video shows why repeatedly halving an input does not make merge
sort logarithmic-time. Two concrete array traces reveal the missing factor:
the recursion tree has logarithmically many levels, while merging touches all
`n` items at every level. The motion earns and then unpacks
`T(n) = 2T(n/2) + Θ(n)`, ending at `Θ(n log n)`.

Module: **Divide and Conquer Strategies — Mastering Merge Sort**

## Your Turn

Trace merge sort on `[9, 2, 8, 1, 7, 3]`. Count the merge work at every level,
then explain why logarithmic recursion depth does not imply logarithmic total
work.

## Paste this prompt into your AI assistant

```text
I just watched a video about the merge-sort recurrence. Act as my trace partner, not a lecturer. Give me the array [9, 2, 8, 1, 7, 3]. Make me draw every split before you reveal it. Then make me merge bottom-up, asking me to predict each next comparison and keeping a comparison count per level. Do not state the recurrence until I have explained why one level touches all six items. Then help me write T(n)=2T(n/2)+Theta(n), identify what every symbol does, and ask me why logarithmic depth does not imply logarithmic total time. End by generating a small Python visualization of my recursion tree and level counts.
```
