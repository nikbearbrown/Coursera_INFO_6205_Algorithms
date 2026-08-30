# Randomized Quickselect: The Input Cannot Aim at a Coin

This INFO 6205 film shows why a fixed pivot rule can make sorted input adversarial for Quickselect, then traces how random pivots change the performance story without changing correctness. The visual invariant is that partition permanently places the pivot in its final sorted position, so the requested rank identifies exactly one side to retain. Concrete inspection counts distinguish expected linear time from a worst case that remains quadratic.

Module: **Randomized Algorithms**, with the Quickselect procedure from **Divide and Conquer Strategies**.

## Your Turn

Trace Quickselect on `[9,1,8,2,7,3,6,4,5]` for the fourth-smallest element using pivot values `7, 2, 4`. At every partition, predict which side survives, update the local rank when necessary, and count inspected active elements. Then explain why correctness is guaranteed while linear runtime is only expected.

## Paste-able AI prompt

```text
Act as my Quickselect trace partner, not a lecturer. Use the array `[9,1,8,2,7,3,6,4,5]` and ask me to find the fourth-smallest element using this pivot-value tape in order: `7, 2, 4`. At each partition, make me write the less-than, pivot, and greater-than regions; state the pivot's final rank; adjust k when the retained side is on the right; and cross out the discarded side. Before revealing which side survives, ask me to predict it. Keep a running total of how many active elements were inspected. At the end, ask me to explain separately why the answer is always correct and why random pivots give expected linear rather than guaranteed linear time. Correct one mistake at a time without immediately giving the answer.
```

Deferred: the expected-time probability proof, implementation code, duplicate-key partition variants, and median-of-medians.
