# SOURCES — claude-liam-algorithmic-foundations

Course-intro film for INFO 6205 (Module 1, "The Importance of Algorithms").
Not one of the 11 `ai-algorithms` roster deepers — a standalone welcome/
motivation film, additive to the course's existing Module 1 material.

## Course text (this repo)

- `INFO_6205_Algorithms_and_Data_Structures_Book/INFO_6205_Algorithms.md`
  lines 47-260: "Understanding Algorithms" — the book's own
  finite/definite/effective properties (§Definition and Core Concepts,
  line 63), "The Importance of Studying Algorithms" (line 85: Software
  Development, Data Science, Networking, Finance), Sequences/Selections/
  Iterations (lines 111-168), Time Complexity list + Big-Omega/Big-Theta
  (lines 169-208).

## External sources (verified this session via WebSearch/WebFetch — see
## FACTCHECK.md for what was checked against the primary source directly)

1. Duan, Mao, Mao, Shu, Yin, "Breaking the Sorting Barrier for Directed
   Single-Source Shortest Paths," STOC 2025 Best Paper.
   https://dl.acm.org/doi/10.1145/3717823.3718179 · arXiv:2504.17033
2. Independent implementation/benchmark of (1): arXiv:2511.03007,
   "Implementation and Brief Experimental Analysis of the Duan et al.
   (2025) Algorithm for Single-Source Shortest Paths."
3. Leiserson et al., "There's Plenty of Room at the Top," Science 368
   (2020), eaam9744. https://doi.org/10.1126/science.aam9744 — matrix
   multiplication example (§ on performance engineering).
4. Sherry & Thompson, "How Fast Do Algorithms Improve?," Proceedings of
   the IEEE (Sept 2021). https://ide.mit.edu/wp-content/uploads/2021/09/How_Fast_Do_Algorithms_Improve.pdf
5. Edunov et al., "Three and a half degrees of separation," Meta
   Research blog (Feb 2016).
   https://research.facebook.com/blog/2016/02/three-and-a-half-degrees-of-separation/
6. PCAST, "Designing a Digital Future" (2010), p.71 — the Grötschel LP
   benchmark figure. Presented WITH its caveat (Bixby's competing
   multipliers for the same era) per the research pack below.
7. Mankowitz et al., "Faster sorting algorithms discovered using deep
   reinforcement learning" (AlphaDev), Nature (2023).
8. DeepMind, "AlphaEvolve" blog post (2025) — 4x4 complex matrix
   multiplication in 48 scalar multiplications.

## Editorial source material (Bear-provided, triaged)

Bear supplied three research documents for this topic. Two are used as
primary editorial sources — a sourced "research pack" (spine question,
evidence bank, explicit claims-to-avoid section) and a "research brief"
(NIST algorithm definition, worked BFS grid example, narration draft) —
because both are self-aware about their own caveats and their numeric
claims independently verified true (see FACTCHECK.md).

**The third document — an "Algorithmic Foundations in Computational
Science" lecture blueprint — is EXCLUDED from this film's factual basis.**
It contains specific, unverifiable administrative detail (an exact
Liberty Mutual "Career Connect" event date/time/location, a named TA
with weekly office hours, a specific Sept 13 deadline) that this session
could not verify and that reads as fabricated rather than sourced. None
of that content appears in this film. Its general framing (Knuth's five
properties, the quadratic paradox, the doubling hypothesis) overlaps
with the other two documents and the course book, and is not itself
excluded where independently corroborated.

## Deliberately not used

- The GitHub repo `danalec/DMMSY-SSSP` claim of "speedups exceeding
  20,000x over standard Dijkstra" — an unofficial experimental
  implementation, contradicted by the peer-reviewed independent
  benchmark (arXiv:2511.03007) actually cited in this film. The film
  uses the peer-reviewed conclusion (loses to well-tuned Dijkstra in
  practice), not the GitHub claim.
- The precise "62,806×" / "23,224×" intermediate multipliers from
  Bear's research pack for the Leiserson matrix-multiplication table —
  this session's own primary-source extraction only confirms "more than
  60,000 times faster" and the named per-stage multipliers actually
  printed in the paper's prose (Java 10.8×, C total 47× vs. Python,
  hardware tailoring ~1300× more). The film uses only the confirmed
  figures.
- The claim that applying Strassen's algorithm to the fastest version
  gains "about another 10%" — could not locate this in the extracted
  primary-source text this session; dropped rather than asserted
  unverified.

## Credits

No external media. All frames Manim/Remotion, house-drawn.
