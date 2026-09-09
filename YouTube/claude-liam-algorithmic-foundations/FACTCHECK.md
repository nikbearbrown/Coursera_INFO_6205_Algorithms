# FACTCHECK — claude-liam-algorithmic-foundations

Every narrated claim, checked before rendering. Verdicts: PASS (verified
against the named source, several via live WebSearch/WebFetch this
session against the primary source itself), EXEMPT (an authored framing
choice, not a factual claim).

| # | Beat | Claim | Source / method | Verdict |
|---|---|---|---|---|
| 1 | B00 | Modern hardware is "roughly a million times faster" than Dijkstra's 1959 machine | Order-of-magnitude framing device from the editorial research pack; not a precise measurement — hedged with "roughly" on screen and in narration | EXEMPT |
| 2 | B04 | An algorithm is a finite sequence of well-defined steps: input, output, finiteness, definiteness, effectiveness | Course text, `INFO_6205_Algorithms.md` lines 63-83 ("Definition and Core Concepts") | PASS |
| 3 | B05 | Five named growth classes (constant/log/linear/linearithmic/quadratic) and their ordering | Course text, lines 169-190 ("Time Complexity") | PASS |
| 4 | B06 | The quadratic paradox: 10x hardware + proportional memory growth -> 10x more data fed to an O(N²) algorithm -> new machine runs 10x slower than the old one on the smaller job | Robert Sedgewick's formulation, as described in the editorial research pack; the arithmetic is independently verifiable: (10N)²/(10×speed) = 10 × N²/speed | PASS |
| 5 | B08 | PCAST 2010 "Designing a Digital Future" p.71: a production-planning LP benchmark, 1988-2003, ~82 years -> ~1 minute; ~1,000x hardware, ~43,000x algorithms | PCAST report (per editorial research pack); WebSearch this session located the report title/page reference but did not independently re-derive the multipliers from the primary PDF text | PASS (source-cited; multipliers not independently re-extracted from primary text this session) |
| 5b | B08 | The solver vendor's own accounting of the same era gives different hardware/algorithm multipliers | Explicit caveat carried from the editorial research pack; stated ON SCREEN precisely because it is a known point of disagreement, not resolved to a single number | EXEMPT (disclosed uncertainty, not a bare claim) |
| 6 | B09 | Sherry & Thompson, "How Fast Do Algorithms Improve?", Proc. IEEE 2021: 57 textbooks, 1,137+ papers, 113 algorithm families; ~half saw little/no improvement, ~13% transformative, 30-45% (moderate/large problems) matched or exceeded Moore's-Law-rate hardware gains | Verified this session via WebSearch directly against the paper's reported findings (MIT CSAIL news summary + paper abstract) — exact percentages confirmed | PASS |
| 7 | B11 | Leiserson et al., "There's Plenty of Room at the Top," Science 2020: naive Python 4096x4096 matrix multiply ~7 hours; Java 10.8x faster; C brings total to 47x faster than Python; hardware-tailored version (parallel + cache + vectorized + AVX) finishes in ~0.41s, "more than 60,000 times faster" than the Python start | Verified this session via WebFetch of the primary Microsoft Research-hosted PDF — the 10.8x, 47x-vs-Python, ~0.41s, and "more than 60,000 times faster" figures are quoted directly from the paper's own prose | PASS |
| 7b | — | The precise intermediate multipliers "23,224x" and "62,806x" from Bear's editorial doc | NOT independently confirmed in the extracted primary-source text this session (table garbled in extraction) — DROPPED from narration; only the confirmed prose figures are spoken | EXEMPT (claim removed, not asserted) |
| 7c | — | "Applying Strassen's algorithm gains ~10% more" | Could not locate in this session's primary-source extraction — DROPPED from the film entirely | EXEMPT (claim removed, not asserted) |
| 8 | B14 | Sequence/selection/iteration examples (sum an array, max of two, factorial) | Course text, lines 111-168 | PASS |
| 9 | B15 | BFS on a grid expands one full ring of distance at a time via a FIFO queue; the full correctness proof is covered in a separate film | The already-published `claude-liam-algo-bfs-dfs` deeper (FACTCHECK.md there verifies the BFS layer-invariant proof); this film deliberately does NOT re-derive it, only references it, per the "light touch" scoping decision | PASS |
| 10 | B16 | Facebook 2016 ("Three and a half degrees of separation"): 1.59 billion active accounts, average distance 4.57 hops (3.57 intermediary "degrees"); computed via probabilistic estimation, not exact BFS from every account; 2011 figure was 3.74 degrees across 721 million accounts | Verified this session via WebSearch directly against the Meta Research blog post and contemporaneous press coverage — all figures match | PASS |
| 11 | B18 | AlphaDev (Mankowitz et al., Nature 2023): found shorter sorting routines for short sequences, merged into LLVM's libc++; "up to 70% faster" for short sequences, "~1.7% faster" for sequences over 250,000 elements | Widely reported primary result (Nature / DeepMind); not independently re-verified against the Nature PDF this session — carried from the editorial research pack as a well-known, consistently reported figure | PASS (source-cited, not independently re-extracted this session) |
| 12 | B18 | AlphaEvolve (DeepMind, 2025): found a 4x4 complex-valued matrix multiplication using 48 scalar multiplications (vs. 49 via composed Strassen), first improvement in that exact setting since 1969; a derived scheduling heuristic recovered ~0.7% of Google's fleet-wide compute on Borg | DeepMind's own blog post (per editorial research pack); not independently re-verified against the primary announcement this session | PASS (source-cited, not independently re-extracted this session) |
| 12b | B18/BVDT | Correction carried from the editorial research pack, NOT restated as a bare claim in this film: Strassen's 1969 result is 2x2 in 7 multiplications, not a directly published 4x4-in-49; 49 is what recursive composition of the 2x2 result gives for 4x4 | The film says "since 1969" without restating the frequently-garbled "49 multiplications" figure, to avoid repeating the common error while still being accurate | EXEMPT (imprecision avoided by omission) |
| 13 | B19 | Duan, Mao, Mao, Shu, Yin, "Breaking the Sorting Barrier for Directed Single-Source Shortest Paths," STOC 2025 Best Paper: a deterministic algorithm, first to beat Dijkstra's O(m + n log n) bound on sparse graphs with real non-negative weights | Verified this session via WebSearch directly against the ACM DL listing, the MPI-Informatik award announcement, and the arXiv abstract | PASS |
| 14 | B20 | An independent implementation/benchmark of the Duan et al. algorithm found it does NOT outperform well-tuned Dijkstra in practice | Verified this session via WebFetch of the primary arXiv paper (2511.03007) itself — its stated conclusion is quoted, not inferred | PASS |
| 14b | — | The GitHub repo `danalec/DMMSY-SSSP` claim of "speedups exceeding 20,000x over standard Dijkstra" | Found via WebSearch; an unofficial experimental implementation that DIRECTLY CONTRADICTS the peer-reviewed arXiv:2511.03007 finding used in row 14. NOT used anywhere in this film — flagged in SOURCES.md as deliberately excluded | EXEMPT (claim rejected, not asserted) |
| 15 | BVDT | All five verdict lines | Recaps of rows 2, 4, 6, 12b/7, 13-14 — no new claims | PASS |

## Strip-the-datable check

Beat B00/B01/BOUT name the course (INFO 6205 — Algorithms), the
professor (Nik Bear Brown), and the institution (Northeastern University)
— deliberately, per Bear's explicit instruction that this film be
unambiguous as course material. This is the one exception to the house
"strip the datable" convention for the ai-algorithms roster films: THIS
film is not a topic-evergreen deeper, it is course-branded intro
material, so the branding is the point, not a defect. No semester dates,
deadlines, event names/times, or personnel (TAs, office hours) appear
anywhere — those were in the excluded third source document and are
never spoken or shown.

## Excluded source (see SOURCES.md for full reasoning)

The "Algorithmic Foundations in Computational Science..." lecture
blueprint's administrative specifics (Liberty Mutual Career Connect
event date/time/location, named TA + office hours, Sept 13 deadline) do
not appear anywhere in this film's beat sheet, narration, or on-screen
text. They could not be verified this session and read as fabricated
rather than sourced.
