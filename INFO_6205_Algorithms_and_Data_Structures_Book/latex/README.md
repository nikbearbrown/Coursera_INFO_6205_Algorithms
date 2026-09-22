# LaTeX source — INFO 6205 Algorithms and Data Structures

This is the Overleaf project export for the book (Nik Bear Brown and Nimish Magre, March 2024), unpacked from `Coursera_I2A_Algorithms.zip` on 2026-09-22.

## What builds the book

| File | Role |
|---|---|
| **`Final_Book_1.tex`** | **The complete book — this is the file to compile.** 27 chapters, self-contained (no `\input`s). Byte-identical to `../INFO_6205_Algorithms_and_Data_Structures_Book.tex`. |
| `images/*.png` | The 9 figures `Final_Book_1.tex` includes (Kruskal, weighted graph, Ford-Fulkerson, push-relabel, flow network, assignment problem ×2, greedy vertex cover). These were missing from the repo before this export. |
| `Final_Book_1.pdf` | Built from `Final_Book_1.tex` with pdflatex on 2026-09-22 — 0 errors. |
| `main.tex` | A 29-chapter **outline skeleton** (headings only, no body text) — the book's original plan. Not the book. |
| `<Chapter> Edited.tex` / `<Chapter>.tex` | Per-chapter working drafts: `Edited` versions are the ones merged into `Final_Book_1.tex`; the un-suffixed ones are earlier drafts kept for history. Neither set is `\input` by anything. |
| `Graph_Creation.tex` | A small TikZ scratch file. |

## Build

```bash
cd latex
pdflatex -interaction=nonstopmode Final_Book_1.tex   # run three times for TOC + cross-refs
```

Packages required beyond BasicTeX: `algorithms`, `algorithmicx`, `makecell`, `pgfplots`, `placeins`. Without admin, install them in tlmgr user mode:

```bash
tlmgr init-usertree && tlmgr --usermode install algorithms algorithmicx makecell pgfplots placeins
```

## Known warnings in the source (harmless, worth fixing someday)

- `\label{ford-fulkerson}` defined twice (lines ~8081 and ~9021) and `\label{alg:ford-fulkerson}` twice (~8204 and ~18772) — cross-refs resolve to the last one.
- `\ref{alg:ssp}` (line ~8650, capacity-scaling paragraph) has no matching `\label` — renders as "??".
- ~290 overfull hboxes, mostly long URLs and code listings; one float 19.6pt too tall.
