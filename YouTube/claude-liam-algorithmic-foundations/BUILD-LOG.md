# BUILD-LOG — claude-liam-algorithmic-foundations

## 2026-09-09 — built end-to-end, ALL GATES PASS (awaiting Bear's review)

Course-intro film for INFO 6205 Module 1 ("The Importance of Algorithms"),
per Bear's request: ai deep explainer, Liam persona, title "Algorithmic
Foundations", course-branded (INFO 6205 — Algorithms · Nik Bear Brown ·
Northeastern University), figures animated fresh from the course book.
NOT one of the 11 ai-algorithms roster deepers — a standalone sibling.

**Source triage (before any beat was authored):** Bear supplied three
research documents. Two — a sourced "research pack" and a "research
brief" — were used, after this session independently re-verified their
highest-stakes claims via WebSearch/WebFetch against primary sources
(Duan et al. 2025 STOC Best Paper + the independent benchmark that shows
it losing to well-tuned Dijkstra in practice; the Leiserson Science 2020
matrix-multiply table; Sherry & Thompson 2021's uneven-progress
percentages; Facebook's 2016 "4.57 hops" figure — all confirmed exactly).
The third document (an "Algorithmic Foundations in Computational
Science" lecture blueprint) was EXCLUDED entirely: it contained a named
TA with office hours, an exact Liberty Mutual event date/time, and a
specific Sept 13 deadline — none verifiable, all read as fabricated.
None of that appears anywhere in this film. See SOURCES.md/FACTCHECK.md.

**Plan:** 24 beats, ~6:12. B00 narrated cold open (course line spoken +
shown), B01 course-branded title card, B02 hesitant-writer overview, five
acts (What an Algorithm Actually Is / The Receipts / What This Course Is
Not / The Queue Is the Algorithm / Still Open), BVDT/BHTF/BOUT close.
9 Manim scenes (book's own definitions/pseudocode/complexity figures,
animated fresh — none copied), 4 DeckPattern REMOTION beats, 6 segment
cards. No pantry/vox — this genre, like ai-algorithms, never asks for
media.

**Defects found by reading frames, fixed at root cause:**
- B01's title-card actLabel carried the full course-credit line and hit
  BOTH title-safe edges (GATE V edge-bleed BLOCKER) — actLabel is a short
  kicker field, not a full credit line; shortened to "INFO 6205 —
  Algorithms" (the fuller credit already lives in B00's segment field,
  the narration, and BOUT).
- B04's four washed property-boxes: labels stayed MUTE-colored under a
  terracotta highlight wash, reading genuinely low-contrast (not the
  usual false-positive class) — fixed by transitioning each label to
  full INK the moment its wash lands, per house convention that
  highlighted elements read at full ink.
- B04 (post-fix) and B15: a large-area wash/fill (property-box
  highlights; the fully-filled BFS grid) pulled the WHOLE-FRAME mean ink
  luminance below the 0.3 floor even though every actual text element
  reads at full local contrast (eye-verified both). Same mechanism as
  the house's existing B06_TheBand/MbgDeletePages exemptions — added
  `B04_FiveProperties` and `B15_FrontierRipple` to
  `LOW_CONTRAST_OK_PATTERNS` in `runtime/qc/final_frame_check.py`,
  frames cited in the justification comment.
- GATE T: three DeckPattern `note` fields ran one word over the 12-word
  pull-quote budget (em-dashes count as tokens) — trimmed all three to
  8-9 words without losing the claim.

Warning accepted (pre-justified): drawon carries 62% of beats — this
genre is bookend/card-heavy by design (five act-cards, a title card, a
hesitant-writer overview), same waiver as the ai-algorithms roster.

**Verification:** 3840x2160, 372.4s (6:12), audible from t=0
(mean_volume -23.6 dB in the first 4s). Two mid-film frames read by eye,
both clean. NOT staged to TOPOST, NOT published — Bear reviews first.

## 2026-09-09 — text artifacts pushed to GitHub (safeguard)

This session already lost 7 built ai-algorithms films on 2026-09-06 when
this exact `Coursera_INFO_6205_Algorithms/` folder was replaced by a
fresh `git clone` — none of that work had ever been committed. This film
is pushed immediately after building (beat_sheet.json, SOURCES.md,
FACTCHECK.md, BUILD-LOG.md, manim/scenes.py, scenes.py proxy — media
stays gitignored per the existing `.gitignore` rules) so a future re-clone
cannot repeat that loss for this film's text/recipe, even though the
rendered master itself is not committed.
