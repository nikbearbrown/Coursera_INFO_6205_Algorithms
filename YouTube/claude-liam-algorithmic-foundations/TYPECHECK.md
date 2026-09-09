# TYPECHECK.md — GATE T

Reel: `claude-liam-algorithmic-foundations`  |  Checked: 2026-09-09T13:38  |  Overall: PASS  |  Beats checked: 24  |  FAILs: 0

Spec: `skills/make/kerning/reference/type-spec.md` §8.  Floor: 1.9% frame-height.  Contrast: 4.5:1 WCAG.  Kern threshold: 3.5× expected advance.  Wordy budget: 2 elements.

> **§8.10 REDUNDANCY (advisory — does not block cut):**
> Narration should DISCUSS on-screen text, not recite it.
> Exception: LITERAL beats (viewer types/copies/runs the text) are exempt.

> - §8.10 [BVDT] narration recites the card (0.93) — discuss it, don't read it

| beat | lane | polarity | worst finding | status | fix |
|------|------|----------|---------------|--------|-----|
| B00 | BOOKEND | light | min-size §8.1: hand-drawn pattern (ClaudeComposerAsk) — §8.1 hachure/crossbar fragments ar… | PASS | — |
| B01 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B02 | BOOKEND | light | min-size §8.1: min text-run height 64px >= floor 41px (individual-char fallback at 2×) | PASS | — |
| B03 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B04 | MANIM | light | min-size §8.1: min text-run height 145px >= floor 41px | PASS | — |
| B05 | MANIM | light | min-size §8.1: min text-run height 52px >= floor 41px | PASS | — |
| B06 | REMOTION | light | no-wordy-card §8.5: DeckPattern: per-element check passed (max 10 words in 'left.note') | PASS | — |
| B07 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B08 | MANIM | light | min-size §8.1: min text-run height 41px >= floor 41px | PASS | — |
| B09 | MANIM | light | min-size §8.1: min text-run height 44px >= floor 41px (individual-char fallback at 2×) | PASS | — |
| B10 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B11 | MANIM | light | min-size §8.1: min text-run height 77px >= floor 41px | PASS | — |
| B12 | REMOTION | light | no-wordy-card §8.5: DeckPattern: per-element check passed (max 9 words in 'left.note') | PASS | — |
| B13 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B14 | MANIM | light | min-size §8.1: min text-run height 44px >= floor 41px | PASS | — |
| B15 | MANIM | light | min-size §8.1: min text-run height 45px >= floor 41px (individual-char fallback at 2×) | PASS | — |
| B16 | REMOTION | light | no-wordy-card §8.5: DeckPattern: per-element check passed (max 9 words in 'right.note') | PASS | — |
| B17 | CARD | light | no-wordy-card §8.5: no prose payload found | PASS | — |
| B18 | REMOTION | light | no-wordy-card §8.5: DeckPattern: per-element check passed (max 11 words in 'right.note') | PASS | — |
| B19 | MANIM | light | min-size §8.1: min text-run height 44px >= floor 41px (individual-char fallback at 2×) | PASS | — |
| B20 | MANIM | light | min-size §8.1: min text-run height 136px >= floor 41px | PASS | — |
| BVDT | BOOKEND | light | min-size §8.1: hand-drawn pattern (ClaudeVerdictArtifact) — §8.1 hachure/crossbar fragment… | PASS | — |
| BHTF | BOOKEND | light | min-size §8.1: hand-drawn pattern (ClaudeComposerAsk) — §8.1 hachure/crossbar fragments ar… | PASS | — |
| BOUT | BOOKEND | dark | min-size §8.1: min text-run height 65px >= floor 41px | PASS | — |

---

## Failures requiring action before cut

*None — GATE T PASS.*
---

## Check summary

| Check | Beats checked | FAILs |
|-------|---------------|-------|
| no-wordy-card §8.5 | 10 | 0 |
| min-size §8.1 | 24 | 0 |
| overflow §8.2 | 24 | 0 |
| contrast §8.3 | 24 | 0 |
| contrast-local §8.3b | 24 | 0 |
| bbox-overlap §8.6b | 24 | 0 |
| card-clip §8.13 | 24 | 0 |
| kerning §8.4 | 9 | 0 |
| redundancy §8.10 (advisory) | 1 | 1 (advisory — no exit effect) |

---

*GATE T: any FAIL blocks `./art run` and `./art final`. Fix the flagged beats and re-run `scripts/type_check.py` until green.*
