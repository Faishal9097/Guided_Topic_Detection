# Step 4: Label Construction — Weibo (CED)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

Since the paper's exact per-sample "guided" labeling methodology
(manual review + unpublished remapping rule) can't be reproduced
directly, we computed a continuous "guided score" per post from three
signals approximating the paper's Table I framework:

1. **Propagation dynamics**: statistically significant activity peaks
   in reply/repost timing (mean + 2×std threshold)
2. **Propagator structure**: burst density of unique users replying
   within short (5-min) windows, relative to the thread's own baseline
   rate
3. **Textual narrative**: ratio of templated/low-effort repost phrases
   (e.g., "转发微博") vs. genuine commentary

Each signal was normalized (0-1) across the dataset and averaged into
a single combined score per post.

## Calibration (important methodological note)

The cutoff threshold for "guided" vs "nonguided" was **calibrated to
match the paper's known aggregate proportion** (1,849 guided / 1,538
nonguided = 54.6%/45.4%), since the paper does not publish which
specific samples were labeled which way — only the aggregate counts.

**This means the overall totals matching the paper is BY DESIGN, not
independent validation.** The genuinely informative, independently-
derived result is the **cross-tabulation** against the original
rumor/non-rumor labels (see below), which was NOT calibrated and
emerged directly from the computed scores.

## Results

| | Nonguided | Guided | Total |
|---|---|---|---|
| Rumor | 736 | 802 | 1,538 |
| Non-rumor | 802 | 1,047 | 1,849 |

## Interpretation

- Rumors split nearly evenly between guided/nonguided (52% guided),
  supporting the paper's framing that rumors can spread either
  organically or through coordination — being a rumor does not imply
  being guided.
- Non-rumors lean more guided (57%), suggesting coordinated
  amplification is not exclusive to misinformation — consistent with
  real-world astroturfing/promotional campaign behavior.
- No degenerate collapse into either original category — the computed
  labels are a genuinely distinct signal, not a disguised copy of the
  rumor/non-rumor split.

## Known limitations
- Textual narrative signal used only simple templated-phrase detection
  (not full NLP sentiment/bias analysis) due to Chinese-language NLP
  tooling being out of scope for this stage.
- Thresholds for dynamics/structure signals were manually tuned through
  iterative diagnostic testing (see script history) rather than derived
  from the paper's own (unpublished) exact methodology.
- This is a best-effort computational approximation, not a reproduction
  of the paper's manual annotation process.

## Output
`data/processed/weibo_ced/guided_labels_computed.json` — per-post
scores, normalized signals, and final guided/nonguided label.

## Script used
See `scripts/label_construction_weibo.py`.