# Step 4: Label Construction — FakeNewsNet (UPFD)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

UPFD provides no timestamp data, so the propagation dynamics dimension
(Table I) could not be computed for this dataset — a hard limitation,
not a modeling choice. Guided-score was computed from 2 dimensions:

1. **Propagator structure**: "star-ness" of each propagation graph —
   ratio of the single most-connected node's out-degree to total edges
   in that graph. High ratio = centralized/hub-dominated spread
   (coordination-like); low ratio = distributed organic spread.
2. **Textual narrative**: average pairwise cosine similarity of
   content-feature vectors across nodes in each graph. High similarity
   = near-identical/templated content (coordination-like); low
   similarity = varied organic commentary.

Both signals normalized (0-1) per dataset and averaged into a combined
score. Cutoff calibrated to a 50/50 split (matching UPFD's inherent
class balance).

## Results

**Politifact** (n=314 — small sample, weak signal):
| | Nonguided | Guided |
|---|---|---|
| Real | 71 | 86 |
| Fake | 85 | 72 |

Close to 50/50 in both classes — likely too small a sample for a clear
pattern to emerge from these 2 signals alone.

**Gossipcop** (n=5,464 — large sample, clear signal):
| | Nonguided | Guided |
|---|---|---|
| Real | 1,003 | 1,729 (63%) |
| Fake | 1,728 | 1,004 (37%) |

Clear pattern: real content skews guided, fake content skews nonguided.

## Cross-dataset consistency (notable finding)

This directional pattern (true/real content → more guided; false/fake
content → less guided) **matches the independently-computed Weibo
result** (non-rumor 57% guided vs. rumor 52% guided), despite using
completely different signal types (Weibo: timing-based; Gossipcop:
graph-structure + content-similarity based). This consistency across
two different datasets and methodologies suggests the pattern may
reflect something real, rather than being an artifact of either
specific method — though this remains a preliminary, descriptive
observation, not a validated causal claim.

## Known limitations
- Propagation dynamics dimension entirely unavailable (no timestamps
  in UPFD) — only 2 of 3 Table I dimensions used.
- Politifact's small sample size (314) yields a weak/near-random
  signal; Gossipcop's larger sample (5,464) shows a much clearer
  pattern — sample size appears to matter significantly for signal
  reliability with this method.
- Content similarity computed via provided pre-encoded features
  (bag-of-words/TF-IDF style, 310-dim), not raw text — limits
  interpretability of exactly what "similarity" captures.

## Output
`data/processed/fakenewsnet/{politifact,gossipcop}_guided_labels_computed.json`

## Script used
See `scripts/label_construction_fakenewsnet.py`.