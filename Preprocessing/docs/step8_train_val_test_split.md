# Step 8: Train/Val/Test Split

**Date:** 18-09-2026
**Author:** Adrija Pal

## Approach

Split at the **topic level** (each topic stays whole, never split across
sets) into 80% train / 10% validation / 10% test, per the paper's
Algorithm 1 (step 11). Split is **stratified by guided/nonguided label**
within each dataset, so class balance is preserved across all three
sets — done independently per class, then combined.

Fixed random seed (42) used for reproducibility.

## Results

| Dataset | Total | Train | Val | Test |
|---|---|---|---|---|
| Weibo | 3,300 | 2,639 (80.0%) | 329 (10.0%) | 332 (10.1%) |
| Politifact | 314 | 250 (79.6%) | 30 (9.6%) | 34 (10.8%) |
| Gossipcop | 5,464 | 4,370 (80.0%) | 546 (10.0%) | 548 (10.0%) |

Politifact's ratios deviate slightly from exact 80/10/10 due to
rounding at small sample size (314 topics split three ways) — expected
and within acceptable tolerance.

## Class balance preserved across splits

| Dataset | Split | Guided | Nonguided |
|---|---|---|---|
| Weibo | Train | 1,442 | 1,197 |
| | Val | 180 | 149 |
| | Test | 181 | 151 |
| Politifact | Train | 126 | 124 |
| | Val | 15 | 15 |
| | Test | 17 | 17 |
| Gossipcop | Train | 2,186 | 2,184 |
| | Val | 273 | 273 |
| | Test | 274 | 274 |

## Output files