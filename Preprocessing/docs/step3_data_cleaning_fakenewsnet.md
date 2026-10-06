# Step 3: Data Cleaning — FakeNewsNet (UPFD)

**Date:** 18-09-2026
**Author:** Adrija Pal

## Cleaning checks performed

1. Edge list validity — all node indices in `A.txt` within valid range
2. Graph ID / label consistency — every graph ID has exactly one label
3. Feature matrix row alignment — all four feature files match node count
4. Empty graph detection — no graphs with zero assigned nodes
5. Label validity — confirmed strictly binary (0.0 / 1.0)

## Results

| Check | Politifact | Gossipcop |
|---|---|---|
| Nodes | 41,054 | 314,262 |
| Graphs | 314 | 5,464 |
| Edges (raw) | 40,740 | 308,798 |
| Invalid edges | 0 | 0 |
| Graph ID/label mismatch | None | None |
| Feature rows match nodes | ✅ Yes | ✅ Yes |
| Empty graphs | 0 | 0 |
| Labels valid binary | ✅ Yes | ✅ Yes |

**No exclusions required** — unlike Weibo, this dataset needed zero
records removed. This is expected: UPFD is a pre-processed academic
benchmark already cleaned by its original creators (safe-graph/
GNN-FakeNews) prior to release.

## Output files