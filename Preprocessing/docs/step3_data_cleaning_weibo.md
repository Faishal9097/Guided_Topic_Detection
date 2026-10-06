## ⚠️ Important finding: excluded posts are exclusively from the rumor class

Follow-up analysis (see `scripts/check_excluded_split.py`) revealed that
**all 87 excluded posts belong to the rumor (guided) class — none are
non-rumor.**

| Class | Before cleaning | After cleaning | Change |
|---|---|---|---|
| Rumor (guided) | 1,538 | 1,451 | -87 (-5.7%) |
| Non-rumor (nonguided) | 1,849 | 1,849 | 0 |

**New class balance**: 1,451 / 3,300 (44.0%) rumor vs. 1,849 / 3,300
(56.0%) non-rumor — shifted slightly from the original 45.4% / 54.6%.

**Possible explanation**: rumor-flagged posts on Weibo are more likely
to have had their author accounts suspended, deleted, or restricted
(since they're identified as misinformation), which may corrupt
associated user metadata at crawl time. This would make data corruption
systematically more likely for the rumor class than the non-rumor class.

**Implication**: this is a non-random exclusion pattern, not simple
noise. It slightly reduces rumor-class representation and may remove
disproportionately "hard to crawl" (potentially more aggressively
suppressed/deleted) rumor examples, which could make the remaining
rumor examples marginally easier to distinguish than a fully
representative sample would be.

**Recommendation**: flagged for team lead awareness. No corrective
action taken at this stage (correcting for this would require either
recovering the missing data, which isn't possible, or statistical
reweighting, which is a modeling decision outside the scope of data
preparation).