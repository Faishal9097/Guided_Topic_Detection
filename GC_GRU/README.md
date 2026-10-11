# Group 4 — GC-GRU (evaluation of group and stance features)

Group 4 consumes the final Group 3 feature matrices and evaluates how well user-group and stance features predict
**rumor vs. non-rumor (Weibo)** and **fake vs. real (PolitiFact, GossipCop)**. It also compares GC-GRU against
non-temporal and non-group baselines.

> **Target.** The primary label is `original_label` (rumor / fake = 1, non-rumor / real = 0), stored in the
> `labels` array of every feature `.npz`. The earlier rule-based "guided / non-guided" label is not used for
> evaluation. On Weibo it is a secondary label only (it depends on the same volume and repost signals as the features,
> and it is not reproducible on FakeNewsNet).

## Evaluation protocol
- **Repeated stratified CV:** 5 folds × 3 repeats over all topics (`--cv`). 10% of each training part is a validation set
  for epoch selection. The fixed train/val/test split is only used by runs without `--cv`.
- **Metrics:** accuracy, balanced accuracy, precision, recall, F1, macro-F1, ROC-AUC and PR-AUC (mean ± std over folds).
  Class weights are not needed: the classes are about 44/56 on Weibo and 50/50 on FakeNewsNet.
- **Significance:** paired t-tests over folds. Folds overlap, so p-values are optimistic. Treat them as indicative.
- **Baselines:** size-only, gradient boosting on pooled structural / stance features, and a text-only model (character TF-IDF +
  logistic regression on the original post), plus text + struct + mut stacking.

## Data
| Dataset | Topics | Slices | Stance | Target |
|---|---|---|---|---|
| Weibo CED | 3,300 | 10 | yes (Mut, `clean` mode) | rumor = 1 |
| PolitiFact | 266 | 1 | no | fake = 1 |
| GossipCop | 5,416 | 1 | no | fake = 1 |

Matrices come from `Ag2vec_Grp_Representation/scripts/run_mut_export_windows.py` with `MODE = "clean"`: Mut is computed without
the repost share (Act) and topic heat, and empty-text reposts are excluded from group stance. The stance classifier is
trained on `stance_sample_labeled.csv` (300 LLM-labelled comments).

## Setup (PowerShell)
```powershell
cd "D:\#Project\Innovative 7th\Guided_Topic_Detection"
.\.venv\Scripts\Activate.ps1
Set-Location GC_GRU\scripts
$G = "..\..\Ag2vec_Grp_Representation\data_output"
$R = "..\results"
$OM = "..\..\Preprocessing\data\processed\weibo_ced\original_microblog_cleaned.json"
```

## Run order
1. **Structural features** (11 per topic/slice): `group4_build_struct.py --npz ... --chunks "$G" --kind weibo|politifact|gossipcop --out "$R\*_struct.npz"`
2. **GC-GRU, CV:** `group4_train.py --npz ... --struct ... --blocks struct,mut --cv`  (add `--lr 0.001 --epochs 40` for the lower-LR run)
3. **Suites:** `group4_experiments.py --suite ablation|temporal|early --cv --out "$R\results_*.json"`
4. **Baselines:** `group4_baselines.py` (gradient boosting) and `group4_text_baselines.py` (Weibo text and stacking, needs `--om "$OM"`)
5. **Fixed-split model for interpretation:** `group4_train.py --blocks vec,mut --seeds 5 --save-model "$R\m_vecmut.pt" --out "$R\r_vecmut.json"`, then `group4_interpret.py`

Do not use `--class-weight` or `mut` blocks on PolitiFact / GossipCop (no stance, balanced classes).

## Latest results (5×3 CV, AUC / PR-AUC) — refresh after the rerun
| Weibo | AUC | PR-AUC |
|---|---|---|
| GB, size only | 0.674 | 0.611 |
| GB, struct | 0.753 | 0.703 |
| GB, struct + mut | 0.840 | 0.801 |
| GC-GRU, struct + mut (lr 0.004 / 0.001) | 0.820 / 0.823 | 0.774 / 0.776 |
| MLP, struct + mut | 0.830 | 0.783 |
| Text only | 0.961 | 0.956 |
| Text + struct + mut | 0.969 | 0.964 |

PolitiFact: all configurations are about 0.57–0.60 AUC (too few topics to conclude). GossipCop: gradient boosting on struct gets about 0.87; size alone gets about 0.83.

## Findings and caveats
- Group and stance features predict rumor / non-rumor above size-only baselines (0.84 vs 0.67 AUC), but add about 0.006 AUC beyond the post text.
- Temporal order does not help: an MLP over time-averaged features matches or beats GRU/LSTM, and early detection with 2 of 10 slices is within about 0.015 AUC of all 10.
- AG2Vec group vectors (`vec`) add nothing over size on any dataset.
- Results are not comparable with the earlier guided-label results (0.934 AUC on Weibo), which were measured on one split with a label built from the same volume signals.
- The paper's guided / non-guided numbers are a different task and are reference only.
- FakeNewsNet results are structure-only (no stance, one slice).

## Outputs
- `results\*_struct.npz` structural features, `results\m_*.pt` model checkpoint, `results\r_*.json` training metrics,
  `results\results_*.json` suite results, `results\log_*.txt` console logs.

See `scripts\GROUP4_CODE_REFERENCE.md` for the role of each script and the CLI options.