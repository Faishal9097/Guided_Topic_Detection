# Group 4 — Code Reference

## File Map

| File | Responsibility |
|---|---|
| `group4_common.py` | Shared GC-GRU model, feature-block construction, loading Group 3 chunks/data, normalization, training loops, metrics, and seed aggregation. |
| `group4_build_struct.py` | Builds 11 structural features per topic and time slice, aligned to the topic order in a Group 3 `.npz`. |
| `group4_train.py` | Trains/evaluates one feature configuration across seeds; optionally saves the best-validation-AUC model and JSON metrics. |
| `group4_experiments.py` | Runs feature ablation, temporal-model comparison, and early-detection suites; includes paired significance analysis. |
| `group4_interpret.py` | Perturbation-based group contribution and stance-dominance analysis for a saved model. |

## Execution Order

1. Use Group 3 outputs in `Ag2vec_Grp_Representation/data_output/`.
2. Run `group4_build_struct.py` for each dataset.
3. Run `group4_train.py` for the chosen feature configuration.
4. Run `group4_experiments.py` for ablation, temporal comparison, or early detection.
5. Run `group4_interpret.py` for a saved model trained only on supported pooled blocks (`count`, `vec`, `mut`).

## PowerShell Setup

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location GC_GRU\scripts
$G = "..\..\Ag2vec_Grp_Representation\data_output"
$R = "..\results"
New-Item -ItemType Directory -Force $R | Out-Null
```

## Command Reference

### Structural features

```powershell
python group4_build_struct.py --npz "$G\weibo_group_feature_matrices.npz" --chunks "$G" --kind weibo --out "$R\weibo_struct.npz"
python group4_build_struct.py --npz "$G\politifact_group_feature_matrices.npz" --chunks "$G" --kind politifact --out "$R\politifact_struct.npz"
python group4_build_struct.py --npz "$G\gossipcop_group_feature_matrices.npz" --chunks "$G" --kind gossipcop --out "$R\gossipcop_struct.npz"
```

### Single configuration, five seeds

```powershell
python group4_train.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --blocks vec,mut --seeds 5 --save-model "$R\m_vecmut.pt" --out "$R\r_vecmut.json"
```

### Experiment suites

```powershell
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite ablation --out "$R\results_ablation.json"
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite temporal --blocks struct,mut --out "$R\results_temporal.json"
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite early --blocks struct,mut --out "$R\results_early.json"
```

### PolitiFact/GossipCop ablation

These datasets are static and currently have no stance features. Do not include the `mut` block.

```powershell
python group4_experiments.py --npz "$G\politifact_group_feature_matrices.npz" --struct "$R\politifact_struct.npz" --suite ablation --class-weight --out "$R\results_politifact.json"
python group4_experiments.py --npz "$G\gossipcop_group_feature_matrices.npz" --struct "$R\gossipcop_struct.npz" --suite ablation --class-weight --out "$R\results_gossipcop.json"
```

### Interpretation

```powershell
python group4_interpret.py --npz "$G\weibo_group_feature_matrices.npz" --model "$R\m_vecmut.pt" --chunks "$G" --kind weibo --n-topics 100
```

## CLI Options

### `group4_train.py`

- `--npz` (required): Group 3 feature matrix.
- `--struct`: optional structural-feature matrix.
- `--blocks`: comma-separated feature blocks; default `vec,mut`.
- `--unit`: `gru`, `lstm`, `transformer`, or `mlp`; default `gru`.
- `--seeds`: number of runs; default 5.
- `--T`: use only the first N slices (early detection).
- `--epochs`, `--lr`, `--batch`, `--hidden`, `--layers`, `--dropout`, `--fc`: override training defaults.
- `--class-weight`: enable class weighting.
- `--save-model`: save the best-validation-AUC model checkpoint.
- `--out`: write run metrics/configuration to JSON.

### `group4_experiments.py`

- `--suite`: `ablation`, `temporal`, or `early` (required).
- `--npz`: Group 3 feature matrix (required).
- `--struct`: structural features, needed for `struct`/`size` blocks.
- `--blocks`: feature blocks for temporal/early suites; default `vec,mut`.
- `--seeds`: runs per configuration; default 5.
- `--epochs`: epoch limit.
- `--class-weight`: enable class weighting.
- `--ref`: reference configuration for significance comparisons.
- `--out`: output JSON path.

### `group4_build_struct.py`

- `--npz`: Group 3 matrix (required).
- `--chunks`: directory containing Group 3 chunk files (required).
- `--kind`: `weibo`, `politifact`, or `gossipcop` (required).
- `--out`: structural-feature output path (required).

### `group4_interpret.py`

- `--npz`: Group 3 matrix (required).
- `--model`: saved checkpoint (required).
- `--chunks`: Group 3 chunk directory, optional but needed for group metadata.
- `--kind`: dataset kind; default `weibo`.
- `--n-topics`: number of test topics to sample; default 100.
- `--tstar`: optional zero-based slice; default is the slice with the most groups.
- `--across`: remove a persistent group across all slices.
- `--seed`: random seed; default 0.

## Feature Blocks

- `count`: group-count/time information.
- `vec`: pooled AG2Vec group-vector information.
- `mut`: pooled stance/mutual-influence information.
- `flat`: flattened group-feature baseline.
- `struct`: the 11 structural features from `group4_build_struct.py`.
- `size`: simple group-size baseline.

## Saved Artifacts

- `*_struct.npz`: structural features, shape `(n_topics, T, 11)`.
- `m_*.pt`: model state, feature normalization values, feature blocks, architecture/configuration, selected seed, and time horizon.
- `r_*.json`: per-seed training metrics and summary.
- `results_*.json`: suite results and paired comparisons.

## Implementation Notes

- The final Group 3 `.npz` matrices are the primary model inputs; Group 3 chunk files are additionally needed for structural feature creation and interpretation.
- Weibo has 10 time slices; PolitiFact and GossipCop are static (`T=1`).
- `early` experiments require `T > 1`, so run them on Weibo.
- If `stance_included` is false, the `mut` block carries no information and should be omitted.
- The recorded defaults are hidden size 128, 2 layers, dropout 0.5, FC size 256, learning rate 0.004, batch size 32, and 32 epochs.
