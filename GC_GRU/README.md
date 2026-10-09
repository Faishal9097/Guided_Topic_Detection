# Group 4 — GC-GRU

Group 4 consumes the final Group 3 feature matrices and trains/evaluates the GC-GRU classifier for guided vs. non-guided topics. It also supports structural-feature generation, ablation/model-comparison/early-detection experiments, and perturbation-based interpretation.

## Pipeline

1. Complete Preprocessing, Group 2, and Group 3.
2. Generate structural features from Group 3 chunks.
3. Train GC-GRU and save metrics/model.
4. Run experiment suites and interpretation as needed.

## Inputs

Group 3 matrices in `Ag2vec_Grp_Representation/data_output/`:

- `weibo_group_feature_matrices.npz`
- `politifact_group_feature_matrices.npz`
- `gossipcop_group_feature_matrices.npz`

Group 3 chunk files are also needed for structural features and interpretation. The model's structural feature matrix has 11 features per topic/slice.

## Setup (PowerShell)

Run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location GC_GRU\scripts
$G = "..\..\Ag2vec_Grp_Representation\data_output"
$R = "..\results"
New-Item -ItemType Directory -Force $R | Out-Null
```

## Run Commands

### 1. Build structural features

```powershell
python group4_build_struct.py --npz "$G\weibo_group_feature_matrices.npz" --chunks "$G" --kind weibo --out "$R\weibo_struct.npz"
python group4_build_struct.py --npz "$G\politifact_group_feature_matrices.npz" --chunks "$G" --kind politifact --out "$R\politifact_struct.npz"
python group4_build_struct.py --npz "$G\gossipcop_group_feature_matrices.npz" --chunks "$G" --kind gossipcop --out "$R\gossipcop_struct.npz"
```

### 2. Train the default GC-GRU configuration (Weibo)

```powershell
python group4_train.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --blocks vec,mut --seeds 5 --save-model "$R\m_vecmut.pt" --out "$R\r_vecmut.json"
```

Alternative structural + mutual-influence configuration:

```powershell
python group4_train.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --blocks struct,mut --seeds 5 --out "$R\r_structmut.json"
```

### 3. Run experiments (Weibo)

Feature ablation:

```powershell
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite ablation --out "$R\results_ablation.json"
```

Temporal model comparison (GRU, LSTM, Transformer, MLP):

```powershell
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite temporal --blocks struct,mut --out "$R\results_temporal.json"
```

Early detection across the first 2, 4, 6, 8, and all available slices:

```powershell
python group4_experiments.py --npz "$G\weibo_group_feature_matrices.npz" --struct "$R\weibo_struct.npz" --suite early --blocks struct,mut --out "$R\results_early.json"
```

### 4. Run experiments for PolitiFact or GossipCop

Use the corresponding matrix and structural file. These datasets have one time slice and no stance features, so avoid `mut` blocks.

```powershell
python group4_experiments.py --npz "$G\politifact_group_feature_matrices.npz" --struct "$R\politifact_struct.npz" --suite ablation --class-weight --out "$R\results_politifact.json"
python group4_experiments.py --npz "$G\gossipcop_group_feature_matrices.npz" --struct "$R\gossipcop_struct.npz" --suite ablation --class-weight --out "$R\results_gossipcop.json"
```

### 5. Interpret a saved Weibo model

Requires a model trained on `count`, `vec`, and/or `mut` blocks, plus Group 3 chunks.

```powershell
python group4_interpret.py --npz "$G\weibo_group_feature_matrices.npz" --model "$R\m_vecmut.pt" --chunks "$G" --kind weibo --n-topics 100
```

## Main Outputs

- `results\*_struct.npz` — 11 structural features per topic/slice.
- `results\m_*.pt` — saved model checkpoint and normalization statistics.
- `results\r_*.json` — training metrics and run configuration.
- `results\results_*.json` — experiment results and significance comparisons.

## Defaults and Notes

The default configuration recorded for this implementation is hidden size 128, 2 recurrent layers, dropout 0.5, FC size 256, learning rate 0.004, batch size 32, and 32 epochs. `group4_train.py` supports `gru`, `lstm`, `transformer`, and `mlp` units. Training runs use multiple seeds and report validation/test metrics including accuracy, precision, recall, F1, macro-F1, and AUC.

Feature blocks include `count`, `vec`, `mut`, `flat`, `struct`, and `size`. The `mut` block has no information for datasets whose `stance_included` flag is false. The `early` suite requires more than one time slice and is intended for Weibo.

See `GROUP4_CODE_REFERENCE.md` for the role of each script and the CLI options.
