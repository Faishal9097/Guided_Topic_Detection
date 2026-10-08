# Group 4: GC-GRU Detection, Ablations, Early Detection, Interpretability

Reproduction of Sections IV-C (Eq. 40-44) and V of Wang et al., IEEE TCSS 2026.
Input: Group 3's `*_group_feature_matrices.npz` (+ optional chunk files). Output: metrics JSONs and console tables.

## Layout
Guided_Topic_Detection/
  Ag2vec_Grp_Representation/data_output/   (Group 3 outputs: 3 npz + <dataset>_all_NNNN.pkl chunks; npz and chunks share the folder)
  Group4_Detection/scripts/                (the 5 py files)
  Group4_Detection/results/                (struct npz, models, result JSONs)

## Install
pip install torch scikit-learn scipy numpy

## npz contract (from Group 3)
X (N,T,K,67) float32 = 64 group-vector dims + Mut_sup, Mut_opo, Mut_none; mask (N,T,K); gid (N,T,K);
labels (guided=1); split (train/val/test); topic_ids; stance_included.
Weibo T=10; FakeNewsNet T=1 and stance_included=False.

## Run order
1. group4_build_struct.py (per dataset)  2. smoke test  3. group4_train.py  4. group4_experiments.py (ablation, temporal, early)
5. group4_interpret.py  6. FakeNewsNet runs (with --class-weight). Commands are in the chat message / CODE_REFERENCE.
--kind is weibo | politifact | gossipcop; --chunks points to the same data_output folder as the npz.
Always quote Windows paths (the '#' in the folder name is a comment character in PowerShell).

## Design decisions
- Input is POOLED over group slots (masked mean/max/std) instead of flattening 30x67: 2,639 training topics make a 2,010-dim input overfit. `flat` is kept as an ablation.
- Normalisation statistics come from the train split only.
- Paper hyperparameters (GC-GRU_B): hidden 128, 2 layers, dropout 0.5, FC 256, Adam lr 0.004, batch 32, 32 epochs. The best epoch is chosen on validation AUC (the paper does not describe checkpoint selection); test is reported at that epoch.
- Metrics: accuracy, precision/recall/F1 (guided = positive), macro-F1, AUC. Multiple seeds, mean ± std.
- Rotation-invariant `struct` features exist because the 64 embedding dims are not comparable across topics (each topic trains its own Node2vec).

## Deviations from the paper (state in the report)
- Temporal-unit comparison uses GRU vs LSTM vs Transformer vs MLP; TGN/TGT/GATv2 (Table IV) are NOT implemented. The 8 rival detectors of Tables VII-VIII are not reproduced.
- Eq. 42 group contribution uses removal from the pooled input, not zeroing.
- Eq. 43 mean attention is identically 1/|group| (softmax weights sum to 1); max attention is reported as well.
- Significance: paired t-test over seeds (same test set, so optimistic) plus paired bootstrap on test predictions.
- Interpretation works only for models on count/vec/mut blocks.

## Caveats
- Labels are computed proxies derived from the same interactions as the features; accuracy may be inflated (include the `size`/`count` baselines in the report).
- FakeNewsNet: T=1, no stance, graphs static; results are not comparable to the paper's.
- Early detection truncates cumulative slices, which is causal for features, but labels use whole-thread signals.
- Frozen random attention vector q and per-topic embeddings are Group 3 limitations.