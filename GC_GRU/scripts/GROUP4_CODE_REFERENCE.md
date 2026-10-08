# Group 4 Code Reference

## group4_common.py
| Item | Purpose |
|---|---|
| `load_data(npz, struct=None)` | loads npz into dict; checks struct file topic order |
| `load_chunks(dir, kind)` | merges `<kind>_*.pkl` Group 3 chunks into {topic_id: [slice dicts]} |
| `pool_blocks(X, mask)` | masked pooling over group slots -> count (1), vec (65), mut (9) |
| `build_features_arrays(X, mask, blocks, struct)` | concatenates blocks into (N,T,F) |
| `fit_norm(F, split)` | train-only mean/std |
| `GCGRU(in_dim, hidden, layers, dropout, fc, unit)` | GRU/LSTM/Transformer/MLP + FC head; 2 logits |
| `train_eval(...)` | trains one seed, picks best epoch on val AUC, returns val/test metrics |
| `run_seeds(...)` | normalise (train only), optional truncation to first T slices, loop over seeds |
| `metrics`, `summarize`, `fmt` | acc, prec/rec/F1 (guided), macro-F1, AUC; mean±std |

### Feature blocks
count = log1p(#groups); vec = mean group vector + mean norm; mut = mean/max/std of Mut_sup/opo/none;
flat = full masked K x 67 slots; struct = 11 invariant features from chunks; size = 5 struct columns (log groups, log users,
log mean size, max-size share, singleton share), used as the trivial structural baseline.

## group4_build_struct.py
Args: --npz --chunks --kind {weibo,fnn} --out. Output npz: S (N,T,11), names, topic_ids (aligned to the feature npz).
Struct columns: log_n_groups, log_users, log_mean_size, max_size_share, singleton_share, mean_attn_entropy, mean_max_attn,
mean_influence, cos_prev_mean (same group vs previous slice), new_group_frac, dissolved_frac.

## group4_train.py
Args: --npz --struct --blocks --unit --seeds --T --epochs --lr --batch --hidden --layers --dropout --fc --class-weight --save-model --out.
Prints per-seed and mean±std val/test metrics. --save-model stores the best-val seed (state, mu, sd, blocks, cfg).

## group4_experiments.py
--suite ablation: size, count, vec, mut, vec+mut (paper-style), struct, struct+mut, vec+struct+mut, flat.
--suite temporal: gru, lstm, transformer, mlp on --blocks.   --suite early: T=2,4,6,8,Tmax on --blocks.
Prints metrics per config and significance vs reference (--ref, default vec+mut): paired t-test (acc, F1) and paired bootstrap on accuracy.
Writes results_<suite>.json (or --out). Configs needing struct/stance are skipped automatically when unavailable.

## group4_interpret.py
Args: --npz --model --chunks --kind --n-topics --tstar --across --seed.
Reports Spearman(dP, |SDS|), (dP, max attention), (dP, group size); a Table XI-style top-3 table for the most confident guided
test topic; and a Fig. 14-style stance perturbation (Mut_sup*gamma, Mut_opo/gamma, gamma in 0.8/1.0/1.2).

## Metric definitions
dP(g,t) = P_full - P(group g removed at slice t) (Eq. 42). SDS = Mut_sup - Mut_opo (Eq. 44). abar = mean attention = 1/size.

## Troubleshooting
- "struct file topic_ids do not match": rebuild the struct npz from the same feature npz.
- Out of memory with `flat` or Weibo pooling: reduce `chunk` in pool_blocks or skip `flat`.
- AUC nan: a split has a single class; check split files.
- Slow on CPU: use --seeds 3, or run on Colab GPU (the code selects CUDA automatically).