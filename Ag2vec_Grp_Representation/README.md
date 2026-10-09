# Group 3 — AG2Vec Group Representation and Stance

Group 3 converts Group 2 (TA-Louvain) groups into group representations and stance/mutual-influence features for Group 4 GRU training.

## Final Outputs

Generated under `Ag2vec_Grp_Representation/data_output/`:

- `weibo_group_feature_matrices.npz`
- `politifact_group_feature_matrices.npz`
- `gossipcop_group_feature_matrices.npz`

These are the three files passed to Group 4. Each contains `X`, `mask`, `gid`, `labels`, `split`, `topic_ids`, and `stance_included`.

- **Weibo:** 10 temporal slices; expected `X` shape `(n_topics, 10, 30, 67)`.
- **PolitiFact and GossipCop:** static topics; one slice.
- The 67 features are 64 AG2Vec group-vector dimensions plus 3 stance/mutual-influence dimensions.

## Requirements and Inputs

Run commands from the project root with `.venv` activated:

```powershell
cd "D:\#Project\Innovative 7th\Guided_Topic_Detection"
.\.venv\Scripts\Activate.ps1
```

Required upstream data:

- **Weibo:** `Preprocessing/data/processed/weibo_ced/topics_time_sliced.json` and `Group_Detection_Tracking/data_output/topics_groups.json`
- **PolitiFact:** processed topic, edge, profile-feature files and `Group_Detection_Tracking/data_output/politifact_topics_groups.json`
- **GossipCop:** processed topic, edge, profile-feature files and `Group_Detection_Tracking/data_output/gossipcop_topics_groups.json`

Install dependencies from the repository's `requirements.txt`. The stance encoder is `paraphrase-multilingual-MiniLM-L12-v2`; the pipeline uses CUDA when available as configured.

## Run the Complete Group 3 Pipeline

Run these commands in order from the project root.

### 1. Steps 1–3: AG2Vec-style representations for all datasets

```powershell
python Ag2vec_Grp_Representation/scripts/run_steps13_windows.py
```

This creates resumable `.pkl` chunk outputs for Weibo, PolitiFact, and GossipCop. If a run is interrupted, the runner skips existing chunk files, so delete an incomplete chunk before rerunning if necessary.

### 2. Train the stance classifier and generate Weibo stance probabilities

```powershell
python -c "import sys,os,pickle; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); from group3_data_access import WeiboData; import group3_stance as S; wb=WeiboData(); df=S.load_labeled('Ag2vec_Grp_Representation/scripts/stance_sample_labeled.csv'); embed=S.sbert_embedder(device='cuda',batch=256); X=S.features(df.comment,df.source,embed,pair=False); clf=S.fit(X,df.label.values,C=10.0); os.makedirs('Ag2vec_Grp_Representation/data_output',exist_ok=True); pickle.dump(clf,open('Ag2vec_Grp_Representation/data_output/stance_clf.pkl','wb')); S.predict_topics(wb,wb.topic_ids(),clf,embed,pair=False,save_to='Ag2vec_Grp_Representation/data_output/stance_probs_weibo.pkl'); print('Stance complete')"
```

This uses `scripts/stance_sample_labeled.csv` (300 labeled samples; classes: `support`, `oppose`, `observe`) and creates `stance_clf.pkl` and `stance_probs_weibo.pkl`.

### 3. Steps 4–6 and final export

```powershell
python Ag2vec_Grp_Representation/scripts/run_mut_export_windows.py
```

This computes Weibo stance/mutual-influence features and exports the three final `.npz` matrices.

## Important Implementation Notes

- `stance_sample_labeled.csv` contains project-specific Weibo CED samples whose stance labels were assisted by Claude; it is not a dataset released by Claude or Anthropic.
- Stance class order is `[support, oppose, observe]`.
- Empty-text interactions use `[0.1, 0.1, 0.8]`.
- Weibo uses 10 time slices. PolitiFact and GossipCop are processed as static topics.
- Some inputs/features are implementation proxies for unavailable paper variables; document these choices in research reporting.

## Group 3 Scripts

- `group3_data_access.py` — loads dataset and Group 2 outputs.
- `group3_steps1_3.py` — node vectors, influence, and attention-weighted group vectors.
- `group3_run.py` — batch/chunk runner for Steps 1–3.
- `group3_stance.py` — stance classifier and probability generation.
- `group3_stance_features.py` — stance and mutual-influence features.
- `group3_export.py` — exports final `.npz` matrices.
- `run_steps13_windows.py` — Windows runner for Steps 1–3.
- `run_mut_export_windows.py` — runs Steps 4–6 and exports final matrices.

## Handoff to Group 4

Group 3 is ready for Group 4 when all three files exist:

```text
Ag2vec_Grp_Representation/data_output/weibo_group_feature_matrices.npz
Ag2vec_Grp_Representation/data_output/politifact_group_feature_matrices.npz
Ag2vec_Grp_Representation/data_output/gossipcop_group_feature_matrices.npz
```

These are the final Group 3 inputs for GRU training. Intermediate `.pkl` files are used to generate them.
