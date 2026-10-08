# Group 3 --- AG2Vec Group Representation and Stance

This directory implements **Group 3** of the project:

> **A Guided Topic Detection Model Based on Topic Evolution and Group
> Stance**

Group 3 converts the groups detected by Group 2 (TA-Louvain) into
group-level representations and stance/mutual-influence features that
are handed to Group 4 for GRU training.

------------------------------------------------------------------------

## 1. What Group 3 Produces

For the three datasets, the final hand-off to GRU training is:

``` text
weibo_group_feature_matrices.npz
politifact_group_feature_matrices.npz
gossipcop_group_feature_matrices.npz
```

The final `.npz` contains:

``` text
X
mask
gid
labels
split
topic_ids
stance_included
```

For the Weibo representation:

``` text
X shape = (n_topics, 10, 30, 67)
```

where:

-   `10` = temporal slices
-   `30` = maximum group slots
-   `64` = AG2Vec group-vector dimensions
-   `3` = stance/mutual-influence dimensions
-   `67` = `64 + 3`

The export implementation creates these arrays and writes them using
`np.savez_compressed`. `stance_included` is `False` when no stance
dictionary is supplied and `True` when stance vectors are included.

------------------------------------------------------------------------

# 2. Directory Structure

Expected Group 3 structure:

``` text
Ag2vec_Grp_Representation/
├── scripts/
│   ├── group3_data_access.py
│   ├── group3_steps1_3.py
│   ├── group3_run.py
│   ├── group3_stance.py
│   ├── group3_stance_features.py
│   ├── group3_export.py
│   └── stance_sample_labeled.csv
│
└── data_output/
    ├── stance_clf.pkl
    ├── stance_probs_weibo.pkl
    ├── group3_out/
    │   ├── weibo_all_0000.pkl
    │   ├── weibo_all_0001.pkl
    │   └── ...
    ├── stance_features_weibo.pkl
    ├── weibo_group_feature_matrices.npz
    ├── politifact_group_feature_matrices.npz
    └── gossipcop_group_feature_matrices.npz
```

The exact intermediate filename for the Step 4--6 serialized feature
dictionary is implementation-dependent; the important object passed to
`group3_export.export()` is the `stance` dictionary:

``` python
{
    (topic_id, slice_position, group_id): np.ndarray(shape=(3,))
}
```

------------------------------------------------------------------------

# 3. Prerequisites

Run all commands from the **project root**:

``` text
D:\#Project\Innovative 7th\Guided_Topic_Detection
```

Activate the virtual environment:

``` powershell
.\.venv\Scripts\Activate.ps1
```

Required Python packages include:

``` text
numpy
pandas
networkx
scikit-learn
requests
gdown
tqdm
jupyter
ipykernel
python-dateutil
gensim
torch
transformers
sentence-transformers
huggingface-hub
scipy
```

The working ML environment used during implementation includes:

``` text
torch==2.5.1+cu121
transformers==4.57.3
sentence-transformers==5.1.2
huggingface-hub==1.33.0
```

The SentenceTransformer model used for stance detection is:

``` text
paraphrase-multilingual-MiniLM-L12-v2
```

------------------------------------------------------------------------

# 4. Upstream Inputs

Group 3 depends on the outputs of the earlier preprocessing and Group 2
stages.

## Weibo

Required:

``` text
Preprocessing/data/processed/weibo_ced/topics_time_sliced.json
Group_Detection_Tracking/data_output/topics_groups.json
```

`WeiboData` loads the time-sliced topics and the TA-Louvain group
output.

The Weibo pipeline uses **10 temporal slices per topic**.

------------------------------------------------------------------------

## PolitiFact

Required:

``` text
Preprocessing/data/processed/fakenewsnet/politifact_topics_with_categories.json
Preprocessing/data/processed/fakenewsnet/politifact_edges_cleaned.txt
Preprocessing/data/processed/fakenewsnet/politifact_new_profile_feature.npz
Group_Detection_Tracking/data_output/politifact_topics_groups.json
```

------------------------------------------------------------------------

## GossipCop

Required:

``` text
Preprocessing/data/processed/fakenewsnet/gossipcop_topics_with_categories.json
Preprocessing/data/processed/fakenewsnet/gossipcop_edges_cleaned.txt
Preprocessing/data/processed/fakenewsnet/gossipcop_new_profile_feature.npz
Group_Detection_Tracking/data_output/gossipcop_topics_groups.json
```

PolitiFact and GossipCop are handled as **static topics** in Group 3.

------------------------------------------------------------------------

# 5. Group 3 Pipeline

The overall flow is:

``` text
Group 2 TA-Louvain
        │
        ▼
Group membership / group tracking
        │
        ▼
Steps 1–3
AG2Vec-style user vectors
+ influence
+ attention
+ group vectors
        │
        ├─────────────────────────────┐
        ▼                             │
Weibo stance classifier               │
        │                             │
        ▼                             │
stance_probs_weibo.pkl                │
        │                             │
        ▼                             │
Steps 4–6                             │
soft stance → group stance            │
→ internal/external factors           │
→ linear stance influence             │
→ Mutual Influence Mut(g,t)          │
        │                             │
        └──────────────┬──────────────┘
                       ▼
                Final Export
                       │
                       ▼
                 .npz matrices
                       │
                       ▼
                   GRU training
```

------------------------------------------------------------------------

# 6. Stance Annotation Source

## `stance_sample_labeled.csv`

Location:

``` text
Ag2vec_Grp_Representation/scripts/stance_sample_labeled.csv
```

This is a project-specific labeled dataset containing **300 hand-labeled
comments**.

The three stance classes are:

``` text
support
oppose
observe
```

The class order used throughout the implementation is:

``` text
[support, oppose, observe]
```

### Source / provenance

The underlying comments and source information come from the processed
**Weibo CED dataset**.

The stance annotation was assisted by **Claude (Anthropic)**. Claude was
used as an annotation/labeling aid for the sampled interactions.

This file is therefore **not an original dataset released by Claude or
Anthropic**. It is a project-specific annotation artifact created from
sampled Weibo interactions.

Pipeline:

``` text
Weibo CED
   ↓
Sampled Weibo interactions
   ↓
Stance annotation assisted by Claude
   ↓
stance_sample_labeled.csv
   ↓
Stance classifier
```

Any future change to the labeled samples or labels can change the
trained classifier and all downstream stance features.

------------------------------------------------------------------------

# 7. Stance Classifier

The stance implementation is in:

``` text
Ag2vec_Grp_Representation/scripts/group3_stance.py
```

It uses multilingual sentence embeddings from:

``` text
paraphrase-multilingual-MiniLM-L12-v2
```

and logistic regression with balanced class weights.

The labeled data contains:

``` text
300 samples
```

The classifier is trained using:

``` python
C=10.0
```

The classifier output is ordered as:

``` text
[support, oppose, observe]
```

For each interaction, the predicted probability vector is used as the
soft stance vector.

The confidence is:

``` text
alpha_j = max(p_j)
```

------------------------------------------------------------------------

# 8. Train the Stance Classifier

From the project root:

``` powershell
python -c "import sys,os,pickle; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); from group3_data_access import WeiboData; import group3_stance as S; wb=S.WeiboData() if False else WeiboData(); print('Weibo topics:',len(wb.topic_ids())); df=S.load_labeled('Ag2vec_Grp_Representation/scripts/stance_sample_labeled.csv'); print('Labeled samples:',len(df)); embed=S.sbert_embedder(device='cuda',batch=256); X=S.features(df.comment,df.source,embed,pair=False); clf=S.fit(X,df.label.values,C=10.0); os.makedirs('Ag2vec_Grp_Representation/data_output',exist_ok=True); pickle.dump(clf,open('Ag2vec_Grp_Representation/data_output/stance_clf.pkl','wb')); print('stance_clf.pkl saved')"
```

Expected important output:

``` text
Weibo topics: 3300
Labeled samples: 300
stance_clf.pkl saved
```

Output:

``` text
Ag2vec_Grp_Representation/data_output/stance_clf.pkl
```

------------------------------------------------------------------------

# 9. SentenceTransformer Model

If the model has already been downloaded and cached, it can be loaded
locally.

The local snapshot used during implementation was:

``` text
C:\Users\anisu\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42
```

To verify the local model:

``` powershell
python -c "from sentence_transformers import SentenceTransformer; p=r'C:\Users\anisu\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42'; m=SentenceTransformer(p,device='cuda',local_files_only=True); print('LOCAL MODEL LOADED')"
```

Expected:

``` text
LOCAL MODEL LOADED
```

If the model is already cached, using the local snapshot avoids
unnecessary Hugging Face network checks.

------------------------------------------------------------------------

# 10. Generate Weibo Stance Probabilities

If `stance_clf.pkl` already exists, **do not retrain the classifier**.

Use the local SentenceTransformer snapshot:

``` powershell
python -c "import sys,pickle; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); from group3_data_access import WeiboData; import group3_stance as S; wb=WeiboData(); clf=pickle.load(open('Ag2vec_Grp_Representation/data_output/stance_clf.pkl','rb')); p=r'C:\Users\anisu\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42'; embed=S.sbert_embedder(model_name=p,device='cuda',batch=256); print('Weibo topics:',len(wb.topic_ids())); S.predict_topics(wb,wb.topic_ids(),clf,embed,pair=False,save_to='Ag2vec_Grp_Representation/data_output/stance_probs_weibo.pkl'); print('stance_probs_weibo.pkl saved'); print('Stance: DONE')"
```

Expected:

``` text
Weibo topics: 3300
stance_probs_weibo.pkl saved
Stance: DONE
```

Output:

``` text
Ag2vec_Grp_Representation/data_output/stance_probs_weibo.pkl
```

The output has the conceptual structure:

``` python
{
    topic_id: {
        mid: np.array([
            support_probability,
            oppose_probability,
            observe_probability
        ])
    }
}
```

Empty-text interactions receive:

``` text
[0.1, 0.1, 0.8]
```

This is an implementation assumption and should be stated as such in the
report.

------------------------------------------------------------------------

# 11. Steps 1--3 --- AG2Vec Group Representation

Implementation:

``` text
group3_steps1_3.py
group3_run.py
```

Steps 1--3 implement:

1.  User node representations
2.  Influence
3.  Attention-weighted group representations

The group vector dimension is:

``` text
64
```

The configuration defaults include:

``` text
dim          = 64
num_walks    = 8
walk_len     = 15
window       = 5
walk_p       = 1.0
walk_q       = 1.0
epochs_first = 5
epochs_update= 3
attr_knn     = 10
attr_mass    = 0.4
beta         = 0.4
omega        = (0.2, 0.5, 0.3)
seed         = 42
```

The implementation uses warm-started Word2Vec across Weibo slices so
that user vectors remain in a common coordinate space across slices.

------------------------------------------------------------------------

# 12. Run Steps 1--3 for Weibo

`group3_run.py` is a batch runner and saves compact chunk files.

A documented usage pattern is:

``` python
import sys
sys.path.insert(0, 'Ag2vec_Grp_Representation/scripts')

import group3_run as G
from group3_steps1_3 import Cfg
from group3_data_access import WeiboData

wb = WeiboData()

G.run(
    wb,
    'weibo',
    'all',
    'Ag2vec_Grp_Representation/data_output/group3_out',
    Cfg()
)
```

Run it directly from PowerShell:

``` powershell
python -c "import sys; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); import group3_run as G; from group3_steps1_3 import Cfg; from group3_data_access import WeiboData; wb=WeiboData(); G.run(wb,'weibo','all','Ag2vec_Grp_Representation/data_output/group3_out',Cfg())"
```

For a quick test:

``` powershell
python -c "import sys; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); import group3_run as G; from group3_steps1_3 import Cfg; from group3_data_access import WeiboData; wb=WeiboData(); G.run(wb,'weibo','all','Ag2vec_Grp_Representation/data_output/group3_out',Cfg(),limit=20)"
```

The runner saves:

``` text
weibo_all_0000.pkl
weibo_all_0001.pkl
...
```

Each chunk contains:

``` text
cfg
topics
errors
```

and each topic contains per-slice:

``` text
slice_index
n_users
reply_edge_cover
degenerate_frac
groups
influence_users
influence
```

The runner is resumable: completed chunk files are skipped when the same
command is run again.

------------------------------------------------------------------------

# 13. Steps 1--3 Progress

The runner prints progress after each chunk:

``` text
chunk 0: 50 ok, 0 errors, 120s (2.4s/topic) | total 120s
chunk 1: 50 ok, 0 errors, 118s (2.4s/topic) | total 238s
...
```

At completion:

``` text
done: 3300 topics -> Ag2vec_Grp_Representation/data_output/group3_out
```

On Windows, the runner does not use `fork`, so if `workers > 1` is
supplied it falls back to one process. The implementation explicitly
reports this behavior.

------------------------------------------------------------------------

# 14. Steps 4--6 --- Stance and Mutual Influence

Implementation:

``` text
group3_stance_features.py
```

Inputs:

``` text
WeiboData
Steps 1–3 output
stance_probs_weibo.pkl
```

The steps are:

``` text
soft stance
   ↓
group stance proportions
   ↓
internal/external factors
   ↓
linear stance influence
   ↓
Mut(g,t)
```

The implementation uses:

``` text
OMEGA = (0.5, 0.5)
K3 = 3
```

The three stance dimensions are:

``` text
Mut_sup
Mut_opo
Mut_none
```

------------------------------------------------------------------------

# 15. Steps 4--6 Computation

The documented usage is:

``` python
import group3_stance_features as F

feats = {
    tid: F.topic_features(
        tid,
        wb,
        steps[tid],
        probs[tid]
    )
    for tid in steps
}

train = [
    t for t in feats
    if wb.split_of(t) == 'train'
]

rho = F.fit_rho(feats, train)

out = F.mut_vectors(feats, rho)

stance = {
    k: v["mut"]
    for k, v in out.items()
}
```

The regression is fitted **only on train topics**.

The target is the confidence-weighted stance vector of the same
persistent group in the next slice.

The fitted coefficients are:

``` text
rho0
rho1
rho2
```

and the code reports:

``` text
rho fitted on <N> rows:
rho0=...
rho1=...
rho2=...
R2=...
```

------------------------------------------------------------------------

# 16. Step 4--6 Proxies Used for Weibo

Because Weibo CED does not contain all variables used by the original
paper, the implementation uses documented proxies.

### Stance probability

The classifier's probability vector replaces the paper's label-smoothed
stance representation.

### Confidence

``` text
alpha_j = max(p_j)
```

### TopicAware

Implemented as:

``` text
group user's interaction count / most active user's interaction count
```

### Activity

Implemented as the share of the group's cumulative interactions that are
pure reposts / empty text.

``` text
Act(g) = N_r / (N_r + N_ori)
```

### Topic heat

``` text
log1p(cumulative interaction count)
```

### Influence

The Step 2 influence proxy is used.

### Regression target

The implementation predicts the group's stance in the next slice.

These are implementation choices and should be clearly documented in the
research report.

------------------------------------------------------------------------

# 17. Final Export

Implementation:

``` text
group3_export.py
```

The export function takes:

``` python
E.export(
    topics,
    E.meta_from(wb),
    'weibo_group_feature_matrices.npz',
    T=10,
    K=30,
    slot_mode='stable',
    stance=stance
)
```

The recommended slot mode is:

``` text
stable
```

A stable slot allows a persistent group to keep the same slot across the
slices where it exists, allowing the GRU to follow the group over time.

The export creates:

``` text
X
mask
gid
labels
split
topic_ids
stance_included
```

------------------------------------------------------------------------

# 18. Export Final Weibo Matrix

After Steps 1--3 and Steps 4--6 have completed:

``` powershell
python -c "import sys; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); import group3_run as G, group3_export as E; from group3_data_access import WeiboData; wb=WeiboData(); topics=G.load_all('Ag2vec_Grp_Representation/data_output/group3_out','weibo','all'); print('Loaded topics:',len(topics)); print('NOTE: pass the computed stance dictionary from Steps 4-6 to E.export(..., stance=stance)');"
```

The final export call is:

``` python
E.export(
    topics,
    E.meta_from(wb),
    'Ag2vec_Grp_Representation/data_output/weibo_group_feature_matrices.npz',
    T=10,
    K=30,
    slot_mode='stable',
    stance=stance
)
```

The resulting file is:

``` text
Ag2vec_Grp_Representation/data_output/weibo_group_feature_matrices.npz
```

`stance_included` should be:

``` text
True
```

when the `stance` dictionary is supplied.

------------------------------------------------------------------------

# 19. PolitiFact --- Steps 1--3

Load the dataset:

``` python
from group3_data_access import FakeNewsNetData

fn = FakeNewsNetData('politifact')
```

Run:

``` powershell
python -c "import sys; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); import group3_run as G; from group3_steps1_3 import Cfg; from group3_data_access import FakeNewsNetData; fn=FakeNewsNetData('politifact'); G.run(fn,'fnn','all','Ag2vec_Grp_Representation/data_output/politifact_group3_out',Cfg())"
```

Output:

``` text
Ag2vec_Grp_Representation/data_output/politifact_group3_out/
```

with chunk files such as:

``` text
fnn_all_0000.pkl
fnn_all_0001.pkl
...
```

------------------------------------------------------------------------

# 20. GossipCop --- Steps 1--3

Run:

``` powershell
python -c "import sys; sys.path.insert(0,'Ag2vec_Grp_Representation/scripts'); import group3_run as G; from group3_steps1_3 import Cfg; from group3_data_access import FakeNewsNetData; fn=FakeNewsNetData('gossipcop'); G.run(fn,'fnn','all','Ag2vec_Grp_Representation/data_output/gossipcop_group3_out',Cfg())"
```

Output:

``` text
Ag2vec_Grp_Representation/data_output/gossipcop_group3_out/
```

with chunk files.

------------------------------------------------------------------------

# 21. PolitiFact and GossipCop Stance

The provided Group 3 stance pipeline is specifically documented around
**Weibo CED**.

The stance classifier:

``` text
stance_sample_labeled.csv
stance_clf.pkl
stance_probs_weibo.pkl
```

is therefore a **Weibo stance pipeline**.

Do not automatically assume that the same `stance_probs_weibo.pkl` is
applicable to PolitiFact or GossipCop.

For FakeNewsNet, `FakeNewsNetData` provides static topic views, and
Steps 1--3 produce static group vectors for each topic.

------------------------------------------------------------------------

# 22. PolitiFact Final Export

Load the completed Steps 1--3 chunks:

``` python
import group3_run as G
import group3_export as E
from group3_data_access import FakeNewsNetData

fn = FakeNewsNetData('politifact')

topics = G.load_all(
    'Ag2vec_Grp_Representation/data_output/politifact_group3_out',
    'fnn',
    'all'
)

E.export(
    topics,
    E.meta_from(fn),
    'Ag2vec_Grp_Representation/data_output/politifact_group_feature_matrices.npz',
    T=1,
    K=30,
    slot_mode='stable'
)
```

The final file is:

``` text
Ag2vec_Grp_Representation/data_output/politifact_group_feature_matrices.npz
```

Because FakeNewsNet is static in this implementation, it has one slice
rather than Weibo's 10 temporal slices.

------------------------------------------------------------------------

# 23. GossipCop Final Export

``` python
import group3_run as G
import group3_export as E
from group3_data_access import FakeNewsNetData

fn = FakeNewsNetData('gossipcop')

topics = G.load_all(
    'Ag2vec_Grp_Representation/data_output/gossipcop_group3_out',
    'fnn',
    'all'
)

E.export(
    topics,
    E.meta_from(fn),
    'Ag2vec_Grp_Representation/data_output/gossipcop_group_feature_matrices.npz',
    T=1,
    K=30,
    slot_mode='stable'
)
```

Final file:

``` text
Ag2vec_Grp_Representation/data_output/gossipcop_group_feature_matrices.npz
```

------------------------------------------------------------------------

# 24. Final Three Files for GRU

After the complete Group 3 pipeline, the three dataset-level final files
are:

``` text
Ag2vec_Grp_Representation/data_output/
├── weibo_group_feature_matrices.npz
├── politifact_group_feature_matrices.npz
└── gossipcop_group_feature_matrices.npz
```

These are the final Group 3 hand-offs to Group 4 / GRU training.

------------------------------------------------------------------------

# 25. Final Weibo Matrix Contents

For Weibo:

``` text
X
shape = (n_topics, 10, 30, 67)
```

Interpretation:

``` text
topic
  └── 10 temporal slices
       └── up to 30 group slots
            └── 67 features
```

The 67 features are:

``` text
64 AG2Vec group-vector dimensions
+
3 mutual-influence / stance dimensions
```

Additional arrays:

``` text
mask
```

indicates whether a group slot is real or padding.

``` text
gid
```

contains the persistent group ID, with:

``` text
-1 = padding
```

``` text
labels
```

contains the guided-label target.

``` text
split
```

contains:

``` text
train
val
test
```

``` text
topic_ids
```

contains the topic identifiers.

``` text
stance_included
```

indicates whether stance dimensions were actually inserted.

------------------------------------------------------------------------

# 26. Important Difference Between Datasets

## Weibo

``` text
10 temporal slices
↓
group evolution/tracking
↓
temporal GRU input
```

Final:

``` text
weibo_group_feature_matrices.npz
```

## PolitiFact

``` text
static topic
↓
one group representation
```

Final:

``` text
politifact_group_feature_matrices.npz
```

## GossipCop

``` text
static topic
↓
one group representation
```

Final:

``` text
gossipcop_group_feature_matrices.npz
```

------------------------------------------------------------------------

# 27. Recommended Execution Order

Run in this order:

``` text
1. Verify preprocessing outputs
        ↓
2. Verify TA-Louvain group outputs
        ↓
3. Verify Python environment
        ↓
4. Verify SentenceTransformer model
        ↓
5. Train stance classifier
        ↓
6. Generate stance_probs_weibo.pkl
        ↓
7. Run Group 3 Steps 1–3 for Weibo
        ↓
8. Run Group 3 Steps 4–6 for Weibo
        ↓
9. Export Weibo .npz
        ↓
10. Run Steps 1–3 for PolitiFact
        ↓
11. Export PolitiFact .npz
        ↓
12. Run Steps 1–3 for GossipCop
        ↓
13. Export GossipCop .npz
        ↓
14. Pass the three .npz files to Group 4 / GRU
```

------------------------------------------------------------------------

# 28. Output Checklist

Before starting Group 4, verify that these exist:

``` text
[ ] stance_clf.pkl
[ ] stance_probs_weibo.pkl

[ ] Weibo Steps 1–3 chunk files
[ ] PolitiFact Steps 1–3 chunk files
[ ] GossipCop Steps 1–3 chunk files

[ ] weibo_group_feature_matrices.npz
[ ] politifact_group_feature_matrices.npz
[ ] gossipcop_group_feature_matrices.npz
```

The most important final hand-off is:

``` text
weibo_group_feature_matrices.npz
politifact_group_feature_matrices.npz
gossipcop_group_feature_matrices.npz
```

------------------------------------------------------------------------

# 29. Troubleshooting

## SentenceTransformer repeatedly contacts Hugging Face

If the model is already cached, use the local snapshot path:

``` text
C:\Users\anisu\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42
```

Do not repeatedly retrain the classifier if `stance_clf.pkl` already
exists.

------------------------------------------------------------------------

## `torch.accelerator` error

This indicates an incompatible PyTorch / Transformers combination.

The working combination used for this project is:

``` text
torch==2.5.1+cu121
transformers==4.57.3
sentence-transformers==5.1.2
huggingface-hub==1.33.0
```

------------------------------------------------------------------------

## CUDA unavailable

Check:

``` powershell
nvidia-smi
```

and:

``` powershell
python -c "import torch; print('Torch:',torch.__version__); print('CUDA:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU count:',torch.cuda.device_count())"
```

The stance embedding stage is intended to use:

``` text
device='cuda'
```

------------------------------------------------------------------------

## Group 3 appears stuck

Steps 1--3 print progress after each completed chunk.

For stance prediction, the original implementation does not print
per-block progress. If required, a progress print can be added inside
the block loop in `predict_topics()`.

GPU activity can also be checked in another terminal:

``` powershell
nvidia-smi -l 2
```

------------------------------------------------------------------------

# 30. Research / Reporting Notes

The following are implementation choices that should be explicitly
stated in the report rather than presented as direct facts from the
original paper:

-   Weibo stance labels are `support / oppose / observe`.
-   The stance sample was annotated with Claude assistance.
-   Empty-text interactions use `[0.1, 0.1, 0.8]`.
-   Weibo influence uses available reply information as a proxy.
-   TopicAware uses interaction counts.
-   Activity uses the pure-repost/empty-text fraction.
-   Topic heat uses `log1p(cumulative interaction count)`.
-   The regression target is the same persistent group's stance in the
    next slice.
-   The attention vector `q` is frozen random in the current
    implementation.
-   Node2Vec/Walk parameters are implementation defaults where the paper
    does not specify them.
-   Weibo uses 10 temporal slices.
-   PolitiFact and GossipCop are processed statically.

These choices are part of the current implementation and should be
distinguished from parameters explicitly specified by the source paper.

------------------------------------------------------------------------

# 31. Final Handoff to Group 4

Group 3 is complete when the following three files are available:

``` text
weibo_group_feature_matrices.npz
politifact_group_feature_matrices.npz
gossipcop_group_feature_matrices.npz
```

The conceptual hand-off is:

``` text
                 GROUP 3
                    │
                    ▼
        ┌─────────────────────────┐
        │ Group feature matrices  │
        │                         │
        │ Weibo                   │
        │ PolitiFact              │
        │ GossipCop               │
        └────────────┬────────────┘
                     │
                     ▼
                  GROUP 4
                     │
                     ▼
                 GRU training
```

The `.npz` files are therefore the **final Group 3 outputs used by the
GRU stage**.
