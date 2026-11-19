# How Much Does Attention Add Beyond Graph Centrality?

Code and results for:

> S. Dumbre Patil and U. Chaskar, "How Much Does Attention Add Beyond Graph Centrality? Evidence from Self-Supervised Structural Connectome Transformers," CMES 2027.

We train small transformer encoders on HCP structural connectomes (962 subjects, 73 regions) using masked edge reconstruction. We then test whether their attention tells us anything beyond classical graph measures (degree, strength, betweenness, clustering, participation). The tests include:
- untrained controls;
- an anatomical-identity ablation;
- leakage-free held-out edge prediction;
- layer- and head-wise analysis;
- seed-to-seed stability;
- a comparison with gradient-based attribution.

## Repository layout

```
code/
  model.py                 transformer encoder, edge masking, masked loss
  train_colab.ipynb        training and attention extraction (Colab, T4 GPU)
  analysis.py              main analysis: coupling, CVs, depth, robustness  -> results/results.json
  revision_analysis.py     controls and statistics, layers/heads, saliency-map similarity,
                           leakage-free edge prediction, gradient attribution -> results/revision_results.json
  fa_length_analysis.py    pre-QC statistics and FA / fibre-length check     -> results/fa_length_results.json
  make_figures.py          paper figures -> figures/
  requirements.txt
data/
  splits.npz               train/val/test subject indices (seed 42)
  test_idx.npy             test indices
  history_*.npy            training and validation loss per epoch (10 seeds x 2 models)
  model_*.pt               trained weights (seeds 0, 3, 4) and untrained controls
results/                   all numbers reported in the paper
figures/                   figures used in the paper
```

Naming: `A` is the profile-only model ($M_P$) and `B` is the identity-embedding model ($M_{P+I}$). A file with no seed suffix is seed 0.

## Data

The connectomes are **not redistributed** here. To get them:

1. Go to [braingraph.org](https://braingraph.org) → *Download PIT group connectomes* and download the **1064 robust HCP connectomes, 83-node (scale 33) resolution**, as GraphML (Varga & Grolmusz, 2021).
2. Unzip into `data/graphml/`.

Use of the data is subject to the [WU-Minn HCP Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).

The processed connectome file (`scale33_normalized.npz`) and the attention tensors (`attn_*.npy`, about 1.4 GB) are too large for GitHub or are derived from HCP data. They are produced by the notebook.

## Reproducing the results

```bash
pip install -r code/requirements.txt

# 1. train models and extract attention (GPU; about 5 h on a T4)
#    open code/train_colab.ipynb in Colab and set D to your data folder

# 2. analyses (CPU)
python code/analysis.py            # about 3 min
python code/revision_analysis.py   # about 20 min
python code/fa_length_analysis.py  # needs data/graphml/

# 3. figures
python code/make_figures.py
```

## Citation

```bibtex
@inproceedings{dumbrepatil2027attention,
  title     = {How Much Does Attention Add Beyond Graph Centrality? Evidence from
               Self-Supervised Structural Connectome Transformers},
  author    = {Dumbre Patil, Sharanya and Chaskar, Uttam},
  booktitle = {CMES 2027},
  year      = {2027}
}
```

## Licence

MIT. See `LICENSE`.
