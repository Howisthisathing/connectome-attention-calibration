"""
analysis.py  --  reproduces the numbers in
"How Much Does Attention Add Beyond Graph Centrality? Evidence from
 Self-Supervised Structural Connectome Transformers"

Needs only the saved outputs of the Colab training notebook (no GPU / torch):
    data/scale33_normalized.npz   X_raw, X_log, node_names, coords, ...  (962, 73, 73)
    data/splits.npz               train / val / test indices (seed 42)
    data/attn_{A,B}[_seedK].npy   (145, L, H, 73, 73)  A = profile-only, B = +identity
    data/attn_ctrl{A,B}[_seedK].npy  untrained controls
    data/attn_depth{L}_seed{K}.npy   depth sweep (profile-only)
    data/history_{A,B}[_seedK].npy   (300, 2) = (train loss, val loss)

File convention: the un-suffixed file is seed 0  (attn_A.npy == attn_depth2_seed0.npy).

Conventions recovered by matching the paper exactly:
  * received-attention saliency: column sum, averaged over layers and heads (Eq. 4)
  * graph measures on log(1+w) weights; betweenness on lengths 1/w;
    clustering = bct.clustering_coef_wu on W / max(W)
  * per-subject Spearman over non-isolated nodes
  * Table III / Fig 1-2 / depth table: subject-average via Fisher z
    (tanh(mean(arctanh(rho)))), then mean +- SD (ddof=0) over seeds
  * trained: seeds 0-9 ; untrained controls: seeds 0-4 (paper: "5 controls")
  * Sec IV-D text values (-0.180 / -0.171): plain mean over subjects
  * own- vs cohort-degree (Sec IV-G): plain mean, cohort = test-set mean degree
    (original submission used seeds 0-4; revision uses all 10)
  * node-mask sensitivity: seed 0, plain mean
  * strength-CV 29.6 %: mean of the two per-variant CVs

Usage:   python analysis.py [data_dir] [out_dir]
"""
import sys, os, json, glob, warnings
import numpy as np
import networkx as nx
import bct
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import adjusted_mutual_info_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

METRICS = ["degree", "strength", "betweenness", "clustering", "participation"]
GAMMA = 1.5
LOUVAIN_SEED = 3   # group-mean Louvain seed that yields the published partition
                   # (four bilateral modules + Brain-Stem singleton, sizes 20/16/1/20/16)
# Depth-sweep validation losses: only printed in the notebook log (histories were not saved)
DEPTH_VAL_LOSS = {2: [0.4260, 0.4292, 0.4231],
                  4: [0.3875, 0.3847, 0.3869],
                  6: [0.3757, 0.3745, 0.3791]}


# ------------------------------------------------------------------ helpers
def fname(prefix, seed):
    return f"{D}/{prefix}.npy" if seed == 0 else f"{D}/{prefix}_seed{seed}.npy"


def fisher_mean(r):
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return float(np.tanh(np.nanmean(np.arctanh(r))))


def received_attention(A):
    """(S, L, H, N, N) row-stochastic -> (S, N): sum over rows j of A[j, i], mean over L, H."""
    return np.asarray(A, dtype=np.float32).sum(axis=3).mean(axis=(1, 2))


_SAL = {}
def saliency(prefix, seed):
    key = (prefix, seed)
    if key not in _SAL:
        _SAL[key] = received_attention(np.load(fname(prefix, seed), mmap_mode="r"))
    return _SAL[key]


def per_subject_rho(a, x, mask):
    return np.array([spearmanr(a[s][mask[s]], x[s][mask[s]])[0] for s in range(len(a))])


# ------------------------------------------------------------------ graph measures
def group_partition(X_log, names):
    M = X_log.mean(0).astype(float); np.fill_diagonal(M, 0)
    G = nx.from_numpy_array(M)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=GAMMA,
                                             seed=LOUVAIN_SEED)
    # order as in Fig. 4: right-hemisphere modules, singleton (Brain-Stem), left-hemisphere
    hemi = lambda c: np.mean([str(names[i]).lower().startswith(("lh", "left")) for i in c])
    comms = sorted(comms, key=lambda c: (1 if len(c) == 1 else 2 * hemi(c), min(c)))
    ci = np.zeros(len(M), int)
    for k, c in enumerate(comms):
        ci[list(c)] = k + 1
    return ci, nx.community.modularity(G, comms, weight="weight")


def node_measures(W, ci):
    W = W.astype(float).copy(); np.fill_diagonal(W, 0)
    deg = (W > 0).sum(1).astype(float)
    strength = W.sum(1)
    L = np.where(W > 0, 1.0 / np.where(W > 0, W, 1.0), 0.0)
    btw = bct.betweenness_wei(L) / ((len(W) - 1) * (len(W) - 2) / 2)   # normalised
    clus = bct.clustering_coef_wu(W / W.max())
    part = bct.participation_coef(W, ci)
    return np.stack([deg, strength, btw, clus, part])


# ------------------------------------------------------------------ edge-prediction AUC
def incremental_auc(a, Xte, topo, seed=0):
    """RECONSTRUCTED protocol (the original code is lost; see README):
    predict presence of edge (i,j) from symmetric pair features (sum, product, min, max)
    of endpoint node features; fit on half the test subjects, evaluate on the other half."""
    S, N = a.shape
    iu = np.triu_indices(N, 1)
    rng = np.random.default_rng(seed); perm = rng.permutation(S)
    tr, te = perm[: S // 2], perm[S // 2:]
    F_topo = topo.transpose(0, 2, 1)                     # (S, N, 5)
    F_att = a[..., None]

    def feats(subj, F):
        fi, fj = F[subj][:, iu[0]], F[subj][:, iu[1]]
        fi = fi.reshape(-1, F.shape[-1]); fj = fj.reshape(-1, F.shape[-1])
        return np.hstack([fi + fj, fi * fj, np.minimum(fi, fj), np.maximum(fi, fj)])
    y = lambda subj: (Xte[subj][:, iu[0], iu[1]] > 0).ravel().astype(int)

    out = {}
    for kind in ("logistic", "gradient_boosting"):
        res = {}
        for name, F in [("topology", F_topo), ("attention", F_att),
                         ("topology+attention", np.concatenate([F_topo, F_att], -1))]:
            Xtr, Xts = feats(tr, F), feats(te, F)
            if kind == "logistic":
                sc = StandardScaler().fit(Xtr)
                p = LogisticRegression(max_iter=3000).fit(sc.transform(Xtr), y(tr)) \
                    .predict_proba(sc.transform(Xts))[:, 1]
            else:
                p = HistGradientBoostingClassifier(random_state=0).fit(Xtr, y(tr)) \
                    .predict_proba(Xts)[:, 1]
            res[name] = float(roc_auc_score(y(te), p))
        res["delta"] = res["topology+attention"] - res["topology"]
        out[kind] = res
    return out


# ------------------------------------------------------------------ main
def main():
    d = np.load(f"{D}/scale33_normalized.npz"); sp = np.load(f"{D}/splits.npz")
    X_raw, X_log, names = d["X_raw"], d["X_log"], d["node_names"]
    te = sp["test"]; Xte = X_log[te].astype(float)
    S, N = len(te), X_log.shape[1]
    R = {}

    # ---------------- cohort / QC (Table I quantities checkable from the saved data)
    nz = X_raw[X_raw > 0]; nzl = X_log[X_log > 0]
    R["qc"] = dict(
        n_subjects=int(len(X_raw)), n_nodes=int(N),
        split=[int(len(sp["train"])), int(len(sp["val"])), int(len(sp["test"]))],
        final_density=float(((X_raw > 0).sum((1, 2)) / (N * (N - 1))).mean()),
        residual_isolated_pct=float(((X_raw > 0).sum(2) == 0).mean() * 100),
        mean_median_ratio_raw=float(nz.mean() / np.median(nz)),
        mean_median_ratio_log=float(nzl.mean() / np.median(nzl)))

    # ---------------- communities
    ci, Q = group_partition(X_log, names)
    np.save(f"{OUT}/partition_gamma1.5.npy", ci)
    R["partition"] = dict(sizes=np.bincount(ci)[1:].tolist(), modularity_Q_groupmean=Q,
                          singletons=[str(names[i]) for i in range(N)
                                      if np.sum(ci == ci[i]) == 1])
    # resolution sweep on the group-mean connectome (Fig. 5 left)
    M = X_log.mean(0).astype(float); np.fill_diagonal(M, 0); G = nx.from_numpy_array(M)
    sweep = []
    for g in (1.0, 1.5, 2.0, 2.5, 3.0):
        best = max((nx.community.louvain_communities(G, weight="weight", resolution=g, seed=s)
                    for s in range(20)),
                   key=lambda c: nx.community.modularity(G, c, weight="weight", resolution=g))
        sweep.append(dict(gamma=g, n_comm=len(best),
                          Q=nx.community.modularity(G, best, weight="weight")))
    R["louvain_sweep"] = sweep
    # partition stability: 5 Louvain seeds on 100 subjects, mean pairwise AMI
    rng = np.random.default_rng(0); amis = []
    for s in rng.choice(len(X_log), 100, replace=False):
        Gs = nx.from_numpy_array(X_log[s].astype(float)); labs = []
        for k in range(5):
            lab = np.zeros(N, int)
            for c, nodes in enumerate(nx.community.louvain_communities(
                    Gs, weight="weight", resolution=GAMMA, seed=k)):
                lab[list(nodes)] = c
            labs.append(lab)
        amis.append(np.mean([adjusted_mutual_info_score(labs[i], labs[j])
                             for i in range(5) for j in range(i + 1, 5)]))
    R["partition_AMI"] = [float(np.mean(amis)), float(np.std(amis))]   # SD over subjects

    # ---------------- graph measures for test subjects
    topo = np.array([node_measures(W, ci) for W in Xte])        # (S, 5, N)
    mask = topo[:, 0] > 0                                        # non-isolated
    np.savez(f"{OUT}/test_topology.npz", topo=topo, mask=mask, metrics=METRICS)
    # group means for Fig. 4
    allt = np.array([node_measures(W, ci)[:3] for W in X_log.astype(float)])
    np.savez(f"{OUT}/cohort_mean_measures.npz", strength=allt[:, 1].mean(0),
             betweenness=allt[:, 2].mean(0))

    # ---------------- Table III (Fisher-z subject average) + per-seed values for Fig 2
    def coupling(prefix, seeds, avg=fisher_mean):
        rows = []
        for s in seeds:
            a = saliency(prefix, s)
            rows.append([avg(per_subject_rho(a, topo[:, m], mask)) for m in range(5)])
        return np.array(rows)                                     # (n_seeds, 5)

    per_seed = {}
    for tag, nm in [("A", "profile"), ("B", "identity")]:
        per_seed[f"trained_{nm}"] = coupling(f"attn_{tag}", range(10))
        per_seed[f"untrained_{nm}"] = coupling(f"attn_ctrl{tag}", range(5))
    R["table3"] = {k: {m: [float(v[:, i].mean()), float(v[:, i].std())]
                       for i, m in enumerate(METRICS)} for k, v in per_seed.items()}
    R["per_seed"] = {k: v.tolist() for k, v in per_seed.items()}
    R["degree_range_profile"] = [float(per_seed["trained_profile"][:, 0].min()),
                                 float(per_seed["trained_profile"][:, 0].max())]
    # Sec IV-D text: untrained identity, plain mean
    ub = coupling("attn_ctrlB", range(5), avg=np.nanmean)
    R["untrained_identity_plainmean"] = {m: [float(ub[:, i].mean()), float(ub[:, i].std())]
                                         for i, m in enumerate(METRICS[:2])}

    # ---------------- "attention paid" (row sums) sanity check
    rs = []
    for tag in "AB":
        for s in range(10):
            A = np.load(fname(f"attn_{tag}", s), mmap_mode="r")
            paid = np.asarray(A, np.float32).sum(-1).mean((1, 2))
            rs += [np.nanmean(per_subject_rho(paid, topo[:, m], mask)) for m in range(5)]
    R["row_sum_rho_range"] = [float(np.nanmin(rs)), float(np.nanmax(rs))]

    # ---------------- Table II / reconstruction loss and CVs
    loss_cv = {}
    for tag in "AB":
        best = np.array([np.load(fname(f"history_{tag}", s))[:, 1].min() for s in range(10)])
        R[f"best_val_loss_{tag}"] = [float(best.mean()), float(best.std()), best.tolist()]
        loss_cv[tag] = float(best.std() / best.mean() * 100)
    R["loss_cv_pct"] = loss_cv
    R["loss_cv_mean_pct"] = float(np.mean(list(loss_cv.values())))
    R["identity_loss_reduction_pct"] = float(
        (1 - R["best_val_loss_B"][0] / R["best_val_loss_A"][0]) * 100)
    st_cv = {nm: float(per_seed[f"trained_{nm}"][:, 1].std() /
                       per_seed[f"trained_{nm}"][:, 1].mean() * 100)
             for nm in ("profile", "identity")}
    R["strength_cv_pct"] = st_cv
    R["strength_cv_mean_pct"] = float(np.mean(list(st_cv.values())))
    pooled = np.concatenate([per_seed["trained_profile"][:, 1], per_seed["trained_identity"][:, 1]])
    R["strength_cv_pooled_pct"] = float(pooled.std() / pooled.mean() * 100)
    R["attn_to_loss_cv_ratio"] = R["strength_cv_mean_pct"] / R["loss_cv_mean_pct"]

    # ---------------- Table IV depth sweep (seeds 0-2, profile-only)
    R["depth"] = {}
    for L in (2, 4, 6):
        v = np.array([fisher_mean(per_subject_rho(
                received_attention(np.load(f"{D}/attn_depth{L}_seed{s}.npy", mmap_mode="r")),
                topo[:, 1], mask)) for s in range(3)])
        vl = np.array(DEPTH_VAL_LOSS[L])
        R["depth"][L] = dict(val_loss=[float(vl.mean()), float(vl.std())],
                             rho_strength=[float(v.mean()), float(v.std())],
                             rho_per_seed=v.tolist(),
                             attn_cv_pct=float(v.std() / v.mean() * 100),
                             loss_cv_pct=float(vl.std() / vl.mean() * 100))
        R["depth"][L]["ratio"] = R["depth"][L]["attn_cv_pct"] / R["depth"][L]["loss_cv_pct"]

    # ---------------- Sec IV-G own vs cohort-mean degree (plain mean; original paper: seeds 0-4,
    #                  revised paper: all 10 seeds)
    mean_deg = (Xte > 0).sum(2).mean(0).astype(float)          # cohort = test subjects
    g = {}
    for tag, nm in [("A", "profile"), ("B", "identity")]:
        own, coh = [], []
        for s in range(10):
            a = saliency(f"attn_{tag}", s)
            own.append(np.nanmean(per_subject_rho(a, topo[:, 0], mask)))
            coh.append(np.nanmean(per_subject_rho(a, np.tile(mean_deg, (S, 1)), mask)))
        g[nm] = dict(cohort_mean_degree=float(np.mean(coh)), own_degree=float(np.mean(own)),
                     seeds_own_gt_cohort=int(np.sum(np.array(own) > np.array(coh))),
                     own_seeds0to4=float(np.mean(own[:5])), cohort_seeds0to4=float(np.mean(coh[:5])))
    R["own_vs_cohort"] = g

    # ---------------- node-mask sensitivity (seed 0, profile-only)
    a = saliency("attn_A", 0); R["min_degree"] = {}
    for k in (1, 5, 10):
        mk = topo[:, 0] >= k
        R["min_degree"][k] = float(np.nanmean(per_subject_rho(a, topo[:, 0], mk)))

    # ---------------- Sec IV-C incremental AUC (reconstructed protocol, seed 0)
    R["auc_reconstructed"] = {nm: incremental_auc(saliency(f"attn_{t}", 0), Xte, topo)
                              for t, nm in [("A", "profile"), ("B", "identity")]}

    json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in R.items() if k != "per_seed"}, indent=2, default=float))


if __name__ == "__main__":
    main()
