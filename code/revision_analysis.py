"""
revision_analysis.py -- additional analyses requested in the CMES 2027 reviews (paper 342).
Run after analysis.py (uses results/test_topology.npz and results/partition_gamma1.5.npy).

  R3  : trained vs untrained tests/CIs using all 10 saved untrained controls
  R1/R3: layer- and head-wise attention-topology correlations
  R1/R3: seed-to-seed similarity of 73-region saliency maps; bootstrap CIs on CVs
  R3  : node-mask (minimum-degree) sensitivity over all 10 seeds
  R2/R3: leakage-free held-out edge prediction (target edges hidden from BOTH the
         model input and the centrality features), with subject-bootstrap CIs
  R3  : gradient attribution vs attention (checkpoints for seeds 0, 3, 4)

Usage: python revision_analysis.py [data_dir] [results_dir]
"""
import sys, os, json, warnings
import numpy as np
import bct
import torch
from scipy.stats import spearmanr, ttest_ind, mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results")
sys.path.insert(0, HERE)
from model import ConnectomeTransformer          # original model.py

METRICS = ["degree", "strength", "betweenness", "clustering", "participation"]
CKPT = {"A": {0: "model_A.pt", 3: "model_A_seed3.pt", 4: "model_A_seed4.pt"},
        "B": {0: "model_B.pt", 3: "model_B_seed3.pt", 4: "model_B_seed4.pt"}}
rng_global = np.random.default_rng(2027)
torch.set_num_threads(max(1, os.cpu_count() or 1))


def fname(prefix, s):
    return f"{D}/{prefix}.npy" if s == 0 else f"{D}/{prefix}_seed{s}.npy"


def fz(r):
    r = np.clip(np.asarray(r, float), -0.999999, 0.999999)
    return float(np.tanh(np.nanmean(np.arctanh(r))))


def rho_subj(a, x, mask):
    return np.array([spearmanr(a[s][mask[s]], x[s][mask[s]])[0] for s in range(len(a))])


def boot_ci(x, stat=np.mean, n=10000, rng=None):
    rng = rng or np.random.default_rng(0); x = np.asarray(x)
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def cv(x):
    x = np.asarray(x); return float(x.std() / x.mean() * 100)


def node_measures(W, ci):
    W = W.astype(float).copy(); np.fill_diagonal(W, 0)
    deg = (W > 0).sum(1).astype(float); strength = W.sum(1)
    L = np.where(W > 0, 1.0 / np.where(W > 0, W, 1.0), 0.0)
    btw = bct.betweenness_wei(L) / ((len(W) - 1) * (len(W) - 2) / 2)
    clus = bct.clustering_coef_wu(W / W.max())
    part = bct.participation_coef(W, ci)
    return np.stack([deg, strength, btw, clus, part])


def received(t, s):
    return np.asarray(np.load(fname(f"attn_{t}", s), mmap_mode="r"), np.float32).sum(3).mean((1, 2))


def cached(name, fn):
    """Run fn() once and cache its JSON-able result in results/_cache/<name>.json."""
    os.makedirs(f"{RES}/_cache", exist_ok=True)
    f = f"{RES}/_cache/{name}.json"
    if os.path.exists(f):
        return json.load(open(f))
    out = fn(); json.dump(out, open(f, "w"), default=float); return out


def main():
    d = np.load(f"{D}/scale33_normalized.npz"); sp = np.load(f"{D}/splits.npz")
    Xte = d["X_log"][sp["test"]].astype(np.float32); S, N = Xte.shape[:2]
    tt = np.load(f"{RES}/test_topology.npz"); topo, mask = tt["topo"], tt["mask"]
    ci = np.load(f"{RES}/partition_gamma1.5.npy")

    def parts_1_to_4():
        R = {}
        # ---------- load all attention once (received attention per layer/head)
        def lh_sal(prefix, s):                                        # (S, L, H, N)
            return np.asarray(np.load(fname(prefix, s), mmap_mode="r"), np.float32).sum(3)
        SAL = {(t, s): lh_sal(f"attn_{t}", s) for t in ["A", "B", "ctrlA", "ctrlB"] for s in range(10)}
        sal = {k: v.mean((1, 2)) for k, v in SAL.items()}             # (S, N)

        # ---------- 1. trained vs untrained with 10 controls
        def coupling(t, seeds):
            return np.array([[fz(rho_subj(sal[(t, s)], topo[:, m], mask)) for m in range(5)]
                             for s in seeds])
        tr = {t: coupling(t, range(10)) for t in "AB"}
        un = {t: coupling("ctrl" + t, range(10)) for t in "AB"}
        R["table3_10ctrl"] = {}
        for t, nm in [("A", "profile"), ("B", "identity")]:
            R["table3_10ctrl"][nm] = {}
            for i, m in enumerate(METRICS):
                a, b = tr[t][:, i], un[t][:, i]
                rng = np.random.default_rng(i)
                diffs = [rng.choice(a, 10).mean() - rng.choice(b, 10).mean() for _ in range(10000)]
                R["table3_10ctrl"][nm][m] = dict(
                    trained=[float(a.mean()), float(a.std())], untrained=[float(b.mean()), float(b.std())],
                    diff=float(a.mean() - b.mean()),
                    diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                    welch_p=float(ttest_ind(a, b, equal_var=False).pvalue),
                    mannwhitney_p=float(mannwhitneyu(a, b).pvalue))
        R["per_seed_untrained10"] = {k: v.tolist() for k, v in un.items()}
        print("1 done", flush=True)

        # ---------- 2. layer- and head-wise coupling (trained, 10 seeds)
        R["layer_head"] = {}
        for t, nm in [("A", "profile"), ("B", "identity")]:
            L, H = SAL[(t, 0)].shape[1:3]
            lh = np.zeros((10, L, H, 5)); ly = np.zeros((10, L, 5))
            for s in range(10):
                for l in range(L):
                    ly[s, l] = [fz(rho_subj(SAL[(t, s)][:, l].mean(1), topo[:, m], mask)) for m in range(5)]
                    for h in range(H):
                        lh[s, l, h] = [fz(rho_subj(SAL[(t, s)][:, l, h], topo[:, m], mask)) for m in range(5)]
            R["layer_head"][nm] = dict(
                per_layer_mean=ly.mean(0).tolist(), per_layer_sd=ly.std(0).tolist(),
                per_head_mean=lh.mean(0).tolist(), per_head_sd=lh.std(0).tolist(),
                per_head_per_seed_strength=lh[..., 1].tolist(),
                # head specialisation: for each seed/layer/head, which metric is strongest
                best_metric_counts={m: int((lh.argmax(-1) == i).sum()) for i, m in enumerate(METRICS)},
                head_strength_range=[float(lh[..., 1].min()), float(lh[..., 1].max())],
                frac_heads_strength_positive=float((lh[..., 1] > 0).mean()),
                # within-seed spread across heads vs across-seed spread
                sd_across_heads_within_seed=float(lh[..., 1].reshape(10, -1).std(1).mean()),
                sd_across_seeds_of_mean=float(lh[..., 1].reshape(10, -1).mean(1).std()))
        print("2 done", flush=True)

        # ---------- 3. seed-to-seed saliency-map similarity + bootstrap CVs
        R["saliency_similarity"] = {}
        pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
        for t, nm in [("A", "profile_trained"), ("B", "identity_trained"),
                      ("ctrlA", "profile_untrained"), ("ctrlB", "identity_untrained")]:
            subj = np.array([[spearmanr(sal[(t, i)][s][mask[s]], sal[(t, j)][s][mask[s]])[0]
                              for s in range(S)] for i, j in pairs])     # (45, S)
            pair_mean = np.array([fz(r) for r in subj])
            cohort = [spearmanr(sal[(t, i)].mean(0), sal[(t, j)].mean(0))[0] for i, j in pairs]
            R["saliency_similarity"][nm] = dict(
                subject_level_mean=float(pair_mean.mean()), subject_level_sd=float(pair_mean.std()),
                subject_level_range=[float(pair_mean.min()), float(pair_mean.max())],
                subject_level_ci95=boot_ci(pair_mean),
                cohort_map_mean=float(np.mean(cohort)), cohort_map_range=[float(np.min(cohort)), float(np.max(cohort))])
        # reference: similarity of each subject's strength map with itself is 1; similarity of saliency
        # with strength is the coupling. Also report CV bootstrap CIs (resampling seeds).
        loss = {t: np.array([np.load(fname(f"history_{t}", s))[:, 1].min() for s in range(10)]) for t in "AB"}
        R["cv_bootstrap"] = {}
        for t, nm in [("A", "profile"), ("B", "identity")]:
            st = tr[t][:, 1]
            R["cv_bootstrap"][nm] = dict(
                strength_cv=cv(st), strength_cv_ci95=boot_ci(st, cv),
                loss_cv=cv(loss[t]), loss_cv_ci95=boot_ci(loss[t], cv),
                strength_mad_over_median=float(np.median(np.abs(st - np.median(st))) / np.median(st) * 100),
                loss_mad_over_median=float(np.median(np.abs(loss[t] - np.median(loss[t]))) / np.median(loss[t]) * 100))
        print("3 done", flush=True)

        # ---------- 4. node-mask sensitivity over 10 seeds (plain subject mean, as in paper)
        R["min_degree_10seeds"] = {}
        for t, nm in [("A", "profile"), ("B", "identity")]:
            out = {}
            for k in (1, 5, 10):
                mk = topo[:, 0] >= k
                v = [np.nanmean(rho_subj(sal[(t, s)], topo[:, 0], mk)) for s in range(10)]
                out[k] = [float(np.mean(v)), float(np.std(v))]
            R["min_degree_10seeds"][nm] = out
        print("4 done", flush=True)

        return R

    R = cached("parts_1_to_4", parts_1_to_4)
    sal = {(t, s): received(t, s) for t in "AB" for s in (0, 3, 4)}
    per_seed = json.load(open(f"{RES}/results.json"))["per_seed"]
    tr = {"A": np.array(per_seed["trained_profile"]), "B": np.array(per_seed["trained_identity"])}

    # ---------- models (seeds 0, 3, 4)
    def load(t, s):
        m = ConnectomeTransformer(N, use_identity=(t == "B"))
        m.load_state_dict(torch.load(f"{D}/{CKPT[t][s]}", map_location="cpu")); m.eval()
        return m

    def untrained(t, s):
        torch.manual_seed(s); m = ConnectomeTransformer(N, use_identity=(t == "B")); m.eval(); return m

    def attn_sal(m, X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 16):
                _, at = m(torch.tensor(X[i:i + 16]), return_attn=True)
                out.append(torch.stack(at, 1).sum(3).mean((1, 2)).numpy())
        return np.concatenate(out)

    # ---------- 5. leakage-free held-out edge prediction
    iu = np.triu_indices(N, 1); n_pairs = len(iu[0]); n_mask = int(0.15 * n_pairs)
    REPS = 2
    rng = np.random.default_rng(42)
    sel = np.array([[rng.permutation(n_pairs)[:n_mask] for _ in range(REPS)] for _ in range(S)])  # S,REPS,n_mask
    Xm = np.repeat(Xte[:, None], REPS, 1).copy()                                  # S,REPS,N,N
    for s in range(S):
        for r in range(REPS):
            i, j = iu[0][sel[s, r]], iu[1][sel[s, r]]
            Xm[s, r, i, j] = 0; Xm[s, r, j, i] = 0
    Xm_flat = Xm.reshape(-1, N, N)
    f = f"{RES}/_cache/topo_masked_{REPS}.npy"
    if not os.path.exists(f):
        os.makedirs(f"{RES}/_cache", exist_ok=True)
        np.save(f, np.array([node_measures(W, ci) for W in Xm_flat]))
    topo_masked = np.load(f)                                                       # (S*REPS,5,N)
    topo_full = np.repeat(topo, REPS, 0)
    print("5 masked topology done", flush=True)
    i_idx = iu[0][sel.reshape(S * REPS, -1)]; j_idx = iu[1][sel.reshape(S * REPS, -1)]
    y = np.array([(Xte[g // REPS][i_idx[g], j_idx[g]] > 0) for g in range(S * REPS)]).astype(int)
    groups = np.repeat(np.arange(S), REPS)[:, None].repeat(n_mask, 1)

    def pairfeat(F):                                              # F: (G, N, k)
        fi = np.take_along_axis(F, i_idx[..., None], 1); fj = np.take_along_axis(F, j_idx[..., None], 1)
        return np.concatenate([fi + fj, fi * fj, np.minimum(fi, fj), np.maximum(fi, fj)], -1)

    def oof_auc(Fe, kind):
        Xf = Fe.reshape(-1, Fe.shape[-1]); yf = y.ravel(); gf = groups.ravel()
        p = np.zeros(len(yf))
        for trn, tst in GroupKFold(5).split(Xf, yf, gf):
            if kind == "logistic":
                sc = StandardScaler().fit(Xf[trn])
                p[tst] = LogisticRegression(max_iter=3000).fit(sc.transform(Xf[trn]), yf[trn]) \
                    .predict_proba(sc.transform(Xf[tst]))[:, 1]
            else:
                p[tst] = HistGradientBoostingClassifier(random_state=0).fit(Xf[trn], yf[trn]) \
                    .predict_proba(Xf[tst])[:, 1]
        return p

    def subj_boot_delta(p0, p1, n=500):
        yf = y.ravel(); gf = groups.ravel(); idx = [np.where(gf == s)[0] for s in range(S)]
        rng = np.random.default_rng(7); out = []
        for _ in range(n):
            ii = np.concatenate([idx[s] for s in rng.integers(0, S, S)])
            out.append(roc_auc_score(yf[ii], p1[ii]) - roc_auc_score(yf[ii], p0[ii]))
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], float(np.mean(np.array(out) <= 0))

    edge = {}
    base_cache = {}
    for setting, T_in, use_masked_input in [("leakage_free", topo_masked, True),
                                            ("leaky", topo_full, False)]:
        Ft = pairfeat(T_in.transpose(0, 2, 1))
        for kind in ("logistic", "gradient_boosting"):
            f = f"{RES}/_cache/base_{setting}_{kind}.npy"
            if not os.path.exists(f):
                os.makedirs(f"{RES}/_cache", exist_ok=True); np.save(f, oof_auc(Ft, kind))
            base_cache[(setting, kind)] = np.load(f)
        edge[setting] = {}
        for t, nm in [("A", "profile"), ("B", "identity")]:
            edge[setting][nm] = {}
            for s in (0, 3, 4):
              def run_one():
                m = load(t, s)
                a = attn_sal(m, Xm_flat) if use_masked_input else np.repeat(attn_sal(m, Xte), REPS, 0)
                Fa = pairfeat(a[..., None]); Fb = np.concatenate([Ft, Fa], -1)
                res = {}
                for kind in ("logistic", "gradient_boosting"):
                    p0 = base_cache[(setting, kind)]; p1 = oof_auc(Fb, kind); pa = oof_auc(Fa, kind)
                    ci95, p_le0 = subj_boot_delta(p0, p1)
                    res[kind] = dict(topology=float(roc_auc_score(y.ravel(), p0)),
                                     attention=float(roc_auc_score(y.ravel(), pa)),
                                     both=float(roc_auc_score(y.ravel(), p1)),
                                     delta=float(roc_auc_score(y.ravel(), p1) - roc_auc_score(y.ravel(), p0)),
                                     delta_ci95=ci95, boot_p_delta_le0=p_le0)
                return res
              res = cached(f"edge_{setting}_{t}_{s}", run_one)
              edge[setting][nm][s] = res
              print("5", setting, nm, s, {k: round(v["delta"], 4) for k, v in res.items()}, flush=True)
    edge["design"] = dict(subjects=int(S), masks_per_subject=REPS, masked_pairs_per_mask=int(n_mask),
                          positive_rate=float(y.mean()), cv="5-fold GroupKFold by subject",
                          features="sum, product, min, max of endpoint node features")
    R["edge_prediction"] = edge

    # ---------- 6. gradient attribution vs attention
    # Two node attributions from the gradient of the full reconstruction loss w.r.t. the input
    # profile of node i (row i of A):  'gradient' = sum_j |dL/dA_ij|,
    # 'grad_x_input' = sum_j |A_ij * dL/dA_ij|  (scales with input, hence with strength by construction)
    def grad_sal(m, X):
        g_out, gx_out = [], []
        iu_t = torch.triu(torch.ones(N, N, dtype=torch.bool), 1)
        for i in range(0, len(X), 16):
            A = torch.tensor(X[i:i + 16], requires_grad=True)
            pred, _ = m(A)
            loss = ((pred - A.detach())[:, iu_t] ** 2).sum()
            g, = torch.autograd.grad(loss, A)
            g_out.append(g.abs().sum(2).numpy()); gx_out.append((g * A).abs().sum(2).detach().numpy())
        return {"gradient": np.concatenate(g_out), "grad_x_input": np.concatenate(gx_out)}

    def summarise(rows):
        rows = np.array(rows)
        return {mm: [float(rows[:, k].mean()), float(rows[:, k].std())] for k, mm in enumerate(METRICS)}

    R["gradient"] = {}
    for t, nm in [("A", "profile"), ("B", "identity")]:
        out = {}
        for kind in ("gradient", "grad_x_input"):
            out[kind] = {"trained": [], "untrained": [], "vs_attention": []}
        for s in (0, 3, 4):
            gt = grad_sal(load(t, s), Xte); gu = grad_sal(untrained(t, s), Xte); a = sal[(t, s)]
            for kind in ("gradient", "grad_x_input"):
                out[kind]["trained"].append([fz(rho_subj(gt[kind], topo[:, k], mask)) for k in range(5)])
                out[kind]["untrained"].append([fz(rho_subj(gu[kind], topo[:, k], mask)) for k in range(5)])
                out[kind]["vs_attention"].append(fz(rho_subj(gt[kind], a, mask)))
        R["gradient"][nm] = {kind: dict(trained=summarise(v["trained"]), untrained=summarise(v["untrained"]),
                                        vs_attention=[float(np.mean(v["vs_attention"])), float(np.std(v["vs_attention"]))])
                             for kind, v in out.items()}
        R["gradient"][nm]["attention_same_seeds"] = summarise(tr[t][[0, 3, 4]])
        print("6", nm, R["gradient"][nm]["gradient"]["trained"]["strength"], flush=True)

    json.dump(R, open(f"{RES}/revision_results.json", "w"), indent=2, default=float)
    print("saved")


if __name__ == "__main__":
    main()
