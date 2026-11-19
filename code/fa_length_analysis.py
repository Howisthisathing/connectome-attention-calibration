"""
fa_length_analysis.py -- reproduces the "FA / fibre length" robustness check
(Sec. IV-H, last sentence) and the pre-QC numbers (1064 connectomes, 83 ROIs,
12,106 isolated nodes, initial density 0.226, r = -0.435).

NEEDS the raw braingraph.org files (not in the backup):
  braingraph.org -> "Download PIT group connectomes" -> the 1064 *robust* HCP
  connectomes (Varga & Grolmusz 2021), resolution 83 nodes (scale 33), GraphML.
  Unzip them into  data/graphml/   (any sub-folder layout is fine).
Each GraphML edge carries: number_of_fibers, fiber_length_mean, FA_mean.

Question tested: does EDGE-level attention (A_ij + A_ji)/2, averaged over layers
and heads, still correlate with mean FA or mean fibre length once streamline
count is controlled for?  -> per-subject partial Spearman over existing edges.

Usage:  python fa_length_analysis.py [data_dir] [results_dir] [--preqc-only]
"""
import sys, os, glob, json, re
import numpy as np
import networkx as nx
from scipy.stats import rankdata, pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
D = ARGS[0] if len(ARGS) > 0 else os.path.join(HERE, "..", "data")
RES = ARGS[1] if len(ARGS) > 1 else os.path.join(HERE, "..", "results")
GDIR = os.path.join(D, "graphml")


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower().replace("ctx-", "").replace("ctx_", ""))


def node_key(attrs):
    """Build a name comparable to scale33_normalized.npz node_names
    (e.g. 'rh.lateralorbitofrontal', 'Right-Thalamus-Proper', 'Brain-Stem')."""
    name = attrs.get("dn_name") or attrs.get("dn_fsname") or attrs.get("name") or ""
    hemi = str(attrs.get("dn_hemisphere", "")).lower()
    cands = {norm(name)}
    if hemi.startswith("r"):
        cands.add(norm("rh." + str(name)))
    if hemi.startswith("l"):
        cands.add(norm("lh." + str(name)))
    return cands


def load_graph(path, keys=("number_of_fibers", "fiber_length_mean", "FA_mean")):
    G = nx.read_graphml(path)
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    M = {k: np.zeros((len(nodes), len(nodes))) for k in keys}
    for u, v, a in G.edges(data=True):
        for k in keys:
            if k in a:
                M[k][idx[u], idx[v]] = M[k][idx[v], idx[u]] = float(a[k])
    return nodes, [G.nodes[n] for n in nodes], M


def partial_spearman(x, y, z):
    """Spearman correlation of x and y controlling for z (rank, residualise, correlate)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearsonr(ex, ey)[0]


def main():
    files = sorted(glob.glob(os.path.join(GDIR, "**", "*.graphml"), recursive=True))
    files = [f for f in files if re.search(r"(scale_?33|_83|83_)", os.path.basename(f)) or len(files) <= 1100]
    if not files:
        sys.exit(f"No GraphML files under {GDIR}. Download the 83-node robust HCP set from braingraph.org.")
    d = np.load(f"{D}/scale33_normalized.npz"); sp = np.load(f"{D}/splits.npz")
    subj_all = [str(s) for s in d["subjects"]]; names73 = [norm(n) for n in d["node_names"]]
    by_subj = {}
    for f in files:
        m = re.search(r"(\d{6})", os.path.basename(f))
        if m:
            by_subj.setdefault(m.group(1), f)
    print(f"{len(by_subj)} subjects with GraphML files")
    R = {}

    # ---------- 1. pre-QC statistics over ALL available connectomes
    dens, iso_counts, n_nodes, isomat = [], [], set(), []
    for s, f in sorted(by_subj.items()):
        _, _, M = load_graph(f, ("number_of_fibers",))
        W = M["number_of_fibers"]; N = len(W); n_nodes.add(N)
        dens.append((W > 0).sum() / (N * (N - 1))); iso = (W > 0).sum(1) == 0
        iso_counts.append(int(iso.sum())); isomat.append(iso)
    dens, iso_counts, isomat = np.array(dens), np.array(iso_counts), np.array(isomat)
    keep = isomat.mean(0) <= 0.5                       # ROIs isolated in <= 50% of subjects
    resid = isomat[:, keep].sum(1)                     # residual isolated nodes per subject
    R["pre_qc"] = dict(n_connectomes=len(by_subj), n_nodes=sorted(n_nodes),
                       density=[float(dens.mean()), float(dens.std())],
                       isolated_node_instances=int(iso_counts.sum()),
                       isolated_pct=float(iso_counts.sum() / (len(by_subj) * max(n_nodes)) * 100),
                       r_isolated_vs_density=float(np.corrcoef(iso_counts, dens)[0, 1]),
                       n_rois_kept=int(keep.sum()),
                       r_residual_isolated_vs_density=float(np.corrcoef(resid, dens)[0, 1]),
                       n_subjects_resid_le5=int((resid <= 5).sum()))
    print("pre-QC:", R["pre_qc"])
    if "--preqc-only" in sys.argv:
        return

    # ---------- 2. edge-level attention vs FA / length, controlling for streamline count
    test_subj = [subj_all[i] for i in sp["test"]]
    mats = {}
    for s in test_subj:
        if s not in by_subj:
            continue
        _, attrs, M = load_graph(by_subj[s])
        keys = [node_key(a) for a in attrs]
        order = []
        for nm in names73:
            hit = [i for i, k in enumerate(keys) if nm in k]
            order.append(hit[0] if hit else -1)
        if -1 in order:
            missing = [d["node_names"][j] for j, o in enumerate(order) if o == -1]
            sys.exit(f"Could not match node names for subject {s}: {missing[:5]} ... "
                     f"Inspect a GraphML file's node attributes and adjust node_key().")
        o = np.array(order)
        mats[s] = {k: v[np.ix_(o, o)] for k, v in M.items()}
    print(f"{len(mats)}/{len(test_subj)} test subjects matched")
    # sanity check: streamline counts should match X_raw
    i0 = list(sp["test"])[0]; s0 = subj_all[i0]
    if s0 in mats:
        print("max |fibers - X_raw| for first test subject:",
              float(np.abs(mats[s0]["number_of_fibers"] - d["X_raw"][i0]).max()))

    iu = np.triu_indices(73, 1)
    out = {}
    for tag, nm in [("A", "profile"), ("B", "identity")]:
        per_seed = {"FA_mean": [], "fiber_length_mean": []}
        per_subject_seed0 = {"FA_mean": [], "fiber_length_mean": []}
        for seed in range(10):
            f = f"{D}/attn_{tag}.npy" if seed == 0 else f"{D}/attn_{tag}_seed{seed}.npy"
            A = np.asarray(np.load(f, mmap_mode="r"), np.float32).mean((1, 2))      # (S, N, N)
            A = (A + A.transpose(0, 2, 1)) / 2
            vals = {"FA_mean": [], "fiber_length_mean": []}
            for k, s in enumerate(test_subj):
                if s not in mats:
                    continue
                fib = mats[s]["number_of_fibers"][iu]; e = fib > 0
                for prop in vals:
                    vals[prop].append(partial_spearman(A[k][iu][e], mats[s][prop][iu][e], fib[e]))
            for prop in vals:
                v = np.array(vals[prop]); per_seed[prop].append(float(np.tanh(np.arctanh(v).mean())))
                if seed == 0:
                    per_subject_seed0[prop] = v
        res = {}
        for prop in per_seed:
            ps = np.array(per_seed[prop]); v = per_subject_seed0[prop]
            rng = np.random.default_rng(0)
            boot = [np.tanh(np.arctanh(v[rng.integers(0, len(v), len(v))]).mean()) for _ in range(5000)]
            res[prop] = dict(
                mean_over_seeds=float(ps.mean()), sd_over_seeds=float(ps.std()),
                seed0_mean=float(np.tanh(np.arctanh(v).mean())), seed0_sd_over_subjects=float(v.std()),
                seed0_CI95_of_mean=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                seed0_range95_of_subjects=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
        out[nm] = res
        print(nm, json.dumps(res, indent=1))
    R["edge_partial"] = out
    json.dump(R, open(f"{RES}/fa_length_results.json", "w"), indent=2)
    print("saved", f"{RES}/fa_length_results.json")


if __name__ == "__main__":
    main()
