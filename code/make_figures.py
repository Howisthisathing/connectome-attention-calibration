"""
make_figures.py -- regenerates fig1..fig6 of the paper from results/ and data/.
Run analysis.py first.      Usage: python make_figures.py [data_dir] [results_dir] [fig_dir]
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "data")
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "..", "results")
FIG = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9.5,
                     "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
C_P, C_I = "#2b4f91", "#5b9bd5"          # trained profile / +identity
C_UP, C_UI = "#bdbdbd", "#e0e0e0"        # untrained
METRICS = ["degree", "strength", "betweenness", "clustering", "participation"]
LABELS = [m.capitalize() for m in METRICS]
SHORT = ["Degree", "Strength", "Betw.", "Clust.", "Partic."]
R = json.load(open(f"{RES}/results.json"))
RV = json.load(open(f"{RES}/revision_results.json")) if os.path.exists(f"{RES}/revision_results.json") else None


def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{FIG}/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1
def fig1():
    T = {k: dict(v) for k, v in R["table3"].items()}
    if RV:   # revision: all 10 untrained controls
        for nm in ("profile", "identity"):
            T[f"untrained_{nm}"] = {m: RV["table3_10ctrl"][nm][m]["untrained"] for m in METRICS}
    conds = [("trained_profile", C_P, "Trained (profile)"),
             ("trained_identity", C_I, "Trained (+identity)"),
             ("untrained_profile", C_UP, "Untrained (profile)"),
             ("untrained_identity", C_UI, "Untrained (+identity)")]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    x = np.arange(5); w = 0.2
    for k, (key, col, lab) in enumerate(conds):
        m = [T[key][mm][0] for mm in METRICS]; s = [T[key][mm][1] for mm in METRICS]
        ax.bar(x + (k - 1.5) * w, m, w, yerr=s, color=col, edgecolor="black", lw=0.4,
               label=lab, error_kw=dict(lw=0.6, capsize=1.2))
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels(SHORT, fontsize=7.5)
    ax.set_ylabel("Spearman ρ")
    ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7.5,
              columnspacing=1.0, handlelength=1.2)
    save(fig, "fig1_main")


# ---------------------------------------------------------------- Fig 2
def fig2():
    P = {k: np.array(v) for k, v in R["per_seed"].items()}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    for k, (key, col, lab) in enumerate([("trained_profile", C_P, "Profile"),
                                         ("trained_identity", C_I, "+Identity")]):
        for i in range(5):
            v = P[key][:, i]; xc = i + (k - 0.5) * 0.32
            ax.plot([xc, xc], [v.min(), v.max()], color=col, lw=0.8, zorder=1)
            ax.scatter(xc + rng.uniform(-0.05, 0.05, len(v)), v, s=14, color=col,
                       edgecolor="#333", lw=0.4, zorder=2, label=lab if i == 0 else None)
            ax.plot([xc - 0.08, xc + 0.08], [v.mean()] * 2, color="black", lw=2, zorder=3)
    ax.axhline(0, color="black", lw=0.6, ls=":")
    ax.set_xticks(range(5)); ax.set_xticklabels(LABELS); ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-seed variation (10 seeds per condition)")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, "fig2_seeds")


# ---------------------------------------------------------------- Fig 3
def fig3():
    loss = np.r_[R["best_val_loss_A"][2], R["best_val_loss_B"][2]]
    lossdev = np.r_[np.array(R["best_val_loss_A"][2]) / R["best_val_loss_A"][0],
                    np.array(R["best_val_loss_B"][2]) / R["best_val_loss_B"][0]] * 100 - 100
    P = R["per_seed"]
    sp = np.array(P["trained_profile"])[:, 1]; si = np.array(P["trained_identity"])[:, 1]
    attdev = np.r_[sp / sp.mean(), si / si.mean()] * 100 - 100
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    ax.scatter(rng.uniform(-.06, .06, len(lossdev)), lossdev, s=14, color="#9e9e9e",
               edgecolor="#333", lw=0.4)
    ax.scatter(1 + rng.uniform(-.06, .06, len(attdev)), attdev, s=14, color=C_P,
               edgecolor="#333", lw=0.4)
    ax.axhline(0, color="black", lw=0.8)
    ax.text(0, lossdev.max() + 3, f"CV {R['loss_cv_mean_pct']:.1f}%", ha="center", fontsize=8.5)
    ax.text(1, attdev.max() + 3, f"CV {R['strength_cv_mean_pct']:.1f}%", ha="center", fontsize=8.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Validation\nloss", "Attention ρ\n(strength)"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylabel("Deviation from condition mean (%)")
    save(fig, "fig3_loss_vs_attn")


# ---------------------------------------------------------------- Fig 4
def fig4():
    d = np.load(f"{D}/scale33_normalized.npz"); xyz = d["coords"]
    mm = np.load(f"{RES}/cohort_mean_measures.npz")
    ci = np.load(f"{RES}/partition_gamma1.5.npy")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, v, t in [(axs[0], mm["strength"], "Mean strength"),
                     (axs[1], mm["betweenness"], "Mean betweenness")]:
        sc = ax.scatter(xyz[:, 1], xyz[:, 2], c=v, s=18, cmap="viridis", edgecolor="#333", lw=0.3)
        fig.colorbar(sc, ax=ax, fraction=0.05); ax.set_title(t)
        ax.set_xlabel("y (a.u.)"); ax.set_ylabel("z (a.u.)"); ax.set_aspect("equal")
    cols = plt.cm.tab10.colors
    for k in range(1, ci.max() + 1):
        s = ci == k
        axs[2].scatter(xyz[s, 0], xyz[s, 1], s=18, color=cols[k - 1], edgecolor="#333",
                       lw=0.3, label=f"C{k} (n={s.sum()})")
    axs[2].set_title("Louvain, γ=1.5"); axs[2].set_xlabel("x (a.u.)"); axs[2].set_ylabel("y (a.u.)")
    axs[2].set_aspect("equal")
    axs[2].legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout(); save(fig, "fig4_validation")


# ---------------------------------------------------------------- Fig 5
def fig5():
    sw = R["louvain_sweep"]; g = [s["gamma"] for s in sw]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.5))
    a1.plot(g, [s["Q"] for s in sw], "o-", color=C_P, ms=4)
    a1.set_xlabel("Resolution γ"); a1.set_ylabel("Modularity Q", color=C_P)
    a1.axvline(1.5, color="black", ls=":", lw=0.8); a1.set_title("Resolution selection")
    b = a1.twinx(); b.spines["right"].set_visible(True)
    b.plot(g, [s["n_comm"] for s in sw], "s--", color="#d62728", ms=4)
    b.set_ylabel("N communities", color="#d62728")
    for tag, col, lab in [("A", C_P, "Profile"), ("B", C_I, "+Identity")]:
        for s in range(10):
            f = f"{D}/history_{tag}.npy" if s == 0 else f"{D}/history_{tag}_seed{s}.npy"
            a2.plot(np.load(f)[:, 1], color=col, lw=0.6, alpha=0.8, label=lab if s == 0 else None)
    a2.set_ylim(0.2, 2.0); a2.set_xlabel("Epoch"); a2.set_ylabel("Validation loss")
    a2.set_title("Training (10+10 runs)"); a2.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig5_supp")


# ---------------------------------------------------------------- Fig 6
def fig6():
    dp = R["depth"]; L = [2, 4, 6]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.8, 2.3))
    a1.plot(L, [dp[str(l)]["rho_strength"][0] for l in L], "o-", color=C_P, ms=4)
    a1.set_ylim(0, 0.6); a1.set_xticks(L); a1.set_xlabel("Encoder layers")
    a1.set_ylabel("ρ with strength"); a1.set_title("Association")
    a2.plot(L, [dp[str(l)]["attn_cv_pct"] for l in L], "o-", color=C_P, ms=4, label="Attention ρ")
    a2.plot(L, [dp[str(l)]["loss_cv_pct"] for l in L], "s--", color="#9e9e9e", ms=3,
            label="Validation loss")
    a2.set_ylim(0, 25); a2.set_xticks(L); a2.set_xlabel("Encoder layers")
    a2.set_ylabel("CV across seeds (%)"); a2.set_title("Stability"); a2.legend(frameon=False)
    fig.tight_layout(); save(fig, "fig6_depth")


# ---------------------------------------------------------------- Fig 7 (revision): layers and heads
def fig7():
    LH = RV["layer_head"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.3), gridspec_kw=dict(width_ratios=[1, 1]))
    # (a) strength association by layer
    for k, (nm, col, lab) in enumerate([("profile", C_P, "Profile"), ("identity", C_I, "+Identity")]):
        m = np.array(LH[nm]["per_layer_mean"])[:, 1]; sd = np.array(LH[nm]["per_layer_sd"])[:, 1]
        a1.bar(np.arange(2) + (k - 0.5) * 0.36, m, 0.36, yerr=sd, color=col, edgecolor="black", lw=0.4,
               label=lab, error_kw=dict(lw=0.6, capsize=1.5))
    a1.set_xticks([0, 1]); a1.set_xticklabels(["Layer 1", "Layer 2"]); a1.set_ylim(0, 0.7)
    a1.set_ylabel("ρ with strength"); a1.set_title("(a) Per layer", fontsize=8.5)
    a1.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1)
    # (b) every head
    rng = np.random.default_rng(3)
    for k, (nm, col) in enumerate([("profile", C_P), ("identity", C_I)]):
        v = np.array(LH[nm]["per_head_per_seed_strength"])
        for l in range(v.shape[1]):
            xc = l + (k - 0.5) * 0.36; vals = v[:, l].ravel()
            a2.scatter(xc + rng.uniform(-0.07, 0.07, len(vals)), vals, s=5, color=col, edgecolor="#333", lw=0.2)
            a2.plot([xc - 0.12, xc + 0.12], [vals.mean()] * 2, color="black", lw=1.5)
    a2.axhline(0, color="black", lw=0.5, ls=":"); a2.set_xticks([0, 1]); a2.set_xticklabels(["Layer 1", "Layer 2"])
    a2.set_title("(b) Individual heads", fontsize=8.5)
    fig.tight_layout(pad=0.3); save(fig, "fig7_layers_heads")



# ================================================================ 6-page versions (column width, native size)
SMALL = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
         "xtick.labelsize": 7.5, "ytick.labelsize": 7.5}


def figA_community():
    """Resolution sweep + partition map (replaces fig4_validation and fig5_supp in the paper)."""
    with plt.rc_context(SMALL):
        d = np.load(f"{D}/scale33_normalized.npz"); xyz = d["coords"]
        ci = np.load(f"{RES}/partition_gamma1.5.npy"); sw = R["louvain_sweep"]; g = [s["gamma"] for s in sw]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 1.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
        a1.plot(g, [s["Q"] for s in sw], "o-", color=C_P, ms=3)
        a1.set_xlabel("Resolution γ"); a1.set_ylabel("Modularity Q", color=C_P)
        a1.axvline(1.5, color="black", ls=":", lw=0.8); a1.set_xticks([1, 1.5, 2, 2.5, 3])
        b = a1.twinx(); b.spines["right"].set_visible(True)
        b.plot(g, [s["n_comm"] for s in sw], "s--", color="#d62728", ms=3)
        b.set_ylabel("N communities", color="#d62728")
        cols = plt.cm.tab10.colors
        for k in range(1, ci.max() + 1):
            s_ = ci == k
            a2.scatter(xyz[s_, 0], xyz[s_, 1], s=9, color=cols[k - 1], edgecolor="#333", lw=0.25)
        a2.set_xticks([]); a2.set_yticks([]); a2.set_aspect("equal")
        a2.set_xlabel("left–right"); a2.set_ylabel("posterior–anterior")
        a1.set_title("(a)", loc="left"); a2.set_title("(b) γ = 1.5", loc="left")
        fig.tight_layout(pad=0.3, w_pad=0.6); save(fig, "figA_community")


def figB_stability():
    """Per-seed association + loss-vs-attention dispersion (replaces fig2_seeds and fig3)."""
    with plt.rc_context(SMALL):
        P = {k: np.array(v) for k, v in R["per_seed"].items()}
        rng = np.random.default_rng(0)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.05), gridspec_kw=dict(width_ratios=[2.5, 1]))
        short = ["Deg.", "Str.", "Betw.", "Clust.", "Part."]
        for k, (key, col, lab) in enumerate([("trained_profile", C_P, "$M_P$"),
                                             ("trained_identity", C_I, "$M_{P+I}$")]):
            for i in range(5):
                v = P[key][:, i]; xc = i + (k - 0.5) * 0.34
                a1.plot([xc, xc], [v.min(), v.max()], color=col, lw=0.7, zorder=1)
                a1.scatter(xc + rng.uniform(-0.05, 0.05, len(v)), v, s=7, color=col,
                           edgecolor="#333", lw=0.3, zorder=2, label=lab if i == 0 else None)
                a1.plot([xc - 0.09, xc + 0.09], [v.mean()] * 2, color="black", lw=1.6, zorder=3)
        a1.axhline(0, color="black", lw=0.5, ls=":")
        a1.set_xticks(range(5)); a1.set_xticklabels(short); a1.set_ylabel("Spearman ρ")
        a1.legend(frameon=False, ncol=2, loc="upper right", handletextpad=0.1, columnspacing=0.6)
        a1.set_ylim(-0.32, 0.72); a1.set_title("(a) per seed", loc="left")
        lossdev = np.r_[np.array(R["best_val_loss_A"][2]) / R["best_val_loss_A"][0],
                        np.array(R["best_val_loss_B"][2]) / R["best_val_loss_B"][0]] * 100 - 100
        sp_ = P["trained_profile"][:, 1]; si = P["trained_identity"][:, 1]
        attdev = np.r_[sp_ / sp_.mean(), si / si.mean()] * 100 - 100
        rng = np.random.default_rng(1)
        a2.scatter(rng.uniform(-.08, .08, 20), lossdev, s=7, color="#9e9e9e", edgecolor="#333", lw=0.3)
        a2.scatter(1 + rng.uniform(-.08, .08, 20), attdev, s=7, color=C_P, edgecolor="#333", lw=0.3)
        a2.axhline(0, color="black", lw=0.6)
        a2.text(0, 8, f"{R['loss_cv_mean_pct']:.1f}%", ha="center", fontsize=7)
        a2.text(1, attdev.max() + 5, f"{R['strength_cv_mean_pct']:.1f}%", ha="center", fontsize=7)
        a2.set_xticks([0, 1]); a2.set_xticklabels(["Loss", "ρ str."]); a2.set_xlim(-0.5, 1.5)
        a2.set_ylim(-75, 65); a2.set_ylabel("Deviation from mean (%)", labelpad=1)
        a2.set_title("(b) CV", loc="left")
        fig.tight_layout(pad=0.3, w_pad=0.5); save(fig, "figB_stability")


def figC_layers():
    """Layer- and head-wise association, column width."""
    with plt.rc_context(SMALL):
        LH = RV["layer_head"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.5, 2.0), gridspec_kw=dict(width_ratios=[1.55, 1]))
        x = np.arange(5); w = 0.2
        spec = [("profile", 0, C_P, None), ("profile", 1, C_P, "////"),
                ("identity", 0, C_I, None), ("identity", 1, C_I, "////")]
        for k, (nm, l, col, hatch) in enumerate(spec):
            m = np.array(LH[nm]["per_layer_mean"])[l]
            a1.bar(x + (k - 1.5) * w, m, w, color=col, hatch=hatch, edgecolor="black", lw=0.4,
                   alpha=0.5 if l == 0 else 1.0)
        a1.axhline(0, color="black", lw=0.6); a1.set_xticks(x)
        a1.set_xticklabels(["Deg", "Str", "Btw", "Clu", "Par"])
        a1.set_ylabel("Spearman ρ"); a1.set_ylim(-0.1, 0.55); a1.set_title("(a) per layer", loc="left")
        from matplotlib.patches import Patch
        a1.legend(handles=[Patch(facecolor="white", edgecolor="black", lw=0.4, label="layer 1"),
                           Patch(facecolor="white", edgecolor="black", hatch="////", lw=0.4, label="layer 2")],
                  frameon=False, loc="upper right", handlelength=1.2)
        rng = np.random.default_rng(3)
        for k, (nm, col) in enumerate([("profile", C_P), ("identity", C_I)]):
            v = np.array(LH[nm]["per_head_per_seed_strength"])
            for l in range(v.shape[1]):
                xc = l + (k - 0.5) * 0.36; vals = v[:, l].ravel()
                a2.scatter(xc + rng.uniform(-0.08, 0.08, len(vals)), vals, s=4, color=col,
                           edgecolor="#333", lw=0.2)
                a2.plot([xc - 0.12, xc + 0.12], [vals.mean()] * 2, color="black", lw=1.6)
        a2.axhline(0, color="black", lw=0.5, ls=":"); a2.set_xticks([0, 1]); a2.set_xticklabels(["L1", "L2"])
        a2.set_ylabel("ρ with strength", labelpad=1); a2.set_title("(b) per head", loc="left")
        fig.tight_layout(pad=0.3, w_pad=0.5); save(fig, "figC_layers")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, figA_community, figB_stability) + ((fig7, figC_layers) if RV else ()):
        f(); print("wrote", f.__name__)
