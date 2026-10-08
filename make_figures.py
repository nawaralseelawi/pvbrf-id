"""
make_figures.py -- regenerate the manuscript figures 2-5 from the raw result files.

Inputs (produced by the two notebooks):
    results/v5_full/results.csv          870 main runs
    results/v5b_tolerance/results.csv    160 supplementary runs (tolerance-exploiting adaptive attacker)
Outputs:
    figures/fig2_macro_f1.png
    figures/fig3_convergence.png
    figures/fig4_backdoor_tradeoff.png
    figures/fig5_privacy.png
    figures/tableS3_per_class_recall.csv

Every number plotted here is recomputed from results.csv; nothing is typed in by hand.
Run:  python make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
V5 = pd.read_csv(ROOT / "results" / "v5_full" / "results.csv")
TOL = pd.read_csv(ROOT / "results" / "v5b_tolerance" / "results.csv")
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

T95 = {9: 2.262}  # t critical value, df = n - 1, two-sided 95%


def ci(x):
    x = np.asarray(x, dtype=float)
    return T95[len(x) - 1] * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def stats(df, arm, scn, col="final_macro_f1"):
    x = df[(df.arm == arm) & (df.scenario == scn)].sort_values("seed")[col].values
    return float(x.mean()), ci(x)


# Fixed categorical colour assignment (Okabe-Ito, colour-blind safe); FedAvg is the grey baseline.
COLOR = {
    "fedavg": "#7A7A7A",
    "trimmed_mean": "#E69F00",
    "multi_krum": "#D55E00",
    "secure_norm_only": "#CC79A7",
    "faithful_root_t-0.3": "#0072B2",
    "faithful_root_t-0.3_clip": "#009E73",
}
LABEL = {
    "fedavg": "FedAvg",
    "trimmed_mean": "Trimmed mean\u2020",
    "multi_krum": "Multi-Krum\u2020",
    "secure_norm_only": "Secure norm-only",
    "faithful_root_t-0.3": "PVBRF-ID (\u03c4=\u22120.3)",
    "faithful_root_t-0.3_clip": "PVBRF-ID (\u03c4=\u22120.3) + clip",
}
ORDER = list(COLOR)

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#444444", "axes.labelcolor": "#222222", "xtick.color": "#222222",
                     "ytick.color": "#222222", "legend.frameon": False})


# ---------------------------------------------------------------- Figure 2: macro-F1 bars
def source_for(arm, scenario):
    """Adaptive columns for the tau = -0.3 rows use the tolerance-exploiting attacker (v5b);
    rules without a reference are unaffected and identical in both runs."""
    if scenario in ("adaptive", "adaptive_f30") and arm.startswith("faithful_root_t-0.3"):
        return TOL, {"adaptive": "adaptive_tol", "adaptive_f30": "adaptive_tol_f30"}[scenario]
    return V5, scenario


def fig2():
    scns = [("none", "Clean"), ("signflip", "Sign-flip"), ("adaptive", "Adaptive 20%"), ("adaptive_f30", "Adaptive 30%")]
    central, _ = stats(V5, "centralized", "none")
    fig, ax = plt.subplots(figsize=(13, 4.6))
    w = 0.13
    for j, arm in enumerate(ORDER):
        xs, ms, cs = [], [], []
        for i, (scn, _) in enumerate(scns):
            df, s = source_for(arm, scn)
            if df[(df.arm == arm) & (df.scenario == s)].empty:
                continue
            m, c = stats(df, arm, s)
            xs.append(i + (j - 2.5) * w); ms.append(m); cs.append(c)
        ax.bar(xs, ms, width=w * 0.92, color=COLOR[arm], label=LABEL[arm], yerr=cs, capsize=3,
               error_kw=dict(elinewidth=1, ecolor="#222222"))
    ax.axhline(central, color="#222222", ls=":", lw=1)
    ax.text(len(scns) - 0.55, central + 0.012, "centralized", ha="right", fontsize=9, color="#222222")
    ax.set_xticks(range(len(scns))); ax.set_xticklabels([l for _, l in scns])
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 0.85)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.14), fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig2_macro_f1.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Figure 3: convergence
def curves(df, arm, scn):
    g = df[(df.arm == arm) & (df.scenario == scn)].sort_values("seed")
    return np.array([json.loads(c) for c in g.curve_macro_f1])


def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    rounds = np.arange(1, 31)
    left = [("centralized", "Centralized", "#222222", ":"), ("fedavg", LABEL["fedavg"], COLOR["fedavg"], "-"),
            ("multi_krum", LABEL["multi_krum"], COLOR["multi_krum"], "-"),
            ("faithful_root_t-0.3", LABEL["faithful_root_t-0.3"], COLOR["faithful_root_t-0.3"], "-"),
            ("faithful_root_t-0.3_clip", LABEL["faithful_root_t-0.3_clip"], COLOR["faithful_root_t-0.3_clip"], "-")]
    for arm, lab, col, ls in left:
        C = curves(V5, arm, "none"); m = C.mean(0); h = np.array([ci(C[:, k]) for k in range(C.shape[1])])
        axes[0].plot(rounds, m, color=col, ls=ls, lw=2, label=lab)
        if ls == "-":
            axes[0].fill_between(rounds, m - h, m + h, color=col, alpha=0.18, lw=0)
    right = [("fedavg", V5, "adaptive_f30"), ("secure_norm_only", V5, "adaptive_f30"), ("multi_krum", V5, "adaptive_f30"),
             ("faithful_root_t-0.3", TOL, "adaptive_tol_f30"), ("faithful_root_t-0.3_clip", TOL, "adaptive_tol_f30")]
    for arm, df, scn in right:
        C = curves(df, arm, scn); m = C.mean(0); h = np.array([ci(C[:, k]) for k in range(C.shape[1])])
        axes[1].plot(rounds, m, color=COLOR[arm], lw=2, label=LABEL[arm])
        axes[1].fill_between(rounds, m - h, m + h, color=COLOR[arm], alpha=0.18, lw=0)
    axes[0].set_title("Clean"); axes[1].set_title("Adaptive attacker, 30% of sites")
    for ax in axes:
        ax.set_xlabel("round"); ax.set_xlim(0, 31); ax.set_ylim(0, 0.8); ax.legend(fontsize=9)
    axes[0].set_ylabel("macro-F1")
    fig.tight_layout(); fig.savefig(OUT / "fig3_convergence.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Figure 4: backdoor trade-off
def fig4():
    pts = [  # (arm, label, marker, colour, label offset)
        ("fedavg", "FedAvg", "s", COLOR["fedavg"], (6, 6)),
        ("trimmed_mean", "Trimmed\u2020", "s", COLOR["trimmed_mean"], (8, -4)),
        ("multi_krum", "Krum\u2020", "s", COLOR["multi_krum"], (-34, -14)),
        ("faithful_root_t-0.3", "PVBRF-ID", "o", COLOR["faithful_root_t-0.3"], (8, -12)),
        ("faithful_root_t-0.3_clip", "+clip", "o", COLOR["faithful_root_t-0.3_clip"], (-30, -16)),
        ("faithful_root_t-0.3_all_z0.5", "z=0.5", "^", "#009E73", (8, 4)),
        ("faithful_root_t-0.3_all_z1.0", "z=1", "^", "#009E73", (8, -4)),
        ("faithful_root_t-0.3_all_z2.0", "z=2", "^", "#009E73", (8, 2)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for arm, lab, mk, col, off in pts:
        f1, _ = stats(V5, arm, "none"); a, _ = stats(V5, arm, "backdoor", "final_asr")
        ax.scatter(f1, a, marker=mk, s=110, color=col, edgecolor="white", linewidth=1.2, zorder=3)
        ax.annotate(lab, (f1, a), textcoords="offset points", xytext=off, fontsize=9, color="#222222")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="s", color="#7A7A7A", ls="", ms=9, label="plaintext baselines\u2020"),
               Line2D([], [], marker="o", color=COLOR["faithful_root_t-0.3"], ls="", ms=9, label="PVBRF-ID (clip / no clip)"),
               Line2D([], [], marker="^", color="#009E73", ls="", ms=9, label="PVBRF-ID + clip + noise + agg. clip")]
    ax.legend(handles=handles, loc="center left", fontsize=9)
    ax.set_xlabel("clean macro-F1 (higher is better)"); ax.set_ylabel("backdoor ASR (lower is better)")
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout(); fig.savefig(OUT / "fig4_backdoor_tradeoff.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Figure 5: privacy attacker
def fig5():
    pv = V5[(V5.scenario == "none") & V5.obs_attr_rho.notna()]
    plain = pv[pv.arm == "fedavg"]; sec = pv[pv.arm == "faithful_root_t-0.3"]
    metrics = [("Class-mix\ncorrelation", "obs_attr_rho", "obs_collude_attr_rho", 0.0),
               ("Top-3 class\nhit rate", "obs_top3", "obs_collude_top3", 0.20),
               ("Neighbour\nhit rate", "obs_nn_hit", "obs_collude_nn_hit", 1 / 3)]
    views = [("Plaintext updates", "#D55E00"), ("PVBRF-ID: one server's share", "#009E73"), ("PVBRF-ID: both servers collude", "#E69F00")]
    fig, ax = plt.subplots(figsize=(8, 4.6)); w = 0.25
    for i, (lab, col_p, col_c, chance) in enumerate(metrics):
        vals = [(plain[col_p].mean(), ci(plain[col_p])), (sec[col_p].mean(), ci(sec[col_p])), (sec[col_c].mean(), ci(sec[col_c]))]
        for j, ((m, c), (vlab, vcol)) in enumerate(zip(vals, views)):
            ax.bar(i + (j - 1) * w, m, width=w * 0.92, color=vcol, yerr=c, capsize=3, label=vlab if i == 0 else None,
                   error_kw=dict(elinewidth=1, ecolor="#222222"))
        if chance > 0:
            ax.hlines(chance, i - 1.5 * w, i + 1.5 * w, color="#222222", ls="--", lw=1)
    ax.text(2 + 1.5 * w, 1 / 3 + 0.02, "chance", ha="right", fontsize=9, color="#222222")
    ax.set_xticks(range(3)); ax.set_xticklabels([m[0] for m in metrics]); ax.set_ylabel("attacker success")
    ax.set_ylim(0, 1.0); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "fig5_privacy.png", dpi=220); plt.close(fig)


# ---------------------------------------------------------------- Table S3: per-class recall
def table_s3():
    rows = {}
    for arm, lab in [("local_only", "Local-only (site 0)"), ("fedavg", "FedAvg"), ("faithful_root_t-0.3", "PVBRF-ID (tau=-0.3)"), ("centralized", "Centralized")]:
        g = V5[(V5.arm == arm) & (V5.scenario == "none")]
        acc = {}
        for pc in g.per_class_recall:
            for k, v in json.loads(pc).items():
                acc.setdefault(k, []).append(v)
        rows[lab] = {k: round(float(np.mean(v)), 2) for k, v in acc.items()}
    t = pd.DataFrame(rows); t.index.name = "class"
    t.to_csv(OUT / "tableS3_per_class_recall.csv")
    return t


if __name__ == "__main__":
    fig2(); fig3(); fig4(); fig5()
    print(table_s3().to_string())
    print("figures written to", OUT)
