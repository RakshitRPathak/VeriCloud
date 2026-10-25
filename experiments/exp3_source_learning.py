"""E3: how fast does source validation learn?  Reliability-estimate quality and downstream F1 vs #human fact-checks."""
from common import *
import time, pandas as pd
from scipy.stats import spearmanr
from vericloud import *
from vericloud.benchmark import split_masks
from vericloud.pipeline import VeriCloud
from vericloud.metrics import evaluate

SEEDS = [400, 401, 402, 403]
BUDGETS = [0, 20, 50, 100, 200, 400, 700]


def job(seed):
    st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=1.0, campaign_share=0.3, campaign_strength=10))
    nc = VeriCloud(w, stance=st, prior_model=pr); tab = nc.build_table(); claims = w.claims; n = len(claims)
    hist, val, test = split_masks(claims)
    y = np.array([c.label for c in claims]); tr = w.source_reliability()
    is_syn = np.array([s.is_sybil for s in w.sources]); active = np.bincount(tab["s"], minlength=len(tr)) > 0
    honest = active & ~is_syn; syb = active & is_syn
    rng = np.random.default_rng(seed); cand = np.flatnonzero(hist & (y != NEI)); rng.shuffle(cand)
    rows = []
    for B in BUDGETS:
        ver = np.zeros(n, bool); ver[cand[:B]] = True
        for name, kw, lam in [("Metadata prior only", dict(use_verified=False, use_em=False), 0.1),
                              ("Verified labels (VeriCloud default)", dict(use_em=False), 0.1),
                              ("Truth-discovery EM only", dict(use_verified=False), 0.1),
                              ("Verified + EM (optional)", {}, 0.1)]:
            if name == "Truth-discovery EM only" and B > 0: continue         # EM-only does not use labels: one point
            nc.validator.lam = lam
            nc.learn_sources(tab, claims, ver, **kw)
            r = nc.validator.mean(); nc.decider = type(nc.decider)()
            nc.calibrate(tab, claims, val); pr_ = nc.predict(tab, claims)
            rows.append(dict(seed=seed, budget=B, method=name, spearman=spearmanr(r[honest], tr[honest])[0], mae=np.abs(r - tr)[honest].mean(),
                             sybil_mean_rel=float(r[syb].mean()) if syb.any() else np.nan,
                             macro_f1=evaluate(y[test], pr_["pred"][test], pr_["P"][test])["macro_f1"]))
    return rows


if __name__ == "__main__":
    t0 = time.time(); out = pmap(job, SEEDS)
    df = pd.DataFrame([r for o in out for r in o]); df.to_csv(RES / "e3_source_learning_runs.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
    em0 = df[df.method == "Truth-discovery EM only"].groupby("budget").mean(numeric_only=True)
    style(); fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    cols = {"Metadata prior only": "#d98e04", "Verified labels (VeriCloud default)": "#4f86c6", "Verified + EM (optional)": "#c8102e"}
    for k, (col, yl) in enumerate([("spearman", "Spearman ρ (estimated vs true reliability)"), ("mae", "MAE of reliability estimate"), ("macro_f1", "downstream macro-F1")]):
        for m, c in cols.items():
            g = df[df.method == m].groupby("budget")[col].agg(["mean", "std"])
            ax[k].errorbar(np.maximum(g.index, 5), g["mean"], g["std"], marker="o", ms=4, color=c, label=m, capsize=2)
        ax[k].axhline(em0[col].iloc[0], color="#2a9d8f", ls="--", label="Truth-discovery EM only (0 labels)")
        ax[k].set_xscale("log"); ax[k].set_xlabel("# human fact-checked claims (0 plotted at 5)"); ax[k].set_ylabel(yl)
    ax[0].legend(fontsize=7); savefig("e3_source_learning")
    g = df.groupby(["method", "budget"])[["spearman", "mae", "macro_f1", "sybil_mean_rel"]].mean().round(3)
    print(g.to_string()); (TAB / "e3_source_learning.md").write_text(g.to_markdown())
