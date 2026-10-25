"""E2: robustness to coordinated sybil campaigns - sweep campaign share and campaign strength."""
from common import *
import time, pandas as pd
from vericloud import *
from vericloud.benchmark import run_methods, split_masks
from vericloud.pipeline import VeriCloud

SEEDS = [200, 201, 202, 203]
CONF = [("share", s, 10) for s in (0.0, 0.15, 0.3, 0.45, 0.6)] + [("strength", 0.3, k) for k in (5, 20, 30)]
KEEP = ["Majority vote", "Top-1 evidence", "Credibility prior (no learning)", "Truth discovery (EM, no independence)", "VeriCloud", "Oracle reliability"]


def job(a):
    kind, share, strength, seed = a
    st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=share, campaign_strength=strength))
    nc = VeriCloud(w, stance=st, prior_model=pr); tab = nc.build_table(); masks = split_masks(w.claims)
    res, extra, y = run_methods(nc, tab, w.claims, masks)
    te = masks[2]; tg = np.array([c.targeted for c in w.claims])[te]
    yt = y[te]; rows = []
    for m in KEEP:
        pred = extra[m]["pred"][te]
        rows.append(dict(kind=kind, share=share, strength=strength, seed=seed, method=m, **res[m],
                         acc_targeted=float((pred[tg] == yt[tg]).mean()) if tg.any() else np.nan,
                         acc_untargeted=float((pred[~tg] == yt[~tg]).mean())))
    return rows


if __name__ == "__main__":
    t0 = time.time()
    out = pmap(job, [(k, s, n, seed) for (k, s, n) in CONF for seed in SEEDS])
    df = pd.DataFrame([r for o in out for r in o]); df.to_csv(RES / "e2_adversarial_runs.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
    style(); fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    for k, (kind, xcol, xl, col, ttl) in enumerate([("share", "share", "fraction of claims targeted by a campaign (10 sybil articles each)", "macro_f1", "Macro-F1 vs attack breadth"),
                                                   ("strength", "strength", "sybil articles per targeted claim (30% of claims targeted)", "macro_f1", "Macro-F1 vs attack intensity"),
                                                   ("share", "share", "fraction of claims targeted", "acc_targeted", "Accuracy on the targeted claims")]):
        d = df[df.kind == kind]
        if kind == "strength":
            d = pd.concat([d, df[(df.kind == "share") & (df.share == 0.3)].assign(strength=10)])
        for m in KEEP:
            g = d[d.method == m].groupby(xcol)[col].agg(["mean", "std"])
            ax[k].errorbar(g.index, g["mean"], g["std"], marker="o", ms=4, lw=2.2 if m == "VeriCloud" else 1.4, color=PAL[m],
                           ls="--" if "Oracle" in m else "-", label=m, capsize=2)
        ax[k].set_xlabel(xl, fontsize=8); ax[k].set_title(ttl)
    ax[0].legend(fontsize=7); savefig("e2_adversarial")
    g = df[(df.kind == "share")].groupby(["method", "share"]).macro_f1.mean().unstack().loc[KEEP]
    print(g.round(3).to_string())
    gs = df[(df.kind == "strength") | ((df.kind == "share") & (df.share == 0.3))]
    gs = gs.assign(strength=gs.strength).groupby(["method", "strength"]).macro_f1.mean().unstack().loc[KEEP]; print(gs.round(3).to_string())
    (TAB / "e2_attack_breadth.md").write_text(g.round(3).to_markdown()); (TAB / "e2_attack_intensity.md").write_text(gs.round(3).to_markdown())
    (TAB / "e2_attack_breadth.tex").write_text(to_latex(g.round(3).rename(columns=lambda c: f"{int(c*100)}\\%"), "Macro-F1 as the fraction of claims targeted by sybil campaigns grows (10 sybil articles per target).", "tab:attack_breadth"))
