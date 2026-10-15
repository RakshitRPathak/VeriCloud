"""E2b: adaptive attacker - aged domains (no new-domain red flag) and slow-drip posting (no burst). Honest stress test."""
from common import *
import time, pandas as pd
from vericloud import *
from vericloud.benchmark import run_methods, split_masks
from vericloud.pipeline import VeriCloud, Config

SEEDS = [300, 301, 302, 303]
SC = {"Naive attacker (new domains, burst)": dict(sybil_new_prob=0.7, campaign_spread=0.8),
      "Aged domains": dict(sybil_new_prob=0.0, campaign_spread=0.8),
      "Slow drip (6 days)": dict(sybil_new_prob=0.7, campaign_spread=6.0),
      "Aged domains + slow drip": dict(sybil_new_prob=0.0, campaign_spread=6.0)}
KEEP = ["Majority vote", "Top-1 evidence", "Truth discovery (EM, no independence)", "VeriCloud", "Oracle reliability"]


def job(a):
    name, seed = a; st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=0.3, campaign_strength=10, **SC[name]))
    nc = VeriCloud(w, stance=st, prior_model=pr); tab = nc.build_table(); masks = split_masks(w.claims)
    res, extra, y = run_methods(nc, tab, w.claims, masks)
    # VeriCloud without the two heuristic defences (newness shrink + burst) to show what they buy
    nc2 = VeriCloud.__new__(VeriCloud); nc2.__dict__.update(nc.__dict__)
    from dataclasses import replace
    nc2.cfg = replace(nc.cfg, use_burst=False, shrink_new=False)
    from vericloud.sources import SourceValidator
    nc2.validator = SourceValidator(np.array([s.meta for s in w.sources]), nc.prior_model, nc2.cfg.kappa, nc2.cfg.lam_em, False)
    nc2.decider = type(nc.decider)()
    nc2.learn_sources(tab, w.claims, np.array([c.verified for c in w.claims]) & masks[0]); nc2.calibrate(tab, w.claims, masks[1])
    pr_ = nc2.predict(tab, w.claims); from vericloud.metrics import evaluate
    res["VeriCloud without newness/burst defences"] = evaluate(y[masks[2]], pr_["pred"][masks[2]], pr_["P"][masks[2]])
    return [dict(attack=name, seed=seed, method=m, **r) for m, r in res.items() if m in KEEP + ["VeriCloud without newness/burst defences"]]


if __name__ == "__main__":
    t0 = time.time(); out = pmap(job, [(n, s) for n in SC for s in SEEDS])
    df = pd.DataFrame([r for o in out for r in o]); df.to_csv(RES / "e2b_adaptive_runs.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
    g = df.groupby(["method", "attack"]).macro_f1.mean().unstack()[list(SC)]
    order = KEEP[:4] + ["VeriCloud without newness/burst defences", "Oracle reliability"]
    g = g.loc[[m for m in order if m in g.index]]; print(g.round(3).to_string())
    (TAB / "e2b_adaptive.md").write_text(g.round(3).to_markdown())
    (TAB / "e2b_adaptive.tex").write_text(to_latex(g.round(3), "Macro-F1 against adaptive attackers (30\\% of claims targeted, 10 sybil articles each).", "tab:adaptive"))
    style(); fig, ax = plt.subplots(figsize=(9, 3.9)); w_ = 0.13; x = np.arange(len(SC))
    cols = {**PAL, "VeriCloud without newness/burst defences": "#e8a0a8"}
    for i, m in enumerate(g.index):
        ax.bar(x + (i - len(g) / 2 + .5) * w_, g.loc[m].values, w_, label=m, color=cols[m])
    ax.set_xticks(x); ax.set_xticklabels([k.replace(" (", "\n(") for k in SC], fontsize=8); ax.set_ylim(0.4, 0.9); ax.set_ylabel("macro-F1")
    ax.legend(fontsize=7, ncol=2); ax.set_title("Adaptive attackers"); savefig("e2b_adaptive")
