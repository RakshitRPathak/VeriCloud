"""E4: component ablation (independence, burst discount, newness shrinkage, pessimistic quantile, learning) and
retrieval ablation (recall@k and downstream F1)."""
from common import *
import time, pandas as pd
from vericloud import *
from vericloud.benchmark import split_masks, clone_with, fit_eval
from vericloud.pipeline import VeriCloud, Config
from vericloud.world import true_stance

SEEDS = [500, 501, 502, 503, 504, 505]


def job_components(seed):
    st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=0.3, campaign_strength=10))
    nc = VeriCloud(w, stance=st, prior_model=pr); tab = nc.build_table(); masks = split_masks(w.claims); claims = w.claims
    ones = dict(tab); ones["w"] = np.ones_like(tab["w"])
    def cw(x):
        t = dict(tab); t["w"] = np.where(tab["w"] < 0.999, x, 1.0); return t
    V = {"VeriCloud (full)": (clone_with(nc), tab, {}),
         "- independence clustering": (clone_with(nc), ones, {}),
         "- burst discount": (clone_with(nc, use_burst=False), tab, {}),
         "- newness shrinkage": (clone_with(nc, shrink_new=False), tab, {}),
         "- pessimistic quantile (use median)": (clone_with(nc, risk_q=0.5), tab, {}),
         "- all three defences": (clone_with(nc, use_burst=False, shrink_new=False, risk_q=0.5), ones, {}),
         "- source learning (prior only)": (clone_with(nc), tab, dict(use_verified=False)),
         "+ truth-discovery EM (lambda=0.1)": (clone_with(nc, lam_em=0.1), tab, {}),
         "copy weight 0.0": (clone_with(nc), cw(0.0), {}), "copy weight 0.3": (clone_with(nc), cw(0.3), {}),
         "risk quantile 0.1": (clone_with(nc, risk_q=0.1), tab, {}), "risk quantile 0.3": (clone_with(nc, risk_q=0.3), tab, {})}
    rows = []
    for name, (n2, t2, kw) in V.items():
        r = fit_eval(n2, t2, claims, masks, **kw); rows.append(dict(seed=seed, variant=name, **r))
    return rows


def job_theta(a):
    theta, seed = a; st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=0.3, campaign_strength=10))
    nc = VeriCloud(w, stance=st, prior_model=pr, cfg=Config(theta=theta)); tab = nc.build_table(); masks = split_masks(w.claims)
    return [dict(seed=seed, theta=theta, **fit_eval(nc, tab, w.claims, masks))]


def job_retrieval(a):
    methods, seed = a; st, pr = models()
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=0.3, campaign_strength=10))
    nc = VeriCloud(w, stance=st, prior_model=pr, cfg=Config(retrieval_methods=methods))
    if "lsa" in methods:
        nc.ret.weights["lsa"] = 0.25
    claims = w.claims; top = nc.ret.search([c.text for c in claims], [c.t for c in claims], k=nc.cfg.k)
    rec = []
    rel_by_fact = {}
    for art in w.articles:
        for (S, r, o, pol) in art.statements:
            rel_by_fact.setdefault((S, r), []).append(art)
    for c, row in zip(claims, top):
        ids = {a.id for a in rel_by_fact.get((c.subject, c.rel), []) if a.t <= c.t}
        if ids:
            rec.append(len(ids & set(row.tolist())) / min(len(ids), nc.cfg.k))
    tab = nc.build_table(retrieved=top); masks = split_masks(claims)
    return [dict(seed=seed, retrieval="+".join(methods), recall_at_20=float(np.mean(rec)), **fit_eval(nc, tab, claims, masks))]


if __name__ == "__main__":
    t0 = time.time()
    comp = pd.DataFrame([r for o in pmap(job_components, SEEDS) for r in o]); comp.to_csv(RES / "e4_components_runs.csv", index=False)
    th = pd.DataFrame([r for o in pmap(job_theta, [(t, s) for t in (0.6, 0.7, 0.8, 0.9, 0.97) for s in SEEDS[:3]]) for r in o]); th.to_csv(RES / "e4_theta_runs.csv", index=False)
    rt = pd.DataFrame([r for o in pmap(job_retrieval, [(m, s) for m in (("bm25",), ("char",), ("bm25", "char"), ("bm25", "char", "lsa")) for s in SEEDS[:3]]) for r in o]); rt.to_csv(RES / "e4_retrieval_runs.csv", index=False)
    print(f"done {time.time()-t0:.0f}s")
    g = comp.groupby("variant")[["macro_f1", "acc", "precision_asserted", "ece"]].agg(["mean", "std"])
    order = list(dict.fromkeys(comp.variant))
    tab = pd.DataFrame({v: {"Macro-F1": f"{g.loc[v,('macro_f1','mean')]:.3f}±{g.loc[v,('macro_f1','std')]:.3f}", "Acc": f"{g.loc[v,('acc','mean')]:.3f}",
                            "Prec. (asserted)": f"{g.loc[v,('precision_asserted','mean')]:.3f}", "ECE": f"{g.loc[v,('ece','mean')]:.3f}"} for v in order}).T
    print(tab.to_string()); (TAB / "e4_components.md").write_text(tab.to_markdown())
    (TAB / "e4_components.tex").write_text(to_latex(tab, "Component ablation under attack (30\\% targeted, 10 sybil articles; mean$\\pm$sd over 6 worlds).", "tab:ablation"))
    # paired differences vs full
    piv = comp.pivot(index="seed", columns="variant", values="macro_f1")
    from scipy import stats
    diffs = {v: dict(mean_diff=float((piv[v] - piv["VeriCloud (full)"]).mean()), wilcoxon_p=float(stats.wilcoxon(piv[v], piv["VeriCloud (full)"]).pvalue) if v != "VeriCloud (full)" else None) for v in order}
    save_json("e4_ablation_paired.json", diffs); print(json.dumps(diffs, indent=1))
    print(th.groupby("theta")[["macro_f1"]].mean().round(3).T.to_string())
    print(rt.groupby("retrieval")[["recall_at_20", "macro_f1"]].mean().round(3).to_string())
    (TAB / "e4_retrieval.md").write_text(rt.groupby("retrieval")[["recall_at_20", "macro_f1"]].mean().round(3).to_markdown())
    (TAB / "e4_theta.md").write_text(th.groupby("theta")[["macro_f1"]].mean().round(3).to_markdown())
    style(); fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    m = comp.groupby("variant").macro_f1.agg(["mean", "std"]).reindex(order)
    ax[0].barh(range(len(m)), m["mean"], xerr=m["std"], color=["#c8102e" if v == "VeriCloud (full)" else "#9aa0a6" for v in m.index], capsize=2)
    ax[0].set_yticks(range(len(m))); ax[0].set_yticklabels(m.index, fontsize=8); ax[0].invert_yaxis(); ax[0].set_xlim(0.55, 0.85); ax[0].set_xlabel("macro-F1"); ax[0].set_title("Component ablation")
    g2 = th.groupby("theta").macro_f1.agg(["mean", "std"]); ax[1].errorbar(g2.index, g2["mean"], g2["std"], marker="o", color="#c8102e"); ax[1].set_xlabel("duplicate threshold θ"); ax[1].set_title("Clustering threshold")
    g3 = rt.groupby("retrieval")[["recall_at_20", "macro_f1"]].mean(); x = np.arange(len(g3))
    ax[2].bar(x - .2, g3.recall_at_20, .4, label="recall@20", color="#4f86c6"); ax[2].bar(x + .2, g3.macro_f1, .4, label="macro-F1", color="#c8102e")
    ax[2].set_xticks(x); ax[2].set_xticklabels([s.replace("+", "+\n") for s in g3.index], fontsize=7); ax[2].set_ylim(0.5, 1.0); ax[2].legend(fontsize=7); ax[2].set_title("Retrieval")
    savefig("e4_ablation")
