"""E1: main benchmark - VeriCloud vs baselines (clean / under coordinated attack / attack + noisy stance model)."""
from common import *
import time, pandas as pd
from scipy import stats
from vericloud import *
from vericloud.benchmark import run_world, run_methods, split_masks
from vericloud.pipeline import VeriCloud, Config
from vericloud.metrics import evaluate

SETTINGS = {"Clean (no campaign)": dict(campaign_share=0.0, campaign_strength=0, noise=0.0),
            "Under attack (30% of claims targeted, 10 sybil articles each)": dict(campaign_share=0.3, campaign_strength=10, noise=0.0),
            "Attack + noisy stance model (10% random stance outputs)": dict(campaign_share=0.3, campaign_strength=10, noise=0.10)}
SEEDS = list(range(100, 108))


def job(args):
    seed, name = args
    st, pr = models(); s = SETTINGS[name]
    w = generate_world(WorldConfig(seed=seed, labeled_frac=0.2, campaign_share=s["campaign_share"], campaign_strength=s["campaign_strength"]))
    nc = VeriCloud(w, stance=st, prior_model=pr)
    tab = apply_stance_noise(nc.build_table(), s["noise"], seed)
    masks = split_masks(w.claims)
    res, extra, y = run_methods(nc, tab, w.claims, masks)
    te = masks[2]
    pooled = {k: dict(P=extra[k]["P"][te], pred=extra[k]["pred"][te], S=extra[k]["S"][te], m=extra[k]["m"][te]) for k in extra}
    return dict(seed=seed, setting=name, res=res, y=y[te], pooled=pooled, n_articles=len(w.articles), n_claims=len(w.claims))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(s, n) for n in SETTINGS for s in SEEDS]
    out = pmap(job, jobs)
    print(f"done in {time.time()-t0:.0f}s")
    rows = []
    for o in out:
        for m, r in o["res"].items():
            rows.append(dict(setting=o["setting"], seed=o["seed"], method=m, **r))
    df = pd.DataFrame(rows); df.to_csv(RES / "e1_main_runs.csv", index=False)
    import pickle; pickle.dump([dict(setting=o["setting"], seed=o["seed"], y=o["y"], pooled=o["pooled"]) for o in out], open(RES / "_e1_pooled.pkl", "wb"))
    summary = {}
    order = list(PAL)
    for st_ in SETTINGS:
        d = df[df.setting == st_]
        g = d.groupby("method")[["acc", "macro_f1", "ece", "precision_asserted", "acc_covered", "nei_recall"]].agg(["mean", "std"])
        summary[st_] = {m: {k: [float(g.loc[m, (k, "mean")]), float(g.loc[m, (k, "std")])] for k in ["acc", "macro_f1", "ece", "precision_asserted", "acc_covered", "nei_recall"]} for m in order if m in g.index}
        # paired test VeriCloud vs strongest non-oracle baseline (by mean macro-F1)
        base = [m for m in order if m in g.index and m not in ("VeriCloud", "Oracle reliability", "Oracle reliability + stance")]
        best = max(base, key=lambda m: g.loc[m, ("macro_f1", "mean")])
        a = d[d.method == "VeriCloud"].sort_values("seed").macro_f1.values; b = d[d.method == best].sort_values("seed").macro_f1.values
        p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else 1.0
        summary[st_]["_paired_vs_best_baseline"] = dict(best_baseline=best, mean_diff_macro_f1=float((a - b).mean()), wilcoxon_p=float(p), n_seeds=len(a))
        tab = pd.DataFrame({m: {"Acc": f"{g.loc[m,('acc','mean')]:.3f}±{g.loc[m,('acc','std')]:.3f}", "Macro-F1": f"{g.loc[m,('macro_f1','mean')]:.3f}±{g.loc[m,('macro_f1','std')]:.3f}",
                                "Prec.\\ (asserted)": f"{g.loc[m,('precision_asserted','mean')]:.3f}", "ECE": f"{g.loc[m,('ece','mean')]:.3f}"} for m in order if m in g.index}).T
        key = st_.split(" (")[0].split(" +")[0].lower().replace(" ", "_")
        (TAB / f"e1_{key}.tex").write_text(to_latex(tab, f"Main benchmark -- {st_} (mean$\\pm$sd over {len(SEEDS)} worlds; test split).", f"tab:main_{key}"))
        (TAB / f"e1_{key}.md").write_text(tab.to_markdown())
        print("\n##", st_); print(tab.to_string()); print(summary[st_]["_paired_vs_best_baseline"])
    save_json("e1_main_summary.json", summary)
    # figure
    style(); fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
    for k, st_ in enumerate(SETTINGS):
        d = df[df.setting == st_].groupby("method").macro_f1.agg(["mean", "std"]).reindex([m for m in order if m in df.method.unique()])
        ax[k].barh(range(len(d)), d["mean"], xerr=d["std"], color=[PAL[m] for m in d.index], capsize=2)
        ax[k].set_yticks(range(len(d))); ax[k].set_yticklabels(d.index if k == 0 else [], fontsize=8); ax[k].invert_yaxis()
        ax[k].set_title(st_.split(" (")[0] if k < 2 else "Attack + noisy stance", fontsize=10); ax[k].set_xlim(0.4, 0.9); ax[k].set_xlabel("macro-F1 (3-way)")
    savefig("e1_main_benchmark")
