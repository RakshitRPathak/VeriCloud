"""Shared benchmark harness: build once, evaluate VeriCloud + baselines + oracle upper bounds on the same evidence."""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr
from .world import WorldConfig, generate_world, NEI, SUPPORTED, REFUTED
from .pipeline import VeriCloud, Config
from .fusion import Decider, fuse
from .metrics import evaluate


def split_masks(claims, fr=(0.5, 0.2, 0.3)):
    n = len(claims); o = np.argsort([c.t for c in claims])
    h, v, t = (np.zeros(n, bool) for _ in range(3))
    a, b = int(fr[0] * n), int((fr[0] + fr[1]) * n)
    h[o[:a]] = True; v[o[a:b]] = True; t[o[b:]] = True
    return h, v, t


def with_weights(table, w):
    t = dict(table); t["w"] = w; return t


def run_methods(nc: VeriCloud, table, claims, masks, include_oracles=True, mc=0):
    hist, val, test = masks
    n = len(claims); y = np.array([c.label for c in claims])
    verified = np.array([c.verified for c in claims]) & hist
    true_rel = nc.w.source_reliability()
    ones = np.ones_like(table["w"])
    out, extra = {}, {}

    def score_eval(name, tab, rel, uncertainty=False, burst=False):
        S, m = fuse(nc.effective_table(tab) if burst else tab, rel, n)
        d = Decider().fit(S[val], m[val], y[val])
        pred, P = d.verdict(S, m)
        out[name] = evaluate(y[test], pred[test], P[test])
        extra[name] = dict(S=S, m=m, P=P, pred=pred, decider=d)

    # --- baselines (same retrieval + stance evidence, same decision tuning)
    first = {}
    for i, (c, r) in enumerate(zip(table["c"], table["rank"])):
        if table["pS"][i] + table["pR"][i] > .5 and c not in first:
            first[c] = i
    t1 = dict(table); t1["w"] = np.zeros_like(table["w"])
    for i in first.values(): t1["w"][i] = 1.0
    score_eval("Top-1 evidence", t1, np.full(len(true_rel), .73))
    score_eval("Majority vote", with_weights(table, ones), np.full(len(true_rel), .73))
    nc.learn_sources(table, claims, verified, use_verified=False, use_em=False)
    score_eval("Credibility prior (no learning)", with_weights(table, ones), nc.validator.mean())
    lam0 = nc.validator.lam; nc.validator.lam = 0.5          # the EM baseline must actually use its soft counts (VeriCloud's default lam_em is 0)
    nc.learn_sources(with_weights(table, ones), claims, verified, use_verified=False, use_em=True)
    score_eval("Truth discovery (EM, no independence)", with_weights(table, ones), nc.validator.mean())
    nc.validator.lam = lam0
    # --- VeriCloud: verified labels + EM + independence weights (+ optional pessimistic quantile)
    nc.learn_sources(table, claims, verified)
    score_eval("VeriCloud", table, nc.reliability(), burst=True)
    extra["VeriCloud"]["spearman"] = None
    if include_oracles:
        score_eval("Oracle reliability", table, true_rel)
        T = table["true_stance"]; ot = dict(table); ot["pS"] = (T == 0).astype(float); ot["pR"] = (T == 1).astype(float)
        score_eval("Oracle reliability + stance", ot, true_rel)
    return out, extra, y


def run_world(wcfg: WorldConfig, ncfg: Config | None = None, stance=None, prior=None, fr=(0.5, 0.2, 0.3), seed_noise=0, include_oracles=True, retrieved=None):
    w = generate_world(wcfg)
    nc = VeriCloud(w, stance=stance, prior_model=prior, cfg=ncfg)
    tab = nc.build_table(seed=seed_noise)
    masks = split_masks(w.claims, fr)
    res, extra, y = run_methods(nc, tab, w.claims, masks, include_oracles)
    return dict(world=w, nc=nc, table=tab, masks=masks, results=res, extra=extra, y=y)


def clone_with(nc: VeriCloud, **overrides) -> VeriCloud:
    """shallow copy of a VeriCloud sharing index/stance/annotation caches but with its own config, validator, decider"""
    import copy
    from dataclasses import replace
    from .sources import SourceValidator
    c2 = copy.copy(nc)
    c2.cfg = replace(nc.cfg, **overrides)
    c2.validator = SourceValidator(np.array([s.meta for s in nc.w.sources]), nc.prior_model, c2.cfg.kappa, c2.cfg.lam_em, c2.cfg.shrink_new)
    c2.decider = Decider()
    return c2


def fit_eval(nc2: VeriCloud, table, claims, masks, use_verified=True, use_em=None, y=None):
    hist, val, test = masks
    y = np.array([c.label for c in claims]) if y is None else y
    use_em = nc2.cfg.lam_em > 0 if use_em is None else use_em
    ver = np.array([c.verified for c in claims]) & hist
    nc2.learn_sources(table, claims, ver, use_verified=use_verified, use_em=use_em)
    nc2.calibrate(table, claims, val)
    pr = nc2.predict(table, claims)
    return evaluate(y[test], pr["pred"][test], pr["P"][test])
