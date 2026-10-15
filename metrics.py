from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from .world import SUPPORTED, REFUTED, NEI


def ece(p, y, bins=10):
    """expected calibration error of P(claim true) on covered claims; y in {0,1}"""
    e, n = 0.0, len(p)
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def reliability_curve(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() >= 5:
            xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def risk_coverage(conf, correct):
    """selective prediction: sort by confidence, report error among the most-confident fraction"""
    o = np.argsort(-conf); c = correct[o].astype(float)
    cov = np.arange(1, len(c) + 1) / len(c); risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return cov, risk, float(np.trapezoid(risk, cov))


def evaluate(y, pred, P, S_conf_mask=None):
    """y,pred in {0,1,2}; P = P(claim supported). Returns dict of headline metrics."""
    out = dict(acc=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro", labels=[0, 1, 2]))
    cov = y != NEI
    asserted = pred != NEI
    out["acc_covered"] = float((pred[cov] == y[cov]).mean())
    out["abstain_rate_covered"] = float((pred[cov] == NEI).mean())
    out["precision_asserted"] = float((pred[asserted] == y[asserted]).mean()) if asserted.any() else float("nan")
    out["nei_recall"] = float((pred[~cov] == NEI).mean()) if (~cov).any() else float("nan")
    out["false_assert_rate_on_nei"] = float((pred[~cov] != NEI).mean()) if (~cov).any() else float("nan")
    m = cov
    out["ece"] = ece(P[m], (y[m] == SUPPORTED).astype(float))
    out["brier"] = float(np.mean((P[m] - (y[m] == SUPPORTED)) ** 2))
    return out
