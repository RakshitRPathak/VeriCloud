"""Independence- and credibility-weighted evidence fusion in log-odds space, Platt calibration, abstention."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from .sources import logit, sigmoid
from .world import SUPPORTED, REFUTED, NEI


def fuse(table, rel, n_claims, use_indep=True):
    """S_c = sum_i w_i (P_sup,i - P_ref,i) * max(logit(r_src(i)), 0);   m_c = sum_i w_i (P_sup,i + P_ref,i)
    A source judged less reliable than a coin flip is IGNORED, never inverted (inversion would let an attacker steer
    verdicts by deliberately burning accounts)."""
    r = np.clip(rel, .5, .98)
    w = table["w"] if use_indep else np.ones_like(table["w"])
    S = np.bincount(table["c"], w * (table["pS"] - table["pR"]) * logit(r)[table["s"]], minlength=n_claims)
    m = np.bincount(table["c"], w * (table["pS"] + table["pR"]), minlength=n_claims)
    return S, m


def fuse_samples(table, R, n_claims, use_indep=True):
    """Monte-Carlo S for sampled reliability matrices R (n_samples, n_sources)."""
    R = np.clip(R, .5, .98)
    w = table["w"] if use_indep else np.ones_like(table["w"])
    coef = w * (table["pS"] - table["pR"])
    M = sp.csr_matrix((np.ones(len(coef)), (np.arange(len(coef)), table["c"])), shape=(len(coef), n_claims))
    LL = logit(R)[:, table["s"]] * coef[None, :]
    return np.asarray(LL @ M)


class Decider:
    """P = sigmoid(a*S + b) (Platt) ; verdict by evidence-mass gate m >= m_min and confidence gate |P-.5| >= tau"""
    def __init__(self):
        self.a, self.b, self.m_min, self.tau = 1.0, 0.0, 0.5, 0.2

    def prob(self, S):
        return sigmoid(self.a * S + self.b)

    def verdict(self, S, m):
        P = self.prob(S)
        v = np.full(len(S), NEI)
        ok = m >= self.m_min
        v[ok & (P >= .5 + self.tau)] = SUPPORTED
        v[ok & (P <= .5 - self.tau)] = REFUTED
        return v, P

    def fit(self, S, m, y):
        """y in {SUPPORTED, REFUTED, NEI}.  Platt on S/R claims, then grid-search gates for 3-way macro-F1."""
        from sklearn.metrics import f1_score
        cov = y != NEI
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(S[cov, None], (y[cov] == SUPPORTED).astype(int))
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        if self.a <= 1e-3:          # monotonicity constraint: more support must never mean "more likely false"
            pr = np.clip((y[cov] == SUPPORTED).mean(), .02, .98)
            self.a, self.b = 1e-3, float(np.log(pr / (1 - pr)))
        best = (-1, None)
        for mm in [0.1, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]:
            for tau in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.44]:
                self.m_min, self.tau = mm, tau
                f = f1_score(y, self.verdict(S, m)[0], average="macro", labels=[0, 1, 2])
                if f > best[0] + 1e-9:
                    best = (f, (mm, tau))
        self.m_min, self.tau = best[1]
        return self


def burst_multiplier(table, n_obs, window=0.5, n0=2.0, gamma=0.5):
    """Collusion-aware discount.  For an evidence item from a source with no track record (n_obs < n0), count the other
    same-stance items of the same claim that appeared within `window` days from sources that are also unproven;
    multiplier = 1 / (1 + gamma * count).  Established sources are never discounted."""
    c, s, t = table["c"], table["s"], table["t"]
    sign = np.sign(table["pS"] - table["pR"]) * ((table["pS"] + table["pR"]) > 0.5)
    unproven = n_obs[s] < n0
    mult = np.ones(len(c))
    starts = np.flatnonzero(np.r_[True, c[1:] != c[:-1]]); ends = np.r_[starts[1:], len(c)]
    for a, b in zip(starts, ends):
        if b - a < 2:
            continue
        idx = np.arange(a, b)
        sg, un, tt = sign[idx], unproven[idx], t[idx]
        same = (sg[:, None] == sg[None, :]) & (sg[:, None] != 0) & un[None, :] & (np.abs(tt[:, None] - tt[None, :]) <= window)
        np.fill_diagonal(same, False)
        cnt = same.sum(1)
        mult[idx] = np.where(un & (sg != 0), 1.0 / (1.0 + gamma * cnt), 1.0)
    return mult
