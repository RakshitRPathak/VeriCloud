"""Source validation: Bayesian per-source reliability with (1) a metadata prior learned on a separate set of rated
sources, (2) verified outcomes from human fact-checks, (3) copy-aware truth-discovery EM over unlabeled claims."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.linear_model import Ridge


def sigmoid(x): return 1 / (1 + np.exp(-x))
def logit(p): return np.log(p / (1 - p))


def fit_prior_model(meta_train, r_train, alpha=1.0):
    return Ridge(alpha=alpha).fit(meta_train, logit(np.clip(r_train, .03, .97)))


class SourceValidator:
    def __init__(self, meta, prior_model, kappa=6.0, lam_em=0.5, shrink_new=True):
        self.kappa, self.lam = kappa, lam_em
        self.mu0 = np.clip(sigmoid(prior_model.predict(meta)), .05, .95)
        if shrink_new:       # domains younger than one year start near a coin flip until they build a track record
            age = np.expm1(meta[:, 0])
            f = np.clip(age / 1.0, 0.1, 1.0)
            self.mu0 = 0.5 + (self.mu0 - 0.5) * f
        self.a0, self.b0 = self.mu0 * kappa, (1 - self.mu0) * kappa
        n = len(meta)
        self.ver_c = np.zeros(n); self.ver_n = np.zeros(n); self.em_c = np.zeros(n); self.em_n = np.zeros(n)

    def ab(self):
        a = self.a0 + self.ver_c + self.lam * self.em_c
        b = self.b0 + (self.ver_n - self.ver_c) + self.lam * (self.em_n - self.em_c)
        return a, b

    def mean(self):
        a, b = self.ab(); return a / (a + b)

    def quantile(self, q):
        a, b = self.ab(); return beta_dist.ppf(q, a, b)

    def sample(self, rng, m):
        a, b = self.ab(); return rng.beta(a, b, size=(m, len(a)))

    def n_obs(self):
        return self.ver_n + self.lam * self.em_n


def credit(table, p_true, claim_mask, n_sources):
    """expected (#correct, #total) stance-bearing statements per source given P(claim true); copies get w=copy_weight"""
    m = claim_mask[table["c"]]
    pt = p_true[table["c"]]
    corr = table["w"] * (table["pS"] * pt + table["pR"] * (1 - pt)) * m
    tot = table["w"] * (table["pS"] + table["pR"]) * m
    return (np.bincount(table["s"], corr, minlength=n_sources), np.bincount(table["s"], tot, minlength=n_sources))


def credit_loo(table, S, rel, temp, claim_mask, n_sources):
    """Leave-one-out truth-discovery credit: an evidence item is judged against the posterior computed from the OTHER
    evidence of its claim, so a lone source cannot confirm itself (no circular self-reinforcement)."""
    r = np.clip(rel, .5, .98)
    contrib = table["w"] * (table["pS"] - table["pR"]) * logit(r)[table["s"]]
    p = sigmoid(temp * (S[table["c"]] - contrib))
    m = claim_mask[table["c"]]
    corr = table["w"] * (table["pS"] * p + table["pR"] * (1 - p)) * m
    tot = table["w"] * (table["pS"] + table["pR"]) * m
    return (np.bincount(table["s"], corr, minlength=n_sources), np.bincount(table["s"], tot, minlength=n_sources))
