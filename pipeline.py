"""VeriCloud end-to-end: claim parsing -> hybrid retrieval -> stance -> independence -> source validation -> fusion."""
from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .world import World, make_sources, SUPPORTED, REFUTED, NEI, true_stance
from .textproc import Annotator, parse_claim, tokenize
from .retrieval import HybridRetriever
from .stance import StanceModel, train_stance_model
from .independence import cluster_weights
from .sources import SourceValidator, fit_prior_model, credit, credit_loo, sigmoid
from .fusion import fuse, fuse_samples, Decider, burst_multiplier


def train_prior(seed=4242):
    rng = np.random.default_rng(seed)
    src = [s for s in make_sources(rng, with_sybil=False)]
    meta = np.array([s.meta for s in src]); r = np.array([s.reliability for s in src])
    return fit_prior_model(meta, r)


@dataclass
class Config:
    k: int = 20
    theta: float = 0.80
    copy_weight: float = 0.10
    risk_q: float = 0.2            # quantile of the reliability posterior used for fusion (0.5 = median)
    kappa: float = 6.0
    lam_em: float = 0.0            # EM soft counts are optional: they did not help in our experiments (see E3)
    em_iters: int = 8
    em_temp: float = 1.0
    stance_noise: float = 0.0      # robustness experiments: fraction of stance outputs replaced by random draws
    retrieval_methods: tuple = ("bm25", "char")
    use_independence: bool = True
    use_burst: bool = True
    burst_window: float = 0.5
    burst_gamma: float = 0.5
    shrink_new: bool = True


class VeriCloud:
    def __init__(self, world: World, stance: StanceModel | None = None, prior_model=None, cfg: Config | None = None):
        self.w, self.cfg = world, cfg or Config()
        self.ann = Annotator(world.gaz)
        self.stance = stance or train_stance_model(self.ann)
        self.prior_model = prior_model or train_prior()
        self.src_names = [s.name for s in world.sources]
        self.art_src = np.array([a.source for a in world.articles])
        self.art_t = np.array([a.t for a in world.articles])
        self.via = [a.via_tag for a in world.articles]
        t0 = time.time()
        self.ret = HybridRetriever([a.text for a in world.articles], self.art_t, methods=self.cfg.retrieval_methods)
        self.index_seconds = time.time() - t0
        self._ann_cache = {}
        self.validator = SourceValidator(np.array([s.meta for s in world.sources]), self.prior_model, self.cfg.kappa, self.cfg.lam_em, self.cfg.shrink_new)
        self.decider = Decider()

    # ---------------------------------------------------------------- evidence
    def _sents(self, i):
        v = self._ann_cache.get(i)
        if v is None:
            v = self._ann_cache[i] = self.ann.annotate(self.w.articles[i].text)
        return v

    def build_table(self, claims=None, seed=0, retrieved=None):
        claims = claims if claims is not None else self.w.claims
        cfg = self.cfg
        top = retrieved if retrieved is not None else self.ret.search([c.text for c in claims], [c.t for c in claims], k=cfg.k)
        rng = np.random.default_rng(seed)
        rows = {k: [] for k in ("c", "a", "s", "rank", "w")}
        feats = []
        for ci, (c, ids) in enumerate(zip(claims, top)):
            ids = ids[ids >= 0]
            pc = parse_claim(c.text, self.ann)
            wts, _ = cluster_weights(ids, self.ret.wtf, self.art_t, self.art_src, self.src_names, self.via,
                                     cfg.theta, cfg.copy_weight)
            if not cfg.use_independence:
                wts = np.ones(len(ids))
            for r, (aid, wt) in enumerate(zip(ids, wts)):
                art = self.w.articles[aid]
                feats.append(self.stance.featurize(c.text, pc, self._sents(aid)))
                rows["c"].append(ci); rows["a"].append(aid); rows["s"].append(art.source); rows["rank"].append(r); rows["w"].append(wt)
        X = np.array(feats) if feats else np.zeros((0, 17))
        P = self.stance.predict_proba(X) if len(X) else np.zeros((0, 3))
        if cfg.stance_noise > 0:
            flip = rng.random(len(P)) < cfg.stance_noise
            P[flip] = rng.dirichlet(np.ones(3), size=flip.sum())
        t = {k: np.array(v) for k, v in rows.items()}
        t["t"] = self.art_t[t["a"]]
        t.update(pS=P[:, 0], pR=P[:, 1], pN=P[:, 2])
        t["true_stance"] = np.array([true_stance(self.w.articles[a], claims[c]) for c, a in zip(t["c"], t["a"])])
        return t

    # ---------------------------------------------------------------- source validation
    def learn_sources(self, table, claims, verified_mask, em_mask=None, use_verified=True, use_em=True):
        """verified_mask: claims whose human fact-check label may be used.  em_mask: claims over which EM runs."""
        n_s = len(self.w.sources); n_c = len(claims)
        truth = np.array([c.label for c in claims])
        V = self.validator
        V.ver_c[:] = V.ver_n[:] = V.em_c[:] = V.em_n[:] = 0
        known = verified_mask & (truth != NEI)
        if use_verified and known.any():
            c_, n_ = credit(table, (truth == SUPPORTED).astype(float), known, n_s)
            V.ver_c, V.ver_n = c_, n_
        if use_em:
            em_mask = np.ones(n_c, bool) if em_mask is None else em_mask
            em_claims = em_mask & ~(known if use_verified else np.zeros(n_c, bool))
            for _ in range(self.cfg.em_iters):
                S, m = fuse(table, V.mean(), n_c)
                V.em_c, V.em_n = credit_loo(table, S, V.mean(), self.cfg.em_temp, em_claims, n_s)
        return V

    def reliability(self):
        q = self.cfg.risk_q
        return self.validator.mean() if q == 0.5 else self.validator.quantile(q)

    # ---------------------------------------------------------------- decision
    def effective_table(self, table):
        """apply the collusion-aware burst discount (depends on the current track records)"""
        if not self.cfg.use_burst:
            return table
        t = dict(table)
        t["w"] = table["w"] * burst_multiplier(table, self.validator.n_obs(), self.cfg.burst_window, 2.0, self.cfg.burst_gamma)
        return t

    def scores(self, table, n_claims, rel=None):
        return fuse(self.effective_table(table), self.reliability() if rel is None else rel, n_claims)

    def calibrate(self, table, claims, mask):
        S, m = self.scores(table, len(claims))
        y = np.array([c.label for c in claims])
        self.decider.fit(S[mask], m[mask], y[mask])
        return self.decider

    def predict(self, table, claims, mc=0, seed=0):
        S, m = self.scores(table, len(claims))
        v, P = self.decider.verdict(S, m)
        out = dict(S=S, m=m, P=P, pred=v)
        if mc:
            rng = np.random.default_rng(seed)
            Ss = fuse_samples(self.effective_table(table), self.validator.sample(rng, mc), len(claims))
            Ps = self.decider.prob(Ss)
            # 80% range over posterior reliability samples, widened to include the (pessimistic) point estimate
            out["P_lo"] = np.minimum(np.percentile(Ps, 10, axis=0), out["P"])
            out["P_hi"] = np.maximum(np.percentile(Ps, 90, axis=0), out["P"])
            vs = np.full(Ss.shape, NEI)
            ok = m[None, :] >= self.decider.m_min
            vs[ok & (Ps >= .5 + self.decider.tau)] = SUPPORTED; vs[ok & (Ps <= .5 - self.decider.tau)] = REFUTED
            out["flip"] = (vs != v[None, :]).mean(0)
        return out
