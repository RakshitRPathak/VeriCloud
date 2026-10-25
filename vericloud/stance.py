"""Claim-evidence stance model: gazetteer/cue features -> gradient boosted trees -> P(support, refute, neutral).
This is a lightweight stand-in for an NLI transformer: anything exposing `predict_proba(claims, annotations)` can replace it."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from .textproc import Annotator, parse_claim, tokenize
from .world import generate_world, WorldConfig, true_stance

FEATURES = ["subj_any", "n_subj_sents", "rel_match", "obj_in_subj_sent", "obj_any", "neg_obj_sent", "other_obj_sent",
            "neg_other_obj_sent", "hedge_subj_sent", "neg_any", "n_ents", "jaccard", "rel_mismatch", "claim_rel_cue",
            "obj_and_other", "obj_before_subj", "n_sents"]


def pair_features(pc, claim_tokens, sents, art_tokens_set):
    """pc: ParsedClaim; sents: list[Sentence] of the article"""
    f = np.zeros(len(FEATURES), np.float32)
    if pc.subject is None:
        return f
    S, O, rel = pc.subject, pc.obj, pc.rel
    otype = None
    subj_sents = [s for s in sents if any(e[1] == S for e in s.ents)]
    f[0] = float(bool(subj_sents)); f[1] = len(subj_sents); f[16] = len(sents)
    f[10] = sum(len(s.ents) for s in sents)
    f[9] = float(any(s.neg for s in sents))
    f[11] = len(set(claim_tokens) & art_tokens_set) / max(len(set(claim_tokens) | art_tokens_set), 1)
    f[13] = float(pc.rel_from_cue)
    f[4] = float(any(e[1] == O for s in sents for e in s.ents))
    for s in subj_sents:
        ents = s.ents
        otype = next((e[0] for e in ents if e[1] == O), None)
        has_o = any(e[1] == O for e in ents)
        # same-typed entity different from the claim object (and not the subject)
        others = [e for e in ents if e[1] not in (O, S) and e[0] == pc.obj_type]
        if rel in s.rels: f[2] = 1
        elif s.rels: f[12] = 1
        if has_o:
            f[3] = 1
            if s.neg: f[5] = 1
            ps = [e[2] for e in ents if e[1] == S][0]; po = [e[2] for e in ents if e[1] == O][0]
            if po < ps: f[15] = 1
        if others and not has_o:
            f[6] += 1
            if s.neg: f[7] = 1
        if others and has_o:
            f[14] = 1
        if s.hedge: f[8] = 1
    return f


class StanceModel:
    def __init__(self, annotator: Annotator, seed=0):
        self.ann = annotator
        self.clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        self.calls = 0

    def featurize(self, claim_text, parsed, article_text_or_ann, art_tokens=None):
        sents = article_text_or_ann if isinstance(article_text_or_ann, list) else self.ann.annotate(article_text_or_ann)
        ct = tokenize(claim_text)
        ats = art_tokens if art_tokens is not None else set(tokenize(" ".join(s.text for s in sents)))
        return pair_features(parsed, ct, sents, ats)

    def fit_on_world(self, world, negatives_per_claim=8, seed=0):
        rng = np.random.default_rng(seed)
        X, y = [], []
        ann_cache = {}
        by_fact = {}
        hard = {}
        for a in world.articles:
            by_fact.setdefault(a.fact, []).append(a)
            for (S_, r_, o_, _) in a.statements:
                hard.setdefault(("s", S_), []).append(a); hard.setdefault(("o", o_), []).append(a); hard.setdefault(("r", r_), []).append(a)
        for c in world.claims:
            pc = parse_claim(c.text, self.ann)
            pool = list(by_fact.get(c.fact, []))
            pool += [world.articles[int(i)] for i in rng.integers(len(world.articles), size=negatives_per_claim)]
            for key in (("s", c.subject), ("o", c.obj), ("r", c.rel)):          # hard negatives sharing subject / object / relation
                lst = hard.get(key, [])
                if lst:
                    pool += [lst[int(i)] for i in rng.integers(len(lst), size=min(3, len(lst)))]
            for a in pool:
                sents = ann_cache.get(a.id) or ann_cache.setdefault(a.id, self.ann.annotate(a.text))
                X.append(self.featurize(c.text, pc, sents)); y.append(true_stance(a, c))
        X = np.array(X); y = np.array(y)
        self.clf.fit(X, y)
        self.train_acc = float((self.clf.predict(X) == y).mean())
        return self

    def predict_proba(self, X):
        self.calls += len(X)
        P = np.zeros((len(X), 3))
        P[:, self.clf.classes_] = self.clf.predict_proba(X)
        return P


def train_stance_model(annotator, seed=999, n_facts=1200):
    w = generate_world(WorldConfig(n_facts=n_facts, seed=seed))
    return StanceModel(annotator).fit_on_world(w, seed=seed)
