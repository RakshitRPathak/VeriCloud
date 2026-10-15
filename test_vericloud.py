import numpy as np
import pytest
from scipy.sparse import csr_matrix
from vericloud import *
from vericloud.world import true_stance, Article, Claim, VERDICTS
from vericloud.textproc import Annotator, parse_claim, tokenize
from vericloud.retrieval import HybridRetriever
from vericloud.independence import cluster_weights
from vericloud.fusion import fuse, Decider, burst_multiplier
from vericloud.sources import SourceValidator, fit_prior_model, credit, credit_loo, logit
from vericloud.metrics import ece, risk_coverage
from vericloud.report import build_report, to_markdown, verify_text


# ---------------------------------------------------------------- world ground truth
def test_world_is_deterministic_and_consistent():
    a = generate_world(WorldConfig(n_facts=300, seed=3)); b = generate_world(WorldConfig(n_facts=300, seed=3))
    assert [x.text for x in a.articles[:50]] == [x.text for x in b.articles[:50]]
    assert all(x.t <= y.t for x, y in zip(a.articles, a.articles[1:]))
    for art in a.articles:
        assert art.copy_of < art.id                             # copies come after their origin
    assert {c.label for c in a.claims} == {0, 1, 2}


def test_labels_match_structure_and_nei_has_no_evidence():
    w = generate_world(WorldConfig(n_facts=500, seed=4))
    keys = {(f["S"], f["rel"]): f for f in w.facts}
    assert len(keys) == len(w.facts)                            # unique (subject, relation) per fact
    for c in w.claims:
        f = w.facts[c.fact]
        if c.label == NEI:
            assert not f["covered"]
            assert not any(s[0] == c.subject and s[1] == c.rel for a in w.articles for s in a.statements)
        else:
            assert (c.label == SUPPORTED) == (c.obj == f["O"])


def test_true_stance_rules():
    c = Claim(0, "", "A", 0, "X", 0, 0, 0.0, False)
    mk = lambda stm: Article(0, 0, 0.0, "", 0, stm)
    assert true_stance(mk([("A", 0, "X", 1)]), c) == 0
    assert true_stance(mk([("A", 0, "Y", 1)]), c) == 1
    assert true_stance(mk([("A", 0, "X", -1)]), c) == 1
    assert true_stance(mk([("A", 0, "Y", -1)]), c) == 2         # denying something else says nothing about the claim
    assert true_stance(mk([("B", 0, "X", 1)]), c) == 2
    assert true_stance(mk([("A", 0, "X", -1), ("A", 0, "Y", 1)]), c) == 1


# ---------------------------------------------------------------- text
def test_claim_parser_and_annotator():
    w = generate_world(WorldConfig(n_facts=200, seed=5)); ann = Annotator(w.gaz)
    ok = sum(parse_claim(c.text, ann).obj == c.obj and parse_claim(c.text, ann).subject == c.subject for c in w.claims)
    assert ok / len(w.claims) > 0.97
    s = ann.annotate("Fact check: it is false that Maya Verhoeven visited Valmere. Reportedly, Omar Haddad quit Orbis Motors.")
    assert s[0].neg and not s[0].hedge and s[1].hedge and not s[1].neg


def test_bm25_ranks_exact_match_first():
    texts = ["alpha beta gamma", "delta epsilon zeta", "alpha delta", "eta theta iota"] * 10
    R = HybridRetriever(texts, np.arange(40.0), methods=("bm25",))
    top = R.search(["epsilon zeta"], [100.0], k=3)[0]
    assert tokenize(texts[top[0]]) == ["delta", "epsilon", "zeta"]


def test_retrieval_is_time_causal():
    texts = [f"story {i} about topic{i % 3}" for i in range(30)]
    R = HybridRetriever(texts, np.arange(30.0), methods=("bm25",))
    top = R.search(["story about topic1"], [10.5], k=10)[0]
    assert (top[top >= 0] <= 10).all()


# ---------------------------------------------------------------- independence
def test_copies_are_collapsed():
    texts = ["Wire - Maya Verhoeven visited Valmere. The story is developing.",
             "Herald - Maya Verhoeven visited Valmere. The story is developing.",           # near duplicate
             "Post - Totally different words here about markets."]
    R = HybridRetriever(texts, [1.0, 2.0, 3.0], methods=("bm25",))
    w, cl = cluster_weights(np.array([0, 1, 2]), R.wtf, np.array([1.0, 2.0, 3.0]), np.array([0, 1, 2]), ["Wire", "Herald", "Post"],
                            ["", "", ""], theta=0.6, copy_weight=0.1)
    assert w[0] == 1.0 and w[1] == pytest.approx(0.1) and w[2] == 1.0 and cl[0] == cl[1] != cl[2]


def test_explicit_syndication_tag_links_articles():
    texts = ["Wire - alpha story one.", "Herald - completely other words zebra."]
    R = HybridRetriever(texts, [1.0, 2.0], methods=("bm25",))
    w, _ = cluster_weights(np.array([0, 1]), R.wtf, np.array([1.0, 2.0]), np.array([0, 1]), ["Wire", "Herald"], ["", "Wire"])
    assert w[1] == pytest.approx(0.1)


# ---------------------------------------------------------------- fusion maths
def _tab(c, s, pS, pR, w=None):
    n = len(c)
    return dict(c=np.array(c), s=np.array(s), pS=np.array(pS, float), pR=np.array(pR, float), pN=np.zeros(n),
                w=np.ones(n) if w is None else np.array(w, float), t=np.zeros(n), rank=np.arange(n))


def test_fusion_by_hand():
    tab = _tab([0, 0, 1], [0, 1, 2], [1, 0, 1], [0, 1, 0])
    S, m = fuse(tab, np.array([0.9, 0.7, 0.8]), 2)
    assert S[0] == pytest.approx(logit(0.9) - logit(0.7)) and m[0] == pytest.approx(2.0)
    assert S[1] == pytest.approx(logit(0.8))


def test_low_credibility_sources_are_ignored_not_inverted():
    tab = _tab([0], [0], [1.0], [0.0])
    S, _ = fuse(tab, np.array([0.1]), 1)
    assert S[0] == pytest.approx(0.0)


def test_independence_weight_scales_contribution():
    full = _tab([0, 0], [0, 1], [1, 1], [0, 0], w=[1, 1]); cut = _tab([0, 0], [0, 1], [1, 1], [0, 0], w=[1, 0.1])
    r = np.array([0.9, 0.9])
    assert fuse(cut, r, 1)[0][0] == pytest.approx(1.1 * logit(0.9)) and fuse(full, r, 1)[0][0] == pytest.approx(2 * logit(0.9))


def test_decider_is_monotone_and_abstains():
    rng = np.random.default_rng(0)
    S = rng.normal(size=600) * 3; m = np.abs(rng.normal(size=600)) * 2
    y = np.where(m < 0.4, NEI, np.where(S + rng.normal(size=600) > 0, SUPPORTED, REFUTED))
    d = Decider().fit(S, m, y)
    assert d.a > 0 and np.all(np.diff(d.prob(np.sort(S))) >= 0)
    v, _ = d.verdict(np.array([5.0, -5.0, 5.0]), np.array([3.0, 3.0, 0.0]))
    assert list(v) == [SUPPORTED, REFUTED, NEI]


def test_decider_never_inverts_even_if_data_is_adversarial():
    S = np.linspace(-4, 4, 200); m = np.full(200, 3.0); y = np.where(S > 0, REFUTED, SUPPORTED)     # adversarial: anti-correlated
    d = Decider().fit(S, m, y)
    assert d.a >= 0


def test_burst_multiplier_only_hits_unproven_sources():
    tab = _tab([0, 0, 0, 0], [0, 1, 2, 3], [1, 1, 1, 1], [0, 0, 0, 0]); tab["t"] = np.array([0.0, 0.1, 0.2, 0.3])
    n_obs = np.array([50.0, 0.0, 0.0, 0.0])
    mult = burst_multiplier(tab, n_obs, window=0.5, n0=2.0, gamma=0.5)
    assert mult[0] == 1.0 and np.all(mult[1:] < 1.0)


# ---------------------------------------------------------------- source validation
def test_beta_update_moves_reliability():
    rng = np.random.default_rng(0)
    meta = rng.normal(size=(5, 7)); pm = fit_prior_model(rng.normal(size=(40, 7)), rng.uniform(.3, .95, 40))
    V = SourceValidator(meta, pm, kappa=6.0, lam_em=0.1, shrink_new=False)
    before = V.mean().copy(); V.ver_c[0] = 20; V.ver_n[0] = 20; V.ver_c[1] = 0; V.ver_n[1] = 20
    after = V.mean()
    assert after[0] > before[0] and after[1] < before[1] and after[2] == pytest.approx(before[2])
    assert V.quantile(0.2)[0] < after[0]


def test_loo_credit_gives_no_self_confirmation():
    tab = _tab([0], [0], [1.0], [0.0])
    c, n = credit_loo(tab, np.array([logit(.9)]), np.array([0.9]), 1.0, np.array([True]), 1)
    assert c[0] / n[0] == pytest.approx(0.5)         # a lone source is neither rewarded nor punished


def test_prior_model_is_informative(models):
    from vericloud.world import make_sources
    src = make_sources(np.random.default_rng(99), with_sybil=False)
    meta = np.array([s.meta for s in src]); r = np.array([s.reliability for s in src])
    from scipy.stats import spearmanr
    assert spearmanr(models[1].predict(meta), r)[0] > 0.4


# ---------------------------------------------------------------- end to end
def test_stance_model_accuracy(small):
    t = small["table"]; pred = np.stack([t["pS"], t["pR"], t["pN"]], 1).argmax(1)
    assert (pred == t["true_stance"]).mean() > 0.95


def test_vericloud_beats_naive_voting_under_attack(small):
    r = small["results"]
    assert r["VeriCloud"]["macro_f1"] > r["Majority vote"]["macro_f1"] + 0.05
    assert r["VeriCloud"]["macro_f1"] <= r["Oracle reliability + stance"]["macro_f1"] + 0.02


def test_probabilities_are_calibrated_enough(small):
    assert small["results"]["VeriCloud"]["ece"] < 0.15


def test_report_cites_only_validated_sources_and_flags(small):
    nc, tab, w = small["nc"], small["table"], small["world"]; claims = w.claims
    pred = nc.predict(tab, claims, mc=60)
    V = nc.validator; lo = V.quantile(0.2)
    seen_flag = False
    for i in np.flatnonzero(small["masks"][2])[:120]:
        rep = build_report(nc, tab, claims, int(i), pred)
        for e in rep["cited_sources"]:
            assert e["independent"] and e["reliability_80ci"][0] >= 0.62
        seen_flag |= bool(rep["flags"])
        assert rep["verdict"] in VERDICTS and 0 <= rep["probability_supported"] <= 1
        assert rep["probability_80ci"][0] <= rep["probability_supported"] <= rep["probability_80ci"][1]
    md = to_markdown(build_report(nc, tab, claims, int(np.flatnonzero(small["masks"][2])[0]), pred))
    assert "Verdict" in md


def test_verify_free_text(small):
    nc, claims = small["nc"], small["world"].claims
    nc.calibrate(small["table"], claims, small["masks"][1])
    c = claims[int(np.flatnonzero(small["masks"][2])[5])]
    rep = verify_text(nc, c.text, c.t, mc=40)
    assert rep["claim"] == c.text and rep["verdict"] in VERDICTS
    with pytest.raises(ValueError):
        verify_text(nc, "nothing recognisable here", 10.0)


def test_metrics():
    p = np.array([.9, .9, .1, .1]); y = np.array([1, 1, 0, 0]); assert ece(p, y) == pytest.approx(0.1)
    cov, risk, aurc = risk_coverage(np.array([.9, .8, .7, .6]), np.array([1, 1, 0, 0])); assert risk[1] == 0 and risk[-1] == 0.5
