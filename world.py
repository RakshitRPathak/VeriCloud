"""Simulated news ecosystem with a known ground truth.

Entities and facts (subject, relation, object) are reported by sources of hidden reliability.  Errors are
*correlated*: a wrong source reports the fact's popular "decoy" most of the time.  The world contains wire syndication,
content-farm copy chains, fact-checkers, typos, hard distractors and (optionally) coordinated sybil campaigns.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

SUPPORTED, REFUTED, NEI = 0, 1, 2
VERDICTS = ["SUPPORTED", "REFUTED", "NOT ENOUGH INFO"]

FIRST = ["Maya", "Arjun", "Elena", "Tomas", "Priya", "Lucas", "Amara", "Kenji", "Sofia", "Omar", "Ingrid", "Rafael",
         "Neha", "Viktor", "Leila", "Mateo", "Hana", "Daniel", "Zara", "Pavel", "Isabel", "Rohan", "Freya", "Samir",
         "Chloe", "Dmitri", "Anika", "Hugo", "Nadia", "Felix", "Mei", "Andre", "Ruth", "Idris", "Clara", "Jonas",
         "Tara", "Oskar", "Lina", "Rahul"]
LAST = ["Verhoeven", "Okafor", "Lindqvist", "Banerjee", "Castellano", "Novak", "Haddad", "Tanaka", "Moreau", "Petrov",
        "Adeyemi", "Kowalski", "Sandoval", "Iyer", "Brandt", "Farouk", "Nakamura", "Duarte", "Eriksen", "Mbeki",
        "Rinaldi", "Volkov", "Chaudhry", "Albrecht", "Quintero", "Yamada", "Osei", "Halvorsen", "Marchetti", "Bhatt",
        "Zielinski", "Ferreira", "Aziz", "Lombardi", "Sorensen", "Delacroix", "Mukherjee", "Ivanova", "Santos", "Wagner"]
CO_A = ["Northwind", "Bluepeak", "Ironvale", "Solstice", "Quantix", "Harborline", "Redwood", "Lumina", "Cobalt", "Veridian",
        "Stratos", "Maplecrest", "Orbis", "Tidewater", "Granite", "Helix", "Zenith", "Brightfield", "Kestrel", "Evergreen",
        "Falcon", "Meridian", "Obsidian", "Pinnacle", "Sapphire"]
CO_B = ["Dynamics", "Holdings", "Systems", "Labs", "Industries", "Partners", "Motors", "Networks", "Energy", "Foods",
        "Logistics", "Pharma"]
AW_A = ["Golden", "Silver", "Crimson", "Azure", "Emerald", "Ivory", "Amber", "Onyx", "Scarlet", "Jade", "Copper", "Platinum"]
AW_B = ["Lantern", "Meridian", "Compass", "Beacon", "Anchor", "Quill", "Summit", "Horizon"]
ROLE_A = ["Director", "Head", "Chief Advisor", "Secretary", "Chair", "Commissioner", "Ambassador", "Governor", "Minister", "Auditor"]
ROLE_B = ["Finance", "Energy", "Transport", "Health", "Trade", "Defence", "Education", "Culture"]
SYL = ["val", "mere", "dor", "ston", "bray", "kel", "mont", "fair", "wick", "ash", "ford", "lin", "gar", "nes", "tor",
       "bel", "rin", "cas", "dun", "hol"]
PROD_A = ["Aurora", "Nimbus", "Vertex", "Pulse", "Atlas", "Echo", "Zephyr", "Orion", "Helio", "Nova"]

REL_NAMES = ["won", "acquired", "appointed", "visited", "launched", "resigned"]
REL_TYPES = [("person", "award"), ("company", "company"), ("person", "role"), ("person", "city"),
             ("company", "product"), ("person", "company")]
# assertion templates (article style) and claim templates (the last entry of each claim list uses a synonym
# that is NOT in the article cue lexicon -> imperfect relation matching)
ART_T = [
    ["{S} has won the {O}", "{S} clinched the {O}", "The {O} was awarded to {S}", "{S} took home the {O}"],
    ["{S} has acquired {O}", "{S} completed the purchase of {O}", "{O} was bought by {S}", "{S} snapped up {O}"],
    ["{S} was appointed {O}", "{S} has been named {O}", "{S} takes over as {O}", "{S} now serves as {O}"],
    ["{S} visited {O}", "{S} arrived in {O}", "{S} is travelling to {O}", "{S} held talks in {O}"],
    ["{S} launched {O}", "{S} unveiled {O}", "{S} introduced {O} to the market", "{S} rolled out {O}"],
    ["{S} resigned from {O}", "{S} has stepped down from {O}", "{S} quit {O}", "{S} left {O}"],
]
CLAIM_T = [
    ["{S} won the {O}", "{S} has been awarded the {O}", "{S} took home the {O}", "{S} bagged the {O}"],
    ["{S} acquired {O}", "{S} bought {O}", "{S} has completed the purchase of {O}", "{S} took over {O}"],
    ["{S} was appointed {O}", "{S} has been named {O}", "{S} now serves as {O}", "{S} got the post of {O}"],
    ["{S} visited {O}", "{S} arrived in {O}", "{S} held talks in {O}", "{S} flew into {O}"],
    ["{S} launched {O}", "{S} unveiled {O}", "{S} rolled out {O}", "{S} released {O}"],
    ["{S} resigned from {O}", "{S} stepped down from {O}", "{S} quit {O}", "{S} walked out of {O}"],
]
HEDGE = ["Reportedly, ", "Sources claim that ", "Unconfirmed reports suggest that "]
DENY = ["Fact check: it is false that ", "Officials denied that ", "There is no evidence that ", "Debunked: the claim that "]
FILLER = ["The development comes amid growing public attention.", "Analysts say the news could shape the coming months.",
          "Further details are expected later this week.", "Observers described the move as significant.",
          "The announcement was widely discussed online.", "Local reaction has been mixed.",
          "Officials declined to comment further.", "The story is developing and will be updated.",
          "Market watchers are following the situation closely.", "Several experts weighed in on the implications.",
          "The news was first noticed by readers over the weekend.", "Coverage of the topic has increased sharply."]
SRC_PREFIX = ["The Daily", "Morning", "National", "City", "Global", "Evening", "Capital", "Frontier", "Metro", "Open"]
SRC_SUFFIX = ["Herald", "Chronicle", "Courier", "Ledger", "Dispatch", "Standard", "Observer", "Tribune", "Post", "Gazette"]

TYPE_SPECS = {   # n, Beta(a,b) reliability, article volume weight
    "wire": (3, (60, 2), 3.0), "reputable": (25, (30, 2), 2.0), "regional": (40, (14, 3), 1.5),
    "tabloid": (30, (8, 5), 1.5), "farm": (30, (4, 6), 1.0), "factchecker": (5, (60, 2), 0.0)}
N_SYBIL = 800


@dataclass
class WorldConfig:
    n_facts: int = 2400
    uncovered_frac: float = 0.20
    p_true_claim: float = 0.5
    days: float = 180.0
    decoy_prob: float = 0.8              # a wrong report uses the fact's decoy with this probability
    p_wire_copies: float = 0.7           # fact covered by wire gets syndicated
    farm_copy_rate: float = 1.0          # mean # of content-farm copies per covered fact (scaled by popularity)
    p_tag_wire: float = 0.6              # syndicated copies carrying an explicit "via <origin>" tag
    p_tag_farm: float = 0.2
    factcheck_prob: float = 0.45         # fact-check of a fact that has circulating false reports
    campaign_share: float = 0.0          # fraction of false claims targeted by a sybil campaign
    campaign_strength: int = 10          # sybil articles per targeted claim
    sybil_new_prob: float = 0.7          # P(a sybil domain is freshly registered); an adaptive attacker uses aged domains (0.0)
    campaign_spread: float = 0.8         # days over which a campaign's articles are spread (adaptive attacker: slow drip)
    campaign_paraphrase: float = 0.5     # P(a sybil article is freshly re-written rather than a near-copy of the seed)
    typo_prob: float = 0.02
    labeled_frac: float = 0.15           # fraction of claims that human fact-checkers verified
    seed: int = 0


@dataclass
class Source:
    id: int
    name: str
    type: str
    reliability: float
    volume: float
    meta: np.ndarray
    is_sybil: bool = False


@dataclass
class Article:
    id: int
    source: int
    t: float
    text: str
    fact: int                      # -1 = distractor
    statements: list               # [(subject, rel, obj, polarity)] ground truth structure (polarity +1 assert / -1 deny)
    copy_of: int = -1
    via_tag: str = ""              # explicit syndication tag (origin outlet name), "" if none
    campaign: bool = False


@dataclass
class Claim:
    id: int
    text: str
    subject: str
    rel: int
    obj: str
    label: int
    fact: int
    t: float
    verified: bool                 # a human fact-check label exists for it
    targeted: bool = False


@dataclass
class World:
    cfg: WorldConfig
    sources: list
    articles: list
    claims: list
    gaz: dict                      # lowercase phrase -> (type, canonical)
    pools: dict                    # type -> list of names
    facts: list

    def source_reliability(self):
        return np.array([s.reliability for s in self.sources])


def _pools(_unused=None):
    rng = np.random.default_rng(12345)          # fixed lexicon shared by all worlds (a real gazetteer is fixed too)
    persons = [f"{a} {b}" for a in FIRST for b in LAST]
    rng.shuffle(persons)
    comps = [f"{a} {b}" for a in CO_A for b in CO_B]
    awards = [f"{a} {b} Award" for a in AW_A for b in AW_B]
    roles = [f"{a} of {b}" for a in ROLE_A for b in ROLE_B]
    cities = sorted({(rng.choice(SYL) + rng.choice(SYL)).capitalize() for _ in range(400)})[:70]
    prods = [f"{a} X{rng.integers(2, 99)}" for a in PROD_A for _ in range(7)]
    prods = list(dict.fromkeys(prods))[:60]
    return {"person": persons[:700], "company": comps[:300], "award": awards[:80], "role": roles[:80],
            "city": cities, "product": prods}


def _meta(rng, r, typ, sybil, new_prob=0.7):
    rr = 0.85 if sybil else r
    z = (rr - 0.7) * 3.0
    sig = lambda x: 1 / (1 + np.exp(-x))
    age = max(0.1, 2.0 + 0.8 * z + rng.normal(0, 1.6))
    if sybil and rng.random() < new_prob:
        age = rng.uniform(0.05, 0.6)                          # freshly registered domain (red flag)
    return np.array([np.log1p(age), float(rng.random() < sig(1.0 * z)), np.clip(.5 + .12 * z + rng.normal(0, .25), 0, 1),
                     np.clip(.5 - .10 * z + rng.normal(0, .25), 0, 1), float(rng.random() < sig(.5 + .5 * z)),
                     float(rng.random() < sig(.8 * z)), np.clip(.4 - .12 * z + rng.normal(0, .25), 0, 1)])


def make_sources(rng, with_sybil=True, sybil_new_prob=0.7):
    src, names = [], set()
    def name():
        while True:
            n = f"{rng.choice(SRC_PREFIX)} {rng.choice(SRC_SUFFIX)} {rng.integers(1, 99)}"
            if n not in names:
                names.add(n); return n
    for typ, (n, (a, b), vol) in TYPE_SPECS.items():
        for _ in range(n):
            r = float(rng.beta(a, b))
            src.append(Source(len(src), name(), typ, r, vol * float(rng.lognormal(0, .3)), _meta(rng, r, typ, False)))
    if with_sybil:
        for _ in range(N_SYBIL):
            src.append(Source(len(src), name(), "sybil", 0.15, 0.0, _meta(rng, 0.85, "sybil", True, sybil_new_prob), True))
    return src


def _typo(rng, text, ents):
    cand = [e for e in ents if e in text]
    if not cand:
        return text
    e = cand[rng.integers(len(cand))]
    if len(e) < 5:
        return text
    i = int(rng.integers(1, len(e) - 2))
    bad = e[:i] + e[i + 1] + e[i] + e[i + 2:]
    return text.replace(e, bad, 1)


def _assert_sentence(rng, S, rel, O):
    return ART_T[rel][int(rng.integers(4))].format(S=S, O=O)


def _article_text(rng, src_name, statements, hedge=False, filler=2):
    parts = []
    for (S, rel, O, pol) in statements:
        s = _assert_sentence(rng, S, rel, O)
        if pol < 0:
            s = DENY[int(rng.integers(len(DENY)))] + s
        elif hedge:
            s = HEDGE[int(rng.integers(len(HEDGE)))] + s
        parts.append(s + ".")
    body = " ".join(parts)
    fill = " ".join(rng.choice(FILLER, size=filler, replace=False))
    return f"{src_name} - {body} {fill}"


def _perturb(rng, text, frac, protect):
    """paraphrase by swapping filler/function words and shuffling filler sentence order; entity strings are protected"""
    words = text.split(" ")
    swaps = {"has": "had", "the": "a", "was": "got", "of": "of", "and": "plus", "to": "for", "in": "inside", "that": "that"}
    out = []
    for w in words:
        if rng.random() < frac and w.lower() in swaps and not any(w in p for p in protect):
            out.append(swaps[w.lower()])
        elif rng.random() < frac * 0.25 and w.endswith(".") is False and not any(w in p for p in protect) and len(w) > 4 and w.islower():
            continue
        else:
            out.append(w)
    return " ".join(out)


def generate_world(cfg: WorldConfig) -> World:
    rng = np.random.default_rng(cfg.seed)
    pools = _pools(rng)
    gaz = {}
    for typ, lst in pools.items():
        for n in lst:
            gaz[n.lower()] = (typ, n)
    sources = make_sources(rng, True, cfg.sybil_new_prob)
    by_type = {}
    for s in sources:
        by_type.setdefault(s.type, []).append(s)
    pool_src = [s for s in sources if s.volume > 0]
    pw = np.array([s.volume for s in pool_src]); pw /= pw.sum()
    arts: list[Article] = []

    def add_article(src, t, text, fact, stm, copy_of=-1, via="", campaign=False):
        a = Article(len(arts), src.id, float(t), text, fact, stm, copy_of, via, campaign)
        arts.append(a); return a

    if cfg.n_facts > 3000:    # unique (subject, relation) keys: 700 persons x 4 + 300 companies x 2 relations => ~3.4k capacity
        raise ValueError('n_facts > 3000 exceeds the key capacity of the fixed lexicon (would loop forever)')
    facts, keys = [], set()
    rel_p = np.array([.20, .12, .20, .20, .12, .16])
    for f in range(cfg.n_facts):
        while True:                                   # every (subject, relation) pair identifies at most one fact
            rel = int(rng.choice(6, p=rel_p)); st, ot = REL_TYPES[rel]
            S = pools[st][int(rng.integers(len(pools[st])))]
            if (S, rel) not in keys:
                keys.add((S, rel)); break
        objs = pools[ot]
        O = objs[int(rng.integers(len(objs)))]
        while ot == st and O == S:
            O = objs[int(rng.integers(len(objs)))]
        decoy = O
        while decoy == O or (ot == st and decoy == S):
            decoy = objs[int(rng.integers(len(objs)))]
        facts.append(dict(id=f, S=S, rel=rel, O=O, decoy=decoy, t0=float(rng.uniform(0, cfg.days)),
                          pop=float(rng.lognormal(0, .5)), covered=bool(rng.random() > cfg.uncovered_frac)))

    def other_obj(fct):
        objs = pools[REL_TYPES[fct["rel"]][1]]
        while True:
            o = objs[int(rng.integers(len(objs)))]
            if o not in (fct["O"], fct["decoy"]) and o != fct["S"]:
                return o

    fact_arts = {}
    for fct in facts:
        if not fct["covered"]:
            continue
        S, rel, O = fct["S"], fct["rel"], fct["O"]
        n_orig = 1 + rng.poisson(2.2 * fct["pop"])
        srcs = rng.choice(len(pool_src), size=n_orig, p=pw)
        mine = []
        for si in srcs:
            s = pool_src[si]
            if s.type == "farm":
                continue                                    # farms only copy (below)
            ok = rng.random() < s.reliability
            obj = O if ok else (fct["decoy"] if rng.random() < cfg.decoy_prob else other_obj(fct))
            stm = [(S, rel, obj, +1)]
            text = _article_text(rng, s.name, stm, hedge=rng.random() < (0.1 if ok else 0.3))
            if rng.random() < cfg.typo_prob:
                text = _typo(rng, text, [S, obj])
            a = add_article(s, fct["t0"] + rng.exponential(1.5), text, fct["id"], stm)
            mine.append(a)
        # wire syndication
        wires = [a for a in mine if sources[a.source].type == "wire"]
        if wires and rng.random() < cfg.p_wire_copies:
            w = wires[0]
            for _ in range(int(rng.integers(2, 9))):
                cs = [s for s in pool_src if s.type in ("regional", "tabloid")]
                s = cs[int(rng.integers(len(cs)))]
                txt = w.text.replace(sources[w.source].name, s.name, 1)
                tag = sources[w.source].name if rng.random() < cfg.p_tag_wire else ""
                mine.append(add_article(s, w.t + rng.exponential(.7), txt, fct["id"], list(w.statements), w.id, tag))
        # content-farm copies of any existing article of this fact
        for _ in range(rng.poisson(cfg.farm_copy_rate * fct["pop"])):
            if not mine:
                break
            par = mine[int(rng.integers(len(mine)))]
            s = by_type["farm"][int(rng.integers(len(by_type["farm"])))]
            ents = [par.statements[0][0], par.statements[0][2]]
            txt = _perturb(rng, par.text.replace(sources[par.source].name, s.name, 1), 0.3, ents)
            if rng.random() < cfg.typo_prob * 2:
                txt = _typo(rng, txt, ents)
            tag = sources[par.source].name if rng.random() < cfg.p_tag_farm else ""
            mine.append(add_article(s, par.t + rng.exponential(.8), txt, fct["id"], list(par.statements), par.id, tag))
        # fact-checkers
        circ = any(a.statements[0][2] != O for a in mine)
        if rng.random() < (cfg.factcheck_prob if circ else 0.08):
            s = by_type["factchecker"][int(rng.integers(5))]
            wrong = rng.random() > s.reliability
            if wrong:
                stm = [(S, rel, O, -1), (S, rel, fct["decoy"], +1)]
            else:
                stm = [(S, rel, fct["decoy"], -1), (S, rel, O, +1)]
            mine.append(add_article(s, fct["t0"] + 1 + rng.exponential(2.0), _article_text(rng, s.name, stm), fct["id"], stm))
        fact_arts[fct["id"]] = mine

    # hard distractors: same subject other relation / same relation other subject / unrelated
    n_dis = int(0.3 * len(arts))
    for _ in range(n_dis):
        s = pool_src[int(rng.choice(len(pool_src), p=pw))]
        fct = facts[int(rng.integers(len(facts)))]
        rel2 = fct["rel"] if rng.random() < .5 else int(rng.integers(6))
        st, ot = REL_TYPES[rel2]
        S2 = fct["S"] if (rng.random() < .5 and REL_TYPES[fct["rel"]][0] == st) else pools[st][int(rng.integers(len(pools[st])))]
        O2 = pools[ot][int(rng.integers(len(pools[ot])))]
        if (S2, rel2) in keys or S2 == O2:           # distractors never talk about any real fact's (subject, relation)
            continue
        stm = [(S2, rel2, O2, +1)]
        add_article(s, rng.uniform(0, cfg.days + 20), _article_text(rng, s.name, stm), -1, stm)

    # claims
    claims = []
    for fct in facts:
        S, rel = fct["S"], fct["rel"]
        if fct["covered"]:
            true_claim = rng.random() < cfg.p_true_claim
            obj = fct["O"] if true_claim else (fct["decoy"] if rng.random() < .8 else other_obj(fct))
            label = SUPPORTED if true_claim else REFUTED
        else:
            obj = fct["O"] if rng.random() < .5 else fct["decoy"]
            label = NEI
        k = 3 if rng.random() < 0.08 else int(rng.integers(3))          # 8%: synonym unknown to the claim parser
        text = CLAIM_T[rel][k].format(S=S, O=obj)
        t = fct["t0"] + 2.0 + rng.exponential(4.0)
        claims.append(Claim(len(claims), text, S, rel, obj, label, fct["id"], float(t), bool(rng.random() < cfg.labeled_frac)))

    # sybil campaigns on a share of false claims
    sybils = [s for s in sources if s.is_sybil]
    if cfg.campaign_share > 0 and cfg.campaign_strength > 0:
        cov_c = [c for c in claims if c.label != NEI]            # boost campaigns on false claims, smear campaigns on true ones
        tgt = rng.choice(len(cov_c), size=int(cfg.campaign_share * len(cov_c)), replace=False)
        for ti in tgt:
            c = cov_c[ti]; c.targeted = True
            fct = facts[c.fact]
            pushed = c.obj if c.label == REFUTED else fct["decoy"]          # both push the false narrative
            stm = [(c.subject, c.rel, pushed, +1)]
            pick = rng.choice(len(sybils), size=min(cfg.campaign_strength + 1, len(sybils)), replace=False)   # fresh accounts
            seed_s = sybils[int(pick[0])]
            seed_text = _article_text(rng, seed_s.name, stm, hedge=False)
            t_start = max(0.0, c.t - rng.uniform(0.5, 2.5))
            ents = [c.subject, pushed]
            for j in range(cfg.campaign_strength):
                s = sybils[int(pick[1 + j % (len(pick) - 1)])]
                if rng.random() < cfg.campaign_paraphrase:        # freshly re-written: evades near-duplicate clustering
                    txt = _article_text(rng, s.name, stm, hedge=rng.random() < 0.2)
                else:                                              # lightly perturbed copy of the seed script
                    txt = _perturb(rng, seed_text.replace(seed_s.name, s.name, 1), 0.15, ents)
                add_article(s, t_start + rng.uniform(0, cfg.campaign_spread), txt, c.fact, stm, copy_of=-1, campaign=True)

    arts.sort(key=lambda a: a.t)
    idmap = {a.id: i for i, a in enumerate(arts)}
    for i, a in enumerate(arts):
        a.copy_of = idmap[a.copy_of] if a.copy_of >= 0 else -1
        a.id = i
    claims.sort(key=lambda c: c.t)
    for i, c in enumerate(claims):
        c.id = i
    return World(cfg, sources, arts, claims, gaz, pools, facts)


def true_stance(article: Article, claim: Claim) -> int:
    """ground-truth stance of an article toward a claim: 0 support, 1 refute, 2 neutral"""
    rel_stm = [s for s in article.statements if s[0] == claim.subject and s[1] == claim.rel]
    if not rel_stm:
        return 2
    for (_, _, o, pol) in rel_stm:
        if o == claim.obj:
            return 0 if pol > 0 else 1
    if any(pol > 0 for (_, _, _, pol) in rel_stm):
        return 1
    return 2
