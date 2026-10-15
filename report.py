"""Source-validated reporting: a verdict is only ever *explained* with sources that passed validation.

* cited ("validated") sources: lower credible bound of reliability >= cite_threshold
* every other stance-bearing source is listed as "not relied upon" with the reason (unproven / low credibility / copy)
* flags: coordinated burst, single independent source, high flip probability -> "needs human review"
"""
from __future__ import annotations
import json
import numpy as np
from .world import VERDICTS, SUPPORTED, REFUTED, NEI, Claim
from .textproc import parse_claim


def build_report(nc, table, claims, i, pred, cite_threshold=0.62, max_cited=5):
    """pred: dict from nc.predict(..., mc>0). i = claim index inside `claims`."""
    c = claims[i]
    V = nc.validator
    mean = V.mean(); lo = V.quantile(0.2); hi = V.quantile(0.8); nobs = V.n_obs()
    rows = np.flatnonzero((table["c"] == i) & ((table["pS"] + table["pR"]) > 0.5))
    ev = []
    for r in rows:
        s = int(table["s"][r]); art = nc.w.articles[int(table["a"][r])]
        stance = "supports" if table["pS"][r] > table["pR"][r] else "refutes"
        copy = table["w"][r] < 0.999
        validated = lo[s] >= cite_threshold
        reason = None
        if not validated:
            reason = "no track record yet" if nobs[s] < 2.0 else "credibility below citation threshold"
        if copy:
            reason = (reason + "; " if reason else "") + "near-duplicate / syndicated copy (counted at reduced weight)"
        ev.append(dict(source=nc.src_names[s], article_id=int(art.id), published_day=round(float(art.t), 1), stance=stance,
                       stance_confidence=round(float(max(table["pS"][r], table["pR"][r])), 3),
                       reliability=round(float(mean[s]), 3), reliability_80ci=[round(float(lo[s]), 3), round(float(hi[s]), 3)],
                       independent=bool(not copy), validated=bool(validated and not copy), reason_not_relied_on=reason,
                       excerpt=art.text.split(" - ", 1)[-1][:160]))
    cited = sorted([e for e in ev if e["validated"]], key=lambda e: -e["reliability_80ci"][0])[:max_cited]
    other = [e for e in ev if not e["validated"]]
    n_indep_valid = len({e["source"] for e in ev if e["validated"]})
    flags = []
    # coordinated burst among unproven sources
    unproven = [e for e in ev if e["reliability_80ci"][0] < cite_threshold and (nobs[nc.src_names.index(e["source"])] < 2.0)]
    for st in ("supports", "refutes"):
        grp = [e for e in unproven if e["stance"] == st]
        if len(grp) >= 3:
            days = sorted(e["published_day"] for e in grp)
            if days[-1] - days[0] <= 1.0:
                flags.append(f"possible coordinated burst: {len(grp)} '{st}' articles from unproven sources within {days[-1]-days[0]:.1f} day(s)")
    if n_indep_valid <= 1 and pred["pred"][i] != NEI:
        flags.append("verdict rests on at most one independent validated source")
    flip = float(pred.get("flip", np.zeros(len(claims)))[i])
    if flip > 0.25:
        flags.append(f"verdict unstable under source-reliability uncertainty (flip probability {flip:.0%})")
    verdict = VERDICTS[int(pred["pred"][i])]
    review = bool(bool(flags) or (pred["pred"][i] != NEI and pred["P_lo"][i] < 0.5 < pred["P_hi"][i]))
    return dict(claim=c.text, parsed=dict(zip(("subject", "relation", "object"), (c.subject, c.rel, c.obj))),
                verdict=verdict, probability_supported=round(float(pred["P"][i]), 3),
                probability_80ci=[round(float(pred["P_lo"][i]), 3), round(float(pred["P_hi"][i]), 3)],
                evidence_mass=round(float(pred["m"][i]), 2), flip_probability=round(flip, 3),
                needs_human_review=review, flags=flags, cited_sources=cited, not_relied_upon=other[:8],
                n_not_relied_upon=len(other))


def to_markdown(rep):
    L = [f"### Claim: “{rep['claim']}”", "",
         f"**Verdict: {rep['verdict']}**  ·  P(supported) = {rep['probability_supported']} "
         f"(80% interval {rep['probability_80ci'][0]}–{rep['probability_80ci'][1]})  ·  evidence mass {rep['evidence_mass']}  ·  "
         f"flip probability {rep['flip_probability']:.0%}", ""]
    L.append("**Needs human review:** " + ("YES" if rep["needs_human_review"] else "no"))
    for f in rep["flags"]:
        L.append(f"- ⚠ {f}")
    L += ["", "**Validated sources relied upon**", ""]
    if not rep["cited_sources"]:
        L.append("_none — no validated source covers this claim_")
    for e in rep["cited_sources"]:
        L.append(f"- **{e['source']}** ({e['stance']}, reliability {e['reliability']} "
                 f"[{e['reliability_80ci'][0]}–{e['reliability_80ci'][1]}], day {e['published_day']}): “{e['excerpt']}”")
    if rep["n_not_relied_upon"]:
        L += ["", f"**Not relied upon ({rep['n_not_relied_upon']})** — listed for transparency", ""]
        for e in rep["not_relied_upon"]:
            L.append(f"- {e['source']} ({e['stance']}) — {e['reason_not_relied_on']}")
    return "\n".join(L) + "\n"


def verify_text(nc, text, t_now, mc=200):
    """Verify an arbitrary claim string against the indexed corpus (requires a fitted/calibrated VeriCloud)."""
    pc = parse_claim(text, nc.ann)
    if pc.subject is None:
        raise ValueError("could not parse a (subject, relation, object) claim from the text")
    claim = Claim(0, text, pc.subject, pc.rel, pc.obj, NEI, -1, float(t_now), False)
    table = nc.build_table([claim])
    pred = nc.predict(table, [claim], mc=mc)
    return build_report(nc, table, [claim], 0, pred)
