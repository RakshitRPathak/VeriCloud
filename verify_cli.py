#!/usr/bin/env python3
"""Verify a claim against the simulated news corpus and print a source-validated report.

    python3 verify_cli.py --list                      # show example claims
    python3 verify_cli.py "Maya Verhoeven visited Valmere"   # any claim whose entities exist in the gazetteer
    python3 verify_cli.py --claim-id 1500 --attack    # a corpus claim, in a world where 30% of claims are attacked

The corpus is synthetic (see README); this CLI demonstrates the end-to-end system, not a real-world fact checker.
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np
from vericloud import generate_world, WorldConfig, VeriCloud, train_stance_model, VERDICTS
from vericloud.textproc import Annotator
from vericloud.pipeline import train_prior
from vericloud.benchmark import split_masks
from vericloud.report import verify_text, to_markdown

ap = argparse.ArgumentParser()
ap.add_argument("claim", nargs="?"); ap.add_argument("--list", action="store_true"); ap.add_argument("--claim-id", type=int)
ap.add_argument("--attack", action="store_true"); ap.add_argument("--seed", type=int, default=1)
a = ap.parse_args()
w = generate_world(WorldConfig(seed=a.seed, labeled_frac=0.2, campaign_share=0.3 if a.attack else 0.0, campaign_strength=10))
if a.list:
    for c in w.claims[-12:]:
        print(f"[{c.id}] {c.text}   (day {c.t:.0f}, truth: {VERDICTS[c.label]})")
    sys.exit(0)
print("building index and learning source credibility ...", file=sys.stderr)
nc = VeriCloud(w, stance=train_stance_model(Annotator(w.gaz)), prior_model=train_prior())
claims = w.claims; tab = nc.build_table(); masks = split_masks(claims)
nc.learn_sources(tab, claims, np.array([c.verified for c in claims]) & masks[0]); nc.calibrate(tab, claims, masks[1])
if a.claim_id is not None:
    c = claims[a.claim_id]; text, t = c.text, c.t
else:
    if not a.claim:
        ap.error("give a claim string, --claim-id, or --list")
    text, t = a.claim, float(max(x.t for x in w.articles))
rep = verify_text(nc, text, t, mc=200)
print(to_markdown(rep))
if a.claim_id is not None:
    print(f"_Ground truth (evaluation only): {VERDICTS[claims[a.claim_id].label]}_")
