#!/usr/bin/env python3
"""One-command demo: the same targeted claims judged by simple baselines and by VeriCloud, with and without a sybil campaign.

    python3 demo.py            # ~1 minute on 2 cores

Prints (a) test-set macro-F1 and accuracy on *targeted* claims in a clean world vs an attacked world, and
(b) a few concrete claims where the attack flips a baseline but VeriCloud holds, with the report it would publish.
The corpus is synthetic (see README); the demo illustrates behaviour, not real-world accuracy.
"""
import sys, pathlib, warnings
sys.path.insert(0, str(pathlib.Path(__file__).parent)); warnings.filterwarnings("ignore")
import numpy as np
from vericloud import generate_world, WorldConfig, VeriCloud, train_stance_model, VERDICTS
from vericloud.textproc import Annotator
from vericloud.pipeline import train_prior
from vericloud.benchmark import split_masks, run_methods
from vericloud.report import verify_text, to_markdown

stance, prior = train_stance_model(Annotator(generate_world(WorldConfig(n_facts=900, seed=0)).gaz), n_facts=900), train_prior()
KEEP = ["Majority vote", "Top-1 evidence", "Credibility prior (no learning)", "VeriCloud"]
cached = {}
for tag, share in (("clean", 0.0), ("attacked", 0.3)):
    w = generate_world(WorldConfig(seed=11, labeled_frac=0.2, campaign_share=share, campaign_strength=10))
    nc = VeriCloud(w, stance=stance, prior_model=prior); tab = nc.build_table(); masks = split_masks(w.claims)
    res, extra, y = run_methods(nc, tab, w.claims, masks, include_oracles=False)
    cached[tag] = (w, nc, res, extra, y, masks)
    print(f"\n== {tag} world: {len(w.claims)} claims, {sum(c.targeted for c in w.claims)} targeted by sybil campaigns")
    for m in KEEP:
        print(f"  {m:34s} macro-F1 {res[m]['macro_f1']:.3f}   accuracy {res[m]['acc']:.3f}")

w, nc, res, extra, y, masks = cached["attacked"]
test = np.flatnonzero(masks[2]); tg = [i for i in test if w.claims[i].targeted]
mv, vc = extra["Majority vote"]["pred"], extra["VeriCloud"]["pred"]
flips = [i for i in tg if mv[i] != y[i] and vc[i] == y[i]]
print(f"\nOn the {len(tg)} targeted test claims: majority vote is right {np.mean([mv[i]==y[i] for i in tg]):.0%}, "
      f"VeriCloud {np.mean([vc[i]==y[i] for i in tg]):.0%}.  {len(flips)} claims are fooled for majority vote but held by VeriCloud.")
claims = w.claims
for i in flips[:2]:
    print(f"\n----- claim {i}: majority vote says {VERDICTS[mv[i]]}; truth is {VERDICTS[y[i]]} -----")
    print(to_markdown(verify_text(nc, claims[i].text, claims[i].t, mc=100)))
