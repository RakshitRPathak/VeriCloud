# Using VeriCloud in a paper (IEEE Access)

Read the README's scope note first. Everything below assumes you have reconciled this design with your real system, and that you will add a real-data evaluation. A synthetic-only evaluation is unlikely to survive peer review at a journal, so treat it as a controlled study that complements real data, not a replacement.

## Experiment to section map

| Paper section | Experiment | File(s) | What it shows |
|---|---|---|---|
| System overview | – | `results/figures/architecture.pdf` | seven-stage pipeline with validator and defences |
| Main results | E1 | `tables/e1_clean.tex`, `e1_under_attack.tex`, `e1_attack.tex`, `figures/e1_main_benchmark.pdf` | tie when clean, large gain under attack |
| Robustness | E2 | `tables/e2_attack_breadth.tex`, `figures/e2_adversarial.pdf` | stable macro-F1 as the targeted share grows to 60% |
| Adaptive adversary / threat model | E2b | `tables/e2b_adaptive.tex`, `figures/e2b_adaptive.pdf` | aged domains and slow drip cost about 4 points |
| Source validation | E3 | `tables/e3_source_learning.md`, `figures/e3_source_learning.pdf` | reliability estimates improve with the number of human fact-checks |
| Ablation | E4 | `tables/e4_components.tex`, `e4_ablation_paired.json` | which components matter (the three defences and source learning) and which do not (independence clustering, EM) |
| Calibration and selective prediction | E5 | `e5_calibration_selective.json`, `figures/e5_calibration_selective.pdf` | ECE, risk-coverage |
| Qualitative | E5 | `results/sample_reports/` | five example reports, including a wrong verdict |
| Efficiency | E6 | `tables/e6_efficiency.md` | about 20 ms per claim (timings vary by machine and run) up to about 16k articles |

## Claim wording that the evidence supports

Safe, if you keep the "simulated" qualifier:
* "In a controlled simulation with known ground truth, VeriCloud retained macro-F1 of 0.79 when 30% of claims were targeted by coordinated sybil campaigns, versus 0.70 for the best baseline (Wilcoxon p = 0.008, 8 seeds)."
* "Removing the three campaign defences reduced macro-F1 by 0.12."
* "Selective prediction (abstention) reduced the area under the risk-coverage curve to 0.056."

Not supported, so avoid:
* "VeriCloud outperforms existing fact-checking systems." Nothing here compares to existing systems.
* "VeriCloud is robust to adversaries." E2b shows degradation against patient attackers.
* "VeriCloud improves accuracy." In the clean world it ties simple baselines.
* Any statement about real-world accuracy, real-source reliability or scalability beyond 16k articles.
* "Independence clustering improves performance" or "EM improves performance". Both were neutral here.

## Threats to validity (put these in the paper)

1. **Simulator realism.** Template-generated text, a closed entity lexicon and single-fact claims. The stance model is near-perfect in-world.
2. **Co-design.** The simulator and defences were developed together; attacks outside the simulator's families are untested.
3. **Ground-truth reliability.** Source reliability is a simulator parameter; in reality it is contested and time-varying.
4. **Seeds.** 4 to 8 seeds per experiment; p-values are coarse.
5. **Baselines.** Four simple baselines share VeriCloud's retrieval and stance outputs. There is no comparison with published verification systems.

## What to add before submission

* **Real-data claim verification:** run the pipeline with a real stance model (NLI) and dense retrieval on a public benchmark (candidates: FEVER for evidence-based verdicts, LIAR or MultiFC for claims with fact-checker labels, NELA-GT for source-level reliability labels). I have not run any of these; check licences and label schemes first.
* **A real attack study:** inject paraphrased coordinated articles into a real corpus and measure the verdict flips.
* **Stronger baselines:** published claim-verification models and credibility-aware ranking methods.
* **Human evaluation of the reports:** do the "not relied upon" explanations help fact-checkers? Even a small study (10 to 20 claims, 3 annotators) would add substance.
* **Reproducibility statement:** point to `run_all.sh`, the fixed seeds and the figures in `results/`.
