"""Architecture diagram of VeriCloud (results/figures/architecture.png / .pdf)."""
from common import *
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, RED, GREY, GOLD = "#4f86c6", "#c8102e", "#6b7280", "#d98e04"


def box(ax, x, y, w, h, title, sub="", color=BLUE, fc="#ffffff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", lw=1.6, ec=color, fc=fc))
    ax.text(x + w / 2, y + h * (0.64 if sub else 0.5), title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)
    if sub:
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=7.4, color="#374151", linespacing=1.25)


def arrow(ax, a, b, color=GREY, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=11, lw=1.4, color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls))


if __name__ == "__main__":
    style(); fig, ax = plt.subplots(figsize=(13, 5.7)); ax.set_xlim(0, 13); ax.set_ylim(0, 5.7); ax.axis("off")
    # main row
    y, h, w = 2.9, 1.35, 1.9
    xs = [0.2, 2.45, 4.7, 6.95, 9.2]
    box(ax, xs[0], y, w, h, "1  Claim parsing", "gazetteer NER\n+ relation / negation cues")
    box(ax, xs[1], y, w, h, "2  Hybrid retrieval", "BM25 + char n-gram TF-IDF\nweighted RRF, time-causal")
    box(ax, xs[2], y, w, h, "3  Stance model", "pSupport / pRefute\nper (claim, article)")
    box(ax, xs[3], y, w, h, "4  Independence", "near-duplicate + syndication\nclusters: copies weigh 0.1")
    box(ax, xs[4], y, w + 0.4, h, "5  Evidence fusion", "S = Σ w·(pS−pR)·logit⁺(r)\nlow-credibility is ignored,\nnever inverted", color=RED)
    for i in range(4):
        arrow(ax, (xs[i] + w + 0.02, y + h / 2), (xs[i + 1] - 0.02, y + h / 2))
    # news corpus
    box(ax, 2.45, 4.5, w, 0.55, "News corpus", "", color=GREY, fc="#f3f4f6"); arrow(ax, (3.4, 4.48), (3.4, y + h + 0.02))
    # source validator
    box(ax, 4.7, 0.35, 4.1, 1.55, "Source validator", "Beta posterior per source\nprior: ridge on metadata, shrunk toward 0.5 when the domain is new\nupdated by human fact-check outcomes (EM optional)\npessimistic 20% quantile as the reliability used", color=GOLD)
    box(ax, 0.2, 0.35, 3.9, 1.55, "Defences against coordinated campaigns", "newness shrinkage  ·  pessimistic quantile\nburst discount: unproven sources, same stance,\nwithin 0.5 day are discounted as one voice", color=GOLD)
    arrow(ax, (7.9, 1.92), (9.7, y - 0.02), color=GOLD, rad=-0.15); arrow(ax, (4.1, 1.15), (4.68, 1.15), color=GOLD)
    ax.text(8.95, 2.35, "reliability r", fontsize=7.5, color=GOLD, ha="left")
    # decider and report
    box(ax, 11.7, 3.65, 1.2, 1.1, "6  Decider", "Platt + abstain\nm<m₀ or |P−½|<τ", color=RED)
    box(ax, 10.7, 0.35, 2.2, 1.55, "7  Source-validated report", "verdict + 80% interval\ncites only validated sources\n'not relied upon' + reason\nhuman-review flags", color=BLUE)
    arrow(ax, (xs[4] + w + 0.42, y + h / 2 + 0.1), (11.68, 4.2)); arrow(ax, (12.3, 3.63), (12.0, 1.92))
    ax.text(6.5, 5.5, "VeriCloud: evidence-grounded verification with source validation", ha="center", fontsize=12, fontweight="bold")
    savefig("architecture")
    print("wrote architecture figure")
