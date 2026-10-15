import sys, pathlib, json, warnings
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

RES = ROOT / "results"; FIG = RES / "figures"; TAB = RES / "tables"
for d in (RES, FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

PAL = {"Top-1 evidence": "#b8bcc2", "Majority vote": "#9aa0a6", "Credibility prior (no learning)": "#d98e04",
       "Truth discovery (EM, no independence)": "#2a9d8f", "VeriCloud": "#c8102e",
       "Oracle reliability": "#4f86c6", "Oracle reliability + stance": "#2b5c9a"}


def style():
    plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .25, "font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "legend.frameon": False})


def savefig(name):
    plt.tight_layout(); plt.savefig(FIG / f"{name}.png"); plt.savefig(FIG / f"{name}.pdf"); plt.close()


def save_json(name, obj):
    def conv(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    (RES / name).write_text(json.dumps(obj, indent=2, default=conv))


def to_latex(df, caption, label, fmt="{:.3f}", bold_best=None):
    """booktabs table; df entries already strings or numbers"""
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\small",
             "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", " & ".join([""] + [c.replace("_", "\\_") for c in cols]) + " \\\\", "\\midrule"]
    for idx, row in df.iterrows():
        cells = [v if isinstance(v, str) else fmt.format(v) for v in row.values]
        lines.append(" & ".join([str(idx).replace("&", "\\&")] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines) + "\n"


_ST = _PR = None
def models():
    """stance model + metadata prior trained ONCE per process on worlds disjoint from every evaluation world"""
    global _ST, _PR
    if _ST is None:
        from vericloud import generate_world, WorldConfig, train_stance_model
        from vericloud.textproc import Annotator
        from vericloud.pipeline import train_prior
        w0 = generate_world(WorldConfig(n_facts=200, seed=0))
        _ST = train_stance_model(Annotator(w0.gaz)); _PR = train_prior()
    return _ST, _PR


def pmap(fn, args, workers=2):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, args))


def apply_stance_noise(table, eps, seed=0):
    if eps <= 0: return table
    rng = np.random.default_rng(seed); t = dict(table)
    P = np.stack([t["pS"], t["pR"], t["pN"]], 1).copy()
    flip = rng.random(len(P)) < eps
    P[flip] = rng.dirichlet(np.ones(3), size=flip.sum())
    t["pS"], t["pR"], t["pN"] = P[:, 0], P[:, 1], P[:, 2]
    return t
