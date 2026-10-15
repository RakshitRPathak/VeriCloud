import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, pytest
from vericloud import *
from vericloud.textproc import Annotator
from vericloud.pipeline import train_prior


@pytest.fixture(scope="session")
def models():
    w0 = generate_world(WorldConfig(n_facts=900, seed=0))
    return train_stance_model(Annotator(w0.gaz), n_facts=900), train_prior()


@pytest.fixture(scope="session")
def small(models):
    st, pr = models
    from vericloud.benchmark import run_world
    return run_world(WorldConfig(n_facts=900, seed=21, labeled_frac=0.25, campaign_share=0.25, campaign_strength=8),
                     stance=st, prior=pr, include_oracles=True)
