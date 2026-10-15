"""E6: efficiency - index build time and single-claim verification latency vs corpus size (2 CPU cores, pure Python/NumPy)."""
from common import *
import time, pandas as pd
from vericloud import *
from vericloud.benchmark import split_masks
from vericloud.pipeline import VeriCloud
from vericloud.report import verify_text

if __name__ == "__main__":
    st, pr = models(); rows = []
    for nf in (500, 1000, 2000, 3000):
        t = time.time(); w = generate_world(WorldConfig(n_facts=nf, seed=700, labeled_frac=0.2, campaign_share=0.2, campaign_strength=8)); tg = time.time() - t
        nc = VeriCloud(w, stance=st, prior_model=pr)
        claims = w.claims; sub = claims[:min(1000, len(claims))]; masks = split_masks(sub)
        tab = nc.build_table(sub)
        nc.learn_sources(tab, sub, np.array([c.verified for c in sub]) & masks[0]); nc.calibrate(tab, sub, masks[1])
        rng = np.random.default_rng(0); probe = [claims[int(i)] for i in rng.choice(len(claims), 60, replace=False)]
        verify_text(nc, probe[0].text, probe[0].t, mc=100)                      # warm-up
        lat = []
        for c in probe:
            t = time.perf_counter(); verify_text(nc, c.text, c.t, mc=100); lat.append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); nc.build_table(probe); batch = (time.perf_counter() - t)
        rows.append(dict(facts=nf, articles=len(w.articles), index_build_s=nc.index_seconds, p50_ms=np.percentile(lat, 50), p95_ms=np.percentile(lat, 95),
                         batch_claims_per_s=len(probe) / batch))
        print(rows[-1])
    df = pd.DataFrame(rows); df.to_csv(RES / "e6_efficiency.csv", index=False)
    (TAB / "e6_efficiency.md").write_text(df.round(2).to_markdown(index=False))
    style(); fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    ax[0].plot(df.articles, df.p50_ms, "o-", color="#c8102e", label="p50"); ax[0].plot(df.articles, df.p95_ms, "s--", color="#9aa0a6", label="p95")
    ax[0].set_xlabel("articles indexed"); ax[0].set_ylabel("single-claim verification latency (ms)"); ax[0].legend(); ax[0].set_title("Latency (retrieve + stance + fuse + MC interval)")
    ax[1].plot(df.articles, df.index_build_s, "o-", color="#4f86c6"); ax[1].set_xlabel("articles indexed"); ax[1].set_ylabel("index build time (s)"); ax[1].set_title("Index build")
    savefig("e6_efficiency")
