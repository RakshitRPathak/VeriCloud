"""Evidence-independence analysis: syndicated / copied / near-duplicate articles must not count as independent
confirmations.  Evidence items of one claim are clustered (union-find) by (i) TF-IDF cosine >= theta, or (ii) an
explicit syndication tag naming another evidence item's outlet.  In each cluster the earliest article keeps weight 1;
copies receive `copy_weight` each (total capped)."""
from __future__ import annotations
import numpy as np


def cluster_weights(ids, wtf, times, src_of, src_names, via_tags, theta=0.80, copy_weight=0.10, cap=0.5):
    k = len(ids)
    if k == 0:
        return np.zeros(0), np.zeros(0, int)
    X = wtf[ids]
    S = (X @ X.T).toarray()
    parent = list(range(k))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i in range(k):
        for j in range(i + 1, k):
            if S[i, j] >= theta:
                parent[find(i)] = find(j)
    names = [src_names[src_of[i]] for i in ids]
    for i in range(k):
        tag = via_tags[ids[i]]
        if tag:
            for j in range(k):
                if j != i and names[j] == tag:
                    parent[find(i)] = find(j)
    cl = np.array([find(i) for i in range(k)])
    w = np.ones(k)
    for c in set(cl.tolist()):
        mem = np.flatnonzero(cl == c)
        if len(mem) > 1:
            order = mem[np.argsort([times[ids[m]] for m in mem])]
            copies = order[1:]
            w[copies] = min(copy_weight, cap / len(copies))
    return w, cl
