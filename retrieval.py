"""Hybrid evidence retrieval: BM25 (exact terms) + char n-gram TF-IDF (typo-tolerant) + LSA (dense), fused by
reciprocal-rank fusion.  Retrieval is time-causal: a claim only sees articles published before it."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from .textproc import tokenize


class HybridRetriever:
    def __init__(self, texts, times, k1=1.4, b=0.75, lsa_dim=96, methods=("bm25", "char"), weights=None, seed=0):
        self.weights = {"bm25": 1.0, "char": 1.0, "lsa": 0.25}   # LSA is optional; it lowered recall in our ablation so it is off by default
        if weights: self.weights.update(weights)
        self.times = np.asarray(times); self.n = len(texts); self.methods = methods
        if "bm25" in methods:
            self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
            tf = self.cv.fit_transform(texts).tocsr().astype(np.float32)
            dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
            df = np.asarray((tf > 0).sum(0)).ravel()
            idf = np.log(1 + (self.n - df + .5) / (df + .5)).astype(np.float32)
            tf = tf.tocoo()
            denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avg)
            w = idf[tf.col] * tf.data * (k1 + 1) / denom
            self.bm25 = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()      # (V, N)
        if "char" in methods:
            self.cvec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, dtype=np.float32)
            self.char = self.cvec.fit_transform(texts).T.tocsr()
        # word tf-idf also used for near-duplicate detection / lexical similarity
        self.wvec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True, dtype=np.float32)
        self.wtf = self.wvec.fit_transform(texts).tocsr()
        if "lsa" in methods:
            self.svd = TruncatedSVD(lsa_dim, random_state=seed)
            D = self.svd.fit_transform(self.wtf)
            self.lsa = (D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)).astype(np.float32)

    def _topn(self, S, n):
        idx = np.argpartition(-S, n, axis=1)[:, :n]
        sc = np.take_along_axis(S, idx, 1)
        o = np.argsort(-sc, axis=1)
        return np.take_along_axis(idx, o, 1), np.take_along_axis(sc, o, 1)

    def search(self, queries, qtimes, k=15, per_method=40, chunk=120, rrf_k=60.0, causal=True, return_scores=False):
        """returns (n_queries, k) array of article ids (-1 padded) fused over the enabled methods"""
        per_method = max(1, min(per_method, self.n - 1)); kk = min(k, per_method)
        out = -np.ones((len(queries), k), np.int64)
        for a in range(0, len(queries), chunk):
            qs = queries[a:a + chunk]; qt = np.asarray(qtimes[a:a + chunk])
            mask = (self.times[None, :] > qt[:, None]) if causal else None
            fused = {}
            mats = []
            if "bm25" in self.methods:
                Q = self.cv.transform(qs); Q.data[:] = 1
                mats.append(("bm25", (Q @ self.bm25).toarray()))
            if "char" in self.methods:
                mats.append(("char", (self.cvec.transform(qs) @ self.char).toarray()))
            if "lsa" in self.methods:
                ql = self.svd.transform(self.wvec.transform(qs)); ql /= np.linalg.norm(ql, axis=1, keepdims=True) + 1e-9
                mats.append(("lsa", ql.astype(np.float32) @ self.lsa.T))
            acc = np.zeros((len(qs), self.n), np.float32)
            for name, S in mats:
                S = np.asarray(S, np.float32)
                if mask is not None:
                    S[mask] = -1e9
                idx, sc = self._topn(S, per_method)
                r = 1.0 / (rrf_k + np.arange(1, per_method + 1))[None, :]
                ok = sc > -1e8
                np.add.at(acc, (np.repeat(np.arange(len(qs)), per_method), idx.ravel()), (self.weights[name] * r * ok).ravel())
            if mask is not None:
                acc[mask] = -1
            top, ts = self._topn(acc, kk)
            top[ts <= 0] = -1
            out[a:a + chunk, :kk] = top
        return out
