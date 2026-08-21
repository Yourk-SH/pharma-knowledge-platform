"""混合检索模块：BM25 + 向量双路召回，RRF 融合，父子块映射"""
import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

import config
from knowledge import get_embeddings, load_index


def cosine_similarity_py(vec_a: List[float], vec_b: List[float]) -> float:
    """纯 Python 版余弦相似度（面试手写友好版本）"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_np(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """numpy 版批量余弦相似度（生产环境用）"""
    norms_q = np.linalg.norm(query_vec, axis=1, keepdims=True)
    norms_d = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
    return np.dot(query_vec, doc_vecs.T) / (norms_q * norms_d.T)


class SimpleBM25:
    """简易 BM25 实现"""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        self.doc_lens = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lens) / self.N if self.N else 1
        self.doc_freqs = [Counter(doc) for doc in corpus]
        self.df: Dict[str, int] = {}
        for doc in corpus:
            for word in set(doc):
                self.df[word] = self.df.get(word, 0) + 1

    def _idf(self, word: str) -> float:
        df = self.df.get(word, 0)
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_words: List[str], doc_idx: int) -> float:
        s = 0.0
        doc_freq = self.doc_freqs[doc_idx]
        doc_len = self.doc_lens[doc_idx]
        for word in query_words:
            tf = doc_freq.get(word, 0)
            if tf == 0:
                continue
            idf = self._idf(word)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            s += idf * numerator / denominator
        return s

    def rank(self, query_words: List[str], top_k: int = 5) -> List[int]:
        scores = [(i, self.score(query_words, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in scores[:top_k] if s > 0]


def rrf_fusion(rank_lists: List[List[int]], k: int = 60) -> List[int]:
    """RRF 融合：只看排名不看分数"""
    scores: Dict[int, float] = {}
    for ranked_docs in rank_lists:
        for rank, doc_id in enumerate(ranked_docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)


class HybridRetriever:
    """混合检索器：子块双路召回，RRF 融合，回溯父块返回"""

    def __init__(self):
        self.parents, self.children, vecs = load_index()
        self.child_vecs = np.asarray(vecs, dtype=np.float32)
        self.tokenizer = None
        self.bm25 = SimpleBM25(
            corpus=[self._tokenize(c["content"]) for c in self.children]
        )
        print(f"✅ 检索器就绪：{len(self.parents)} 个父块 / {len(self.children)} 个子块")

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            import jieba
            self.tokenizer = jieba
        return list(self.tokenizer.cut(text))

    def search(self, query: str, top_k: int = 2) -> List[Dict]:
        """子块双路召回 → RRF 融合 → 回溯父块（同一父块去重取最高分）"""
        tokens = self._tokenize(query)

        bm25_ids = self.bm25.rank(tokens, top_k=20)

        query_vec = np.array([get_embeddings().embed_query(query)], dtype=np.float32)
        sims = cosine_similarity_np(query_vec, self.child_vecs)[0]
        vector_ids = np.argsort(sims)[::-1][:20].tolist()

        fused_child_ids = rrf_fusion([bm25_ids, vector_ids])

        best_parent: Dict[int, Tuple[float, int]] = {}
        for cid in fused_child_ids:
            pid = self.children[cid]["parent_id"]
            score = float(sims[cid])
            if pid not in best_parent or score > best_parent[pid][0]:
                best_parent[pid] = (score, cid)

        results = [
            {
                "content": self.parents[pid]["content"],
                "source": self.parents[pid]["source"],
                "score": round(score, 4),
            }
            for pid, (score, _) in sorted(best_parent.items(), key=lambda x: -x[1][0])
        ]
        return results[:top_k]