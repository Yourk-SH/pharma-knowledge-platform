"""重排序模块：bge-reranker-v2-m3 交叉编码器精排（硅基流动 API）"""
from typing import Dict, List

import requests

import config

RERANK_URL = f"{config.BASE_URL}/rerank"


class SiliconFlowReranker:
    """cross-encoder 精排：query 和文档配对打分，比向量相似度更准"""

    def rerank(self, query: str, candidates: List[Dict], top_n: int = 3) -> List[Dict]:
        if len(candidates) <= top_n:
            return candidates

        payload = {
            "model": config.RERANK_MODEL,
            "query": query,
            "documents": [
                f"出自《{c['source'].replace('.pdf', '').replace('.txt', '')}》：{c['content']}"
                for c in candidates
            ],
            "top_n": top_n,
        }
        resp = requests.post(
            RERANK_URL,
            json=payload,
            headers={"Authorization": f"Bearer {config.API_KEY}"},
            timeout=30,
        )
        resp.raise_for_status()

        reranked = []
        for item in resp.json()["results"]:
            doc = dict(candidates[item["index"]])
            doc["rerank_score"] = round(item["relevance_score"], 4)
            reranked.append(doc)
        return reranked