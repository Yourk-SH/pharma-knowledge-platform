"""缓存层单元测试：重点验证降级保护与阈值逻辑（不依赖真实 Redis/MySQL/API）"""
import numpy as np

import cache
import config
from cache import RedisLayer, FAQStore, _qkey

BAD_REDIS_URL = "redis://127.0.0.1:1/0"  # 不可达端口，强制触发降级


def test_qkey_stable():
    """缓存键稳定性：同问题同键，首尾空格不影响"""
    assert _qkey("二甲双胍") == _qkey("二甲双胍")
    assert _qkey("二甲双胍") != _qkey("阿司匹林")
    assert _qkey(" 二甲双胍 ") == _qkey("二甲双胍")


def test_redis_degrade_to_memory(monkeypatch):
    """Redis 不可用 → 自动降级内存缓存，读写照常"""
    monkeypatch.setattr(config, "REDIS_URL", BAD_REDIS_URL)
    layer = RedisLayer()
    assert layer._client is None
    layer.set("测试问题", "测试答案")
    assert layer.get("测试问题") == {"answer": "测试答案"}


def test_redis_memory_ttl_expire(monkeypatch):
    """内存降级模式下 TTL 过期后读不到"""
    monkeypatch.setattr(config, "REDIS_URL", BAD_REDIS_URL)
    monkeypatch.setattr(config, "REDIS_TTL", -1)
    layer = RedisLayer()
    layer.set("过期问题", "过期答案")
    assert layer.get("过期问题") is None


class _FakeEmbeddings:
    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, q):
        return self.vec


def _bare_store():
    """绕过构造函数创建 FAQStore（避免真实 MySQL/embedding 调用）"""
    store = FAQStore.__new__(FAQStore)
    store.faq = [{"question": "二甲双胍哪些人不能用", "answer": "禁忌答案"}]
    store._vecs = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    return store


def test_faq_match_above_threshold(monkeypatch):
    """相似度 ≥ 0.85 → 命中返回答案"""
    monkeypatch.setattr(cache, "get_embeddings", lambda: _FakeEmbeddings([0.99, 0.05, 0.0]))
    answer, sim = _bare_store().match("任意问题")
    assert answer == "禁忌答案"
    assert sim >= config.FAQ_SIM_THRESHOLD


def test_faq_match_below_threshold(monkeypatch):
    """相似度 < 0.85 → 拒答（宁缺毋错）"""
    monkeypatch.setattr(cache, "get_embeddings", lambda: _FakeEmbeddings([0.0, 1.0, 0.0]))
    answer, sim = _bare_store().match("任意问题")
    assert answer is None
    assert sim < config.FAQ_SIM_THRESHOLD