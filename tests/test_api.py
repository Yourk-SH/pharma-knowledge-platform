"""API 契约测试：只测不烧 LLM 额度的接口（/ask 全流程走本地集成测试）"""
import json

from fastapi.testclient import TestClient

import api

# 不进入 with 上下文 → 不触发 lifespan → 不加载知识库/缓存（CI 安全）
client = TestClient(api.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_empty_question_rejected():
    """空问题被参数校验拦截"""
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


def test_feedback_invalid_rating():
    resp = client.post("/feedback", json={"question": "测试", "rating": "bad"})
    assert resp.status_code == 400


def test_feedback_write_jsonl(tmp_path, monkeypatch):
    """合法反馈落盘 jsonl，字段完整"""
    monkeypatch.setattr(api, "BADCASE_PATH", tmp_path / "logs" / "badcases.jsonl")
    resp = client.post(
        "/feedback",
        json={"question": "测试问题", "answer": "测试答案", "rating": "down"},
    )
    assert resp.status_code == 200
    record = json.loads(
        (tmp_path / "logs" / "badcases.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["rating"] == "down"
    assert record["question"] == "测试问题"