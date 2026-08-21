"""FastAPI 服务层：HTTP 接口（v5：真 token 流式 + 多轮对话 + 三级缓存）"""
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents import (
    agent_app,
    compliance_check,
    reject,
    retrieve,
    rewrite_query,
    route_after_retrieve,
    stream_generate,
)
from cache import cache_lookup, cache_writeback
from config import BASE_DIR
from knowledge import load_index
from logger import logger, set_request_id

BADCASE_PATH = BASE_DIR / "logs" / "badcases.jsonl"

from cache import cache_lookup, cache_writeback, warmup_cache
@asynccontextmanager
async def lifespan(app: FastAPI):
    parents, children, _ = load_index()
    logger.info(f"知识库就绪：{len(parents)} 个父块 / {len(children)} 个子块")
    warmup_cache()
    yield


app = FastAPI(
    title="PharmaDoc Agent API",
    description="医药文献合规问答服务",
    version="1.4.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")
    history: List[dict] = Field(default_factory=list, description="对话历史 [{'role','content'}]")


class AskResponse(BaseModel):
    question: str
    answer: str
    request_id: str
    trace: list


class FeedbackRequest(BaseModel):
    request_id: str = ""
    question: str
    answer: str = ""
    rating: str = Field(..., description="up / down")
    comment: str = ""


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/knowledge")
def list_knowledge():
    parents, children, _ = load_index()
    sources = sorted({p["source"] for p in parents})
    return {"parent_chunks": len(parents), "child_chunks": len(children), "sources": sources}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """非流式接口（保留：供压测脚本与第三方集成使用）"""
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    t0 = time.time()
    history = req.history[-8:]
    logger.info(f"收到请求: {req.question}")

    search_query = rewrite_query({"question": req.question, "history": history})["search_query"]

    try:
        cached, source = cache_lookup(search_query)
    except Exception as e:
        logger.exception(f"缓存层异常，降级走 RAG: {e}")
        cached, source = None, None

    if cached:
        logger.info(f"缓存命中 [{source}]，直接返回")
        return AskResponse(
            question=req.question,
            answer=cached,
            request_id=rid,
            trace=[f"cache_hit:{source}"],
        )

    try:
        result = agent_app.invoke(
            {"question": search_query, "search_query": search_query, "history": history}
        )
        cost = time.time() - t0
        logger.info(f"请求完成: 总耗时 {cost:.2f}s | trace: {result['trace']}")
        cache_writeback(search_query, result["final_answer"])
        return AskResponse(
            question=req.question,
            answer=result["final_answer"],
            request_id=rid,
            trace=result["trace"],
        )
    except Exception as e:
        logger.exception(f"请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败：{e}")


@app.post("/ask_stream")
def ask_stream(req: AskRequest):
    """SSE 流式接口：直接调用节点函数 + LLM token 级流式（绕过图的节点粒度输出）"""
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    logger.info(f"收到流式请求: {req.question}")

    def event_stream():
        t0 = time.time()
        state = {"question": req.question, "history": req.history[-8:], "trace": []}
        try:
            state.update(rewrite_query(state))
            # 关键：后续检索/生成/合规一律使用补全后的独立问题，
            # 避免生成节点只看到"阿莫西林呢？"这种省略句而答非所问
            state["question"] = state["search_query"]
            yield _sse("rewrite", {"search_query": state["search_query"], "request_id": rid})

            try:
                cached, source = cache_lookup(state["search_query"])
            except Exception as e:
                logger.exception(f"缓存层异常，降级走 RAG: {e}")
                cached, source = None, None

            if cached:
                logger.info(f"缓存命中 [{source}]")
                yield _sse("answer", {"chunk": cached})
                yield _sse("done", {
                    "request_id": rid,
                    "trace": state["trace"] + [f"cache_hit:{source}"],
                    "total_cost": round(time.time() - t0, 2),
                    "cached": source,
                })
                return

            state.update(retrieve(state))
            hits = [r.get("source", "") for r in state.get("retrieved", [])]
            yield _sse("retrieve", {"count": len(hits), "sources": hits})

            if route_after_retrieve(state) == "reject":
                state.update(reject(state))
                yield _sse("answer", {"chunk": state["final_answer"]})
            else:
                tg = time.time()
                parts = []
                for token in stream_generate(state):
                    parts.append(token)
                    yield _sse("answer", {"chunk": token})
                state["draft"] = "".join(parts)
                gen_cost = time.time() - tg
                logger.info(f"[generate] 完成 ({gen_cost:.2f}s)")
                state["trace"] = state["trace"] + [f"generate:{gen_cost:.2f}s"]
                state.update(compliance_check(state))
                yield _sse("meta", {"final_answer": state["final_answer"]})

            cache_writeback(state["search_query"], state["final_answer"])
            logger.info(f"流式请求完成: 总耗时 {time.time()-t0:.2f}s | trace: {state['trace']}")
            yield _sse("done", {
                "request_id": rid,
                "trace": state["trace"],
                "total_cost": round(time.time() - t0, 2),
                "cached": None,
            })
        except Exception as e:
            logger.exception(f"流式请求失败: {e}")
            yield _sse("error", {"detail": str(e)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """用户反馈收集：点踩样本落盘，定期回流评测集"""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating 必须是 up 或 down")
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request_id": req.request_id,
        "question": req.question,
        "answer": req.answer[:500],
        "rating": req.rating,
        "comment": req.comment,
    }
    BADCASE_PATH.parent.mkdir(exist_ok=True)
    with open(BADCASE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"收到反馈 [{req.rating}]: {req.question[:50]}")
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=8000)