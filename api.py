"""FastAPI 服务层：HTTP 接口（v4：三级缓存 + 请求级追踪 + 反馈闭环）"""
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents import agent_app
from cache import cache_lookup, cache_writeback
from config import BASE_DIR
from knowledge import load_index
from logger import logger, set_request_id

BADCASE_PATH = BASE_DIR / "logs" / "badcases.jsonl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    parents, children, _ = load_index()
    logger.info(f"知识库就绪：{len(parents)} 个父块 / {len(children)} 个子块")
    yield


app = FastAPI(
    title="PharmaDoc Agent API",
    description="医药文献合规问答服务",
    version="1.3.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")


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
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    t0 = time.time()
    logger.info(f"收到请求: {req.question}")

    try:
        cached, source = cache_lookup(req.question)
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
        result = agent_app.invoke({"question": req.question})
        cost = time.time() - t0
        logger.info(f"请求完成: 总耗时 {cost:.2f}s | trace: {result['trace']}")
        cache_writeback(req.question, result["final_answer"])
        return AskResponse(
            question=req.question,
            answer=result["final_answer"],
            request_id=rid,
            trace=result["trace"],
        )
    except Exception as e:
        logger.exception(f"请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败：{e}")


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """用户反馈收集：点踩样本落盘，定期回流评测集"""
    if req.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating 必须是 up 或 down")
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "request_id": req.request_id,
        "question": req.question,
        "answer": req.answer[:200],
        "rating": req.rating,
        "comment": req.comment,
    }
    BADCASE_PATH.parent.mkdir(exist_ok=True)
    with open(BADCASE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"收到反馈 [{req.rating}]: {req.question[:50]}")
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)