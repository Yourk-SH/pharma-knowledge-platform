"""Streamlit 前端：SSE token 级流式 + 多轮对话 + 反馈闭环"""
import json
import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")
TIMEOUT = 120

st.set_page_config(page_title="医药文献合规问答助手", page_icon="💊", layout="wide")


def api_get(path):
    return requests.get(f"{API_BASE}{path}", timeout=10)


def send_feedback(msg: dict, rating: str):
    """点赞/点踩上报后端（失败静默，不影响用户体验）"""
    try:
        requests.post(
            f"{API_BASE}/feedback",
            json={
                "request_id": msg.get("request_id", ""),
                "question": msg.get("question", ""),
                "answer": msg.get("content", ""),
                "rating": rating,
            },
            timeout=5,
        )
    except Exception:
        pass


def build_history():
    """提取真实问答轮次（跳过欢迎语），构造 API history 格式"""
    hist = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            hist.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m.get("request_id"):
            hist.append({"role": "assistant", "content": m["content"][:300]})
    return hist[-8:]


def parse_answer(final_answer: str):
    """拆分后端完整回答：正文 / 引用来源 / 审查警告"""
    parts = final_answer.split("【引用来源】")
    answer = parts[0].strip()
    sources = parts[1].split("【审查警告】")[0].strip() if len(parts) > 1 else ""
    warning = final_answer.split("【审查警告】")[1].strip() if "【审查警告】" in final_answer else ""
    return answer, sources, warning


def stream_ask(question: str, history: list, status_box=None):
    """SSE 流式问答：返回 (文本流生成器, 元信息可变字典)"""
    meta = {"request_id": "", "trace": [], "final": "", "cached": None, "sources_hit": []}

    def gen():
        with requests.post(
            f"{API_BASE}/ask_stream",
            json={"question": question, "history": history},
            stream=True,
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data: "):
                    continue
                data = json.loads(raw[6:])
                if "chunk" in data:
                    yield data["chunk"]
                elif "search_query" in data:
                    meta["request_id"] = data.get("request_id", "")
                    if status_box:
                        status_box.caption(f"🔍 查询改写：{data['search_query']}")
                elif "count" in data and "sources" in data:
                    meta["sources_hit"] = data.get("sources", [])
                    if status_box:
                        status_box.caption(f"📚 检索完成：命中 {data['count']} 条资料")
                elif "final_answer" in data:
                    meta["final"] = data["final_answer"]
                elif "trace" in data:
                    meta["request_id"] = data.get("request_id", meta["request_id"])
                    meta["trace"] = data.get("trace", [])
                    meta["cached"] = data.get("cached")
                elif "detail" in data:
                    yield f"\n\n❌ 请求失败：{data['detail']}"

    return gen(), meta


# ===== 后端连通性检查 =====
try:
    health = api_get("/health").json()
    backend_ok = health.get("status") == "ok"
except Exception:
    backend_ok = False

if not backend_ok:
    st.error("❌ 无法连接后端服务。请先在另一个终端运行：python api.py")
    st.stop()

# ===== 侧边栏 =====
with st.sidebar:
    st.title("💊 PharmaDoc Agent")
    st.caption("医药文献智能问答 · 引用溯源 · GxP 合规护栏")
    st.divider()

    kb = api_get("/knowledge").json()
    st.markdown(f"**知识库**：{kb['parent_chunks']} 个父块 / {kb['child_chunks']} 个子块")
    with st.expander("查看文档来源"):
        for s in kb["sources"]:
            st.markdown(f"- {s}")
    st.divider()

    st.markdown("**一键测试**")
    if st.button("示例1：禁忌症查询"):
        st.session_state.pending = "二甲双胍哪些人不能用？"
    if st.button("示例2：用法用量"):
        st.session_state.pending = "奥美拉唑一天吃几次？"
    if st.button("示例3：知识域外（会拒答）"):
        st.session_state.pending = "感冒吃什么药好？"
    if st.button("🗑️ 清空对话"):
        st.session_state.messages = []
        st.rerun()

# ===== 会话状态 =====
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "您好！我是医药文献问答助手。我的回答都会附带引用来源，知识范围外的问题我会如实说明。"}
    ]

# ===== 渲染历史 =====
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 引用来源"):
                st.markdown(msg["sources"])
        if msg.get("warning"):
            st.warning(f"⚠️ 审查警告：{msg['warning']}")
        if msg["role"] == "assistant" and msg.get("request_id"):
            if msg.get("feedback") is None:
                col1, col2, _ = st.columns([1, 1, 6])
                if col1.button("👍", key=f"up_{idx}", help="回答准确"):
                    send_feedback(msg, "up")
                    msg["feedback"] = "up"
                    st.rerun()
                if col2.button("👎", key=f"down_{idx}", help="回答不准确"):
                    send_feedback(msg, "down")
                    msg["feedback"] = "down"
                    st.rerun()
            else:
                st.caption("✅ 已收到反馈，谢谢！")

# ===== 输入处理 =====
question = st.chat_input("输入你的问题，例如：二甲双胍哪些人不能用？")
if not question and st.session_state.get("pending"):
    question = st.session_state.pending
    st.session_state.pending = None

if question:
    history = build_history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status_box = st.empty()
        try:
            gen, meta = stream_ask(question, history, status_box)
            raw = st.write_stream(gen)
        except Exception as e:
            raw = f"请求失败：{e}"
            meta = {"request_id": "", "trace": [], "final": "", "cached": None, "sources_hit": []}
        status_box.empty()

        answer, sources, warning = parse_answer(meta.get("final") or (raw or ""))
        if meta.get("cached"):
            st.caption(f"⚡ 缓存命中：{meta['cached']}")
        if sources:
            with st.expander("📚 引用来源"):
                st.markdown(sources.replace("\n", "\n\n"))
        if warning:
            st.warning(f"⚠️ 审查警告：{warning}")
        if meta.get("trace"):
            with st.expander("🔍 链路追踪"):
                st.code(" → ".join(meta["trace"]))

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "warning": warning,
            "request_id": meta.get("request_id", ""),
            "question": question,
            "feedback": None,
        }
    )