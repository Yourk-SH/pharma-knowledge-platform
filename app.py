"""Streamlit 前端：通过 HTTP 调用后端 API（前后端分离）"""
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"
TIMEOUT = 120

st.set_page_config(page_title="医药文献合规问答助手", page_icon="💊", layout="wide")


def api_get(path):
    return requests.get(f"{API_BASE}{path}", timeout=10)


def api_ask(question: str) -> dict:
    resp = requests.post(
        f"{API_BASE}/ask",
        json={"question": question},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


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


def parse_answer(final_answer: str):
    """拆分后端返回：正文 / 引用来源 / 审查警告"""
    parts = final_answer.split("【引用来源】")
    answer = parts[0].strip()
    sources = parts[1].split("【审查警告】")[0].strip() if len(parts) > 1 else ""
    warning = final_answer.split("【审查警告】")[1].strip() if "【审查警告】" in final_answer else ""
    return answer, sources, warning


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
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 执行中：缓存 → 改写 → 检索 → 生成 → 合规审查..."):
            try:
                resp = api_ask(question)
                final_text = resp.get("answer", "")
                rid = resp.get("request_id", "")
            except Exception as e:
                final_text = f"请求失败：{e}"
                rid = ""

        answer, sources, warning = parse_answer(final_text)
        st.markdown(answer)
        if sources:
            with st.expander("📚 引用来源"):
                st.markdown(sources.replace("\n", "\n\n"))
        if warning:
            st.warning(f"⚠️ 审查警告：{warning}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "warning": warning,
            "request_id": rid,
            "question": question,
            "feedback": None,
        }
    )