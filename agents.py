"""Agent 编排模块：LangGraph 五节点流水线（v5：多轮上下文 + 真 token 流式）"""
import time
from typing import TypedDict, List, Dict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

import config
from logger import logger
from retrieval import HybridRetriever
from rerank import SiliconFlowReranker
from compliance import ComplianceChecker

_llm = None
_retriever = None
_reranker = SiliconFlowReranker()
_checker = ComplianceChecker()


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=config.LLM_MODEL,
            openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL,
            temperature=0.1,
        )
    return _llm


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


class AgentState(TypedDict):
    question: str
    history: List[Dict]
    search_query: str
    retrieved: List[Dict]
    draft: str
    final_answer: str
    trace: List[str]


def timed(node_name: str):
    """节点计时装饰器：自动记录耗时并追加到 trace 清单"""
    def decorator(fn):
        def wrapper(state: AgentState):
            t0 = time.time()
            result = fn(state) or {}
            cost = time.time() - t0
            logger.info(f"[{node_name}] 完成 ({cost:.2f}s)")
            result["trace"] = state.get("trace", []) + [f"{node_name}:{cost:.2f}s"]
            return result
        return wrapper
    return decorator


@timed("rewrite")
def rewrite_query(state: AgentState):
    """查询改写：携带最近对话历史，把省略式追问补全成独立完整的问题"""
    if state.get("search_query"):
        logger.debug("[rewrite] search_query 已预置，跳过重复改写")
        return {"search_query": state["search_query"]}

    history = state.get("history") or []
    if history:
        hist_text = "\n".join(
            f"{'用户' if h.get('role') == 'user' else '助手'}：{h.get('content', '')[:200]}"
            for h in history[-6:]
        )
        prompt = (
            "你是查询改写器。以下是最近的对话历史：\n"
            f"{hist_text}\n\n"
            f"用户最新问题：{state['question']}\n\n"
            "改写规则：\n"
            "1. 如果最新问题省略了指代（如'那XX呢'、'它的禁忌'），补全成独立完整的问题；\n"
            "2. 代词（它/这个/那个）优先继承【最近一轮】讨论的主体（通常是最近提到的药品名），"
            "而不是更早轮次的主体；\n"
            "3. 如果问题本身已经完整，原样保留。\n\n"
            "示例：\n"
            "历史：用户：阿司匹林的主要成分有哪些？\n"
            "最新问题：阿莫西林呢？ → 输出：阿莫西林的主要成分有哪些？\n"
            "历史：用户：阿莫西林的主要成分有哪些？\n"
            "最新问题：它的适应症有哪些？ → 输出：阿莫西林的适应症有哪些？\n\n"
            "只输出一句改写后的问题，不要解释："
        )
    else:
        prompt = f"把用户问题改写成适合检索医药知识库的关键词短句，只输出改写结果：{state['question']}"
    resp = get_llm().invoke(prompt)
    search_query = resp.content.strip().strip('"').strip()
    if not search_query:
        search_query = state["question"]
    logger.info(f"[rewrite] {state['question']} → {search_query}")
    return {"search_query": search_query}


@timed("retrieve")
def retrieve(state: AgentState):
    candidates = get_retriever().search(state["search_query"], top_k=config.CANDIDATE_TOP_K)
    reranked = _reranker.rerank(state["search_query"], candidates, top_n=config.RERANK_TOP_N)
    logger.debug(f"[retrieve] 召回 {len(candidates)} 条 → 精排 Top{len(reranked)}")
    return {"retrieved": reranked}


def route_after_retrieve(state: AgentState):
    if not state["retrieved"] or state["retrieved"][0].get("rerank_score", 0) < config.RERANK_SCORE_THRESHOLD:
        logger.info("路由决策：拒答（rerank 分数低于阈值）")
        return "reject"
    return "generate"


@timed("reject")
def reject(state: AgentState):
    return {"final_answer": "知识库中未检索到足够相关的信息，无法回答，请核实后重新提问。（合规要求：宁缺毋错）"}


def build_generate_prompt(state: AgentState) -> str:
    """生成提示词构造：generate 与 stream_generate 共用，保证两条路径行为一致"""
    ctx = "\n".join(
        f"[{i+1}] {r['content']}" for i, r in enumerate(state["retrieved"])
    )
    return (
        f"仅根据以下资料回答问题，必须用[n]标注引用来源，资料没有的信息回答'不知道'。\n"
        f"资料：\n{ctx}\n\n问题：{state['question']}"
    )


@timed("generate")
def generate(state: AgentState):
    resp = get_llm().invoke(build_generate_prompt(state))
    return {"draft": resp.content}


def stream_generate(state: AgentState):
    """流式生成：逐 token 产出文本片段，调用方自行累积成 draft"""
    for chunk in get_llm().stream(build_generate_prompt(state)):
        if chunk.content:
            yield chunk.content


@timed("compliance")
def compliance_check(state: AgentState):
    result = _checker.check(state["question"], state["draft"], state["retrieved"])
    logger.info(f"[compliance] {'带警告通过' if result['warnings'] else '通过'}")
    return {"final_answer": result["final_answer"]}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("compliance", compliance_check)
    graph.add_node("reject", reject)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", route_after_retrieve)
    graph.add_edge("generate", "compliance")
    graph.add_edge("compliance", END)
    graph.add_edge("reject", END)
    return graph.compile()


agent_app = build_graph()


if __name__ == "__main__":
    for q in ["二甲双胍哪些人不能用？", "感冒吃什么药好？"]:
        print(f"\n问题：{q}")
        result = agent_app.invoke({"question": q, "history": []})
        print(f"回答前100字：{result['final_answer'][:100]}...")
        print(f"链路追踪：{result['trace']}")