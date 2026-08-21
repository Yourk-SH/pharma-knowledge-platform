"""Agent 编排模块：LangGraph 五节点流水线（v3：全链路追踪）"""
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
    resp = get_llm().invoke(
        f"把用户问题改写成适合检索医药知识库的关键词短句，只输出改写结果：{state['question']}"
    )
    logger.debug(f"[rewrite] 结果: {resp.content}")
    return {"search_query": resp.content}


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


@timed("generate")
def generate(state: AgentState):
    ctx = "\n".join(
        f"[{i+1}] {r['content']}" for i, r in enumerate(state["retrieved"])
    )
    resp = get_llm().invoke(
        f"仅根据以下资料回答问题，必须用[n]标注引用来源，资料没有的信息回答'不知道'。\n"
        f"资料：\n{ctx}\n\n问题：{state['question']}"
    )
    return {"draft": resp.content}


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
        result = agent_app.invoke({"question": q})
        print(f"回答前100字：{result['final_answer'][:100]}...")
        print(f"链路追踪：{result['trace']}")