"""评测模块：检索命中率与域外拒答准确率量化评估"""
import time

import config
from retrieval import HybridRetriever
from rerank import SiliconFlowReranker

# ===== 评测集：in 检查来源命中，out 检查是否正确拒答 =====
EVAL_SET = [
    # --- 域内：应命中对应说明书 ---
    {"question": "二甲双胍哪些人不能用", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "二甲双胍一天吃几次", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "二甲双胍能和酒一起服用吗", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "吃二甲双胍需要定期检查什么", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "二甲双胍缓释片怎么吃", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "吃二甲双胍有什么不良反应", "expected": "二甲双胍缓释片说明书.pdf", "type": "in"},
    {"question": "阿司匹林肠溶片的适应症是什么", "expected": "阿司匹林肠溶片说明书.pdf", "type": "in"},
    {"question": "阿司匹林的用法用量", "expected": "阿司匹林肠溶片说明书.pdf", "type": "in"},
    {"question": "阿司匹林肠溶片一天吃几次", "expected": "阿司匹林肠溶片说明书.pdf", "type": "in"},
    {"question": "阿司匹林有什么不良反应", "expected": "阿司匹林肠溶片说明书.pdf", "type": "in"},
    {"question": "阿莫西林胶囊的不良反应有哪些", "expected": "阿莫西林胶囊说明书.pdf", "type": "in"},
    {"question": "阿莫西林一天吃几次", "expected": "阿莫西林胶囊说明书.pdf", "type": "in"},
    {"question": "阿莫西林胶囊一次吃几粒", "expected": "阿莫西林胶囊说明书.pdf", "type": "in"},
    {"question": "青霉素过敏能吃阿莫西林吗", "expected": "阿莫西林胶囊说明书.pdf", "type": "in"},
    {"question": "芬必得是治什么的", "expected": "芬必得布洛芬缓释胶囊说明书.pdf", "type": "in"},
    {"question": "芬必得一天最多吃多少", "expected": "芬必得布洛芬缓释胶囊说明书.pdf", "type": "in"},
    {"question": "芬必得不能和什么药一起吃", "expected": "芬必得布洛芬缓释胶囊说明书.pdf", "type": "in"},
    {"question": "布洛芬缓释胶囊有什么副作用", "expected": "芬必得布洛芬缓释胶囊说明书.pdf", "type": "in"},
    {"question": "奥美拉唑的适应症是什么", "expected": "奥美拉唑肠溶胶囊说明书.pdf", "type": "in"},
    {"question": "奥美拉唑一天吃几次", "expected": "奥美拉唑肠溶胶囊说明书.pdf", "type": "in"},
    {"question": "奥美拉唑肠溶胶囊饭前吃还是饭后吃", "expected": "奥美拉唑肠溶胶囊说明书.pdf", "type": "in"},
    {"question": "奥美拉唑能和什么药发生相互作用", "expected": "奥美拉唑肠溶胶囊说明书.pdf", "type": "in"},
    # --- 域外：应拒答 ---
    {"question": "糖尿病能彻底治愈吗", "type": "out"},
    {"question": "推荐几个保健品", "type": "out"},
    {"question": "帮我翻译一段英文", "type": "out"},
    {"question": "附近有什么医院", "type": "out"},
    {"question": "减肥药哪个牌子好", "type": "out"},
    {"question": "感冒吃什么药好", "type": "out"},
    {"question": "高血压怎么根治", "type": "out"},
    {"question": "今天天气怎么样", "type": "out"},
    {"question": "帮我写一首诗", "type": "out"},
    {"question": "股票怎么买", "type": "out"},
]


def run_evaluation():
    retriever = HybridRetriever()
    reranker = SiliconFlowReranker()

    hit, total_in = 0, 0
    top1_hit = 0
    reject_correct, total_out = 0, 0
    details = []

    for case in EVAL_SET:
        start = time.time()
        candidates = retriever.search(case["question"], top_k=config.CANDIDATE_TOP_K)
        reranked = reranker.rerank(case["question"], candidates, top_n=config.RERANK_TOP_N)
        cost = round(time.time() - start, 2)

        top_score = reranked[0]["rerank_score"] if reranked else 0
        rejected = top_score < config.RERANK_SCORE_THRESHOLD

        if case["type"] == "in":
            total_in += 1
            sources = [r["source"] for r in reranked]
            ok = case["expected"] in sources
            hit += ok
            if reranked and reranked[0]["source"] == case["expected"]:
                top1_hit += 1
            details.append(
                f"  {'✅' if ok else '❌'} [{case['question']}] top={top_score} "
                f"Top1={sources[0] if sources else '-'} ({cost}s)"
            )
        else:
            total_out += 1
            ok = rejected
            reject_correct += ok
            details.append(
                f"  {'✅' if ok else '❌'} [{case['question']}] top={top_score} 拒答={rejected} ({cost}s)"
            )

    print("=" * 60)
    print("评测明细：")
    print("\n".join(details))
    print("=" * 60)
    print(f"来源命中率（域内 {total_in} 条，Top3 口径）：{hit}/{total_in} = {hit/total_in:.1%}")
    print(f"Top-1 准确率（域内 {total_in} 条）：{top1_hit}/{total_in} = {top1_hit/total_in:.1%}")
    print(f"拒答准确率（域外 {total_out} 条）：{reject_correct}/{total_out} = {reject_correct/total_out:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()