"""合规审查模块：GxP 风格的回答护栏"""
import re
from typing import Dict, List

import config

CITATION_PATTERN = re.compile(r"\[\d+\]")
DISCLAIMER = "\n\n⚠️ 合规提示：本内容由AI基于文献生成，仅供专业人士参考，不构成用药建议。"


class ComplianceChecker:
    """对生成结果做合规审查，输出结构化审查报告"""

    def check(self, question: str, draft: str, retrieved: List[Dict]) -> Dict:
        warnings: List[str] = []
        answer = draft.strip()

        # ===== 规则1：防幻觉 —— 回答必须包含引用标注 =====
        if not CITATION_PATTERN.search(answer):
            warnings.append("回答缺少引用标注，可能存在幻觉风险")

        # ===== 规则2：敏感词 —— 用药相关内容强制附加免责声明 =====
        sensitive_hits = [
            w for w in config.SENSITIVE_WORDS
            if w in question or w in answer
        ]
        if sensitive_hits:
            answer += DISCLAIMER

        # ===== 规则3：引用来源清单 —— 可追溯性 =====
        source_lines = "\n".join(
            f"  [{i + 1}] {r['source']}（重排分数 {r.get('rerank_score', '-')}）"
            for i, r in enumerate(retrieved)
        )

        # ===== 组装最终输出 =====
        final = f"{answer}\n\n【引用来源】\n{source_lines}"
        if warnings:
            final += f"\n【审查警告】{'; '.join(warnings)}"

        return {
            "answer": answer,
            "sources": source_lines,
            "warnings": warnings,
            "final_answer": final,
        }