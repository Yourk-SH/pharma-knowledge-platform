"""合规审查单元测试：纯规则逻辑，不依赖任何外部服务"""
from compliance import ComplianceChecker, DISCLAIMER

RETRIEVED = [
    {"content": "资料一", "source": "阿司匹林肠溶片说明书.pdf", "rerank_score": 0.9},
]


def test_missing_citation_triggers_warning():
    """无 [n] 引用标注 → 幻觉风险警告"""
    result = ComplianceChecker().check("天气如何", "今天天气不错", RETRIEVED)
    assert any("幻觉" in w for w in result["warnings"])


def test_citation_present_no_hallucination_warning():
    """有 [1] 引用 → 不触发幻觉警告"""
    result = ComplianceChecker().check("天气如何", "答案内容[1]", RETRIEVED)
    assert not any("幻觉" in w for w in result["warnings"])


def test_sensitive_word_appends_disclaimer():
    """问题含敏感词（剂量）→ 强制附加免责声明"""
    result = ComplianceChecker().check("服用剂量是多少", "每次一片[1]", RETRIEVED)
    assert result["answer"].endswith(DISCLAIMER)


def test_final_answer_contains_sources():
    """最终输出必须包含引用来源清单（可追溯性）"""
    result = ComplianceChecker().check("成分", "阿司匹林[1]", RETRIEVED)
    assert "【引用来源】" in result["final_answer"]
    assert "阿司匹林肠溶片说明书.pdf" in result["final_answer"]