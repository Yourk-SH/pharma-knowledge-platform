"""核心逻辑单元测试：纯算法与规则测试，不访问网络、不调用 API"""
import pytest

from compliance import ComplianceChecker
from retrieval import SimpleBM25, cosine_similarity_py, rrf_fusion


# ===== RRF 融合 =====
def test_rrf_双路第一优先():
    """两个通道都排第一的文档，融合后必须第一"""
    assert rrf_fusion([[1, 2, 3], [1, 3, 2]])[0] == 1


def test_rrf_双路入围胜过单路第一():
    """两个通道都中游的文档，胜过只在一个通道登顶的文档"""
    result = rrf_fusion([[10, 5], [10, 7], [9]])
    assert result[0] == 10


# ===== BM25 =====
def test_bm25_相关文档排最前():
    corpus = [
        ["二甲双胍", "禁忌", "肾功能", "不全"],
        ["阿司匹林", "解热", "镇痛"],
        ["二甲双胍", "用法", "用量"],
    ]
    ids = SimpleBM25(corpus).rank(["二甲双胍", "禁忌"], top_k=3)
    assert ids[0] == 0


def test_bm25_无匹配返回空():
    assert SimpleBM25([["二甲双胍"]]).rank(["股票", "涨停"], top_k=3) == []


# ===== 余弦相似度 =====
def test_cosine_相同向量为1():
    assert abs(cosine_similarity_py([1, 2, 3], [1, 2, 3]) - 1.0) < 1e-9


def test_cosine_正交向量为0():
    assert abs(cosine_similarity_py([1, 0], [0, 1])) < 1e-9


def test_cosine_零向量防御():
    assert cosine_similarity_py([0, 0], [1, 1]) == 0.0


# ===== 合规审查 =====
class TestCompliance:
    def setup_method(self):
        self.checker = ComplianceChecker()
        self.retrieved = [{"content": "测试内容", "source": "二甲双胍缓释片说明书.pdf", "score": 0.9}]

    def test_无引用标注触发幻觉警告(self):
        result = self.checker.check("问题", "这个回答没有任何引用标注", self.retrieved)
        assert any("引用" in w for w in result["warnings"])

    def test_敏感词附加免责声明(self):
        result = self.checker.check("问题", "用法用量如下[1]", self.retrieved)
        assert "合规提示" in result["final_answer"]

    def test_带引用的普通回答无警告(self):
        result = self.checker.check("问题", "该药的性状为白色片剂[1]", self.retrieved)
        assert result["warnings"] == []

    def test_来源清单包含文档名(self):
        result = self.checker.check("问题", "回答内容[1]", self.retrieved)
        assert "二甲双胍缓释片说明书.pdf" in result["final_answer"]