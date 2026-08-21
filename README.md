

# 💊 PharmaDoc Agent · 医药文献合规问答平台

基于 **LangGraph + 混合检索 + 合规护栏** 的医药知识问答系统：
回答必带引用溯源，知识域外诚实拒答，用药类问题自动附加免责声明。

> 演示动图（录制后放这里）
> ![demo](docs/demo.gif)

## ✨ 核心特性

| 能力 | 实现 |
|---|---|
| 🤖 Agent 流水线 | LangGraph 五节点：改写 → 检索 → 条件路由 → 生成 → 合规审查 |
| 🔍 混合检索 | 手写 BM25 + 稠密向量双路召回，RRF 融合，bge-reranker 精排 |
| 📄 父子块切分 | 子块（80字）检索、父块（300字）生成，向量化注入文档名前缀 |
| 🛡️ 三级合规护栏 | 引用标注校验（防幻觉）/ 敏感词免责声明 / 来源溯源清单 |
| 🚫 诚实拒答 | rerank 分数低于阈值自动路由至拒答节点（宁缺毋错） |
| 📊 量化评测 | 32 条自建评测集，检索与拒答质量可回归验证 |
| 🧪 单元测试 | pytest 覆盖核心算法与合规规则，11 条全绿 |

## 🏗️ 系统架构

```
mermaid
graph LR
    A["用户提问"] --> B["Streamlit 前端"]
    B -->|"HTTP"| C["FastAPI 服务层"]
    C --> D["LangGraph Agent"]
    D --> E["Query 改写"]
    E --> F["混合检索"]
    F --> G{"rerank 分数 ≥ 阈值?"}
    G -->|"否"| H["拒答节点"]
    G -->|"是"| I["带引用生成"]
    I --> J["合规审查"]
    J --> K["返回: 回答+来源+警告"]
    F --> L["BM25 + 向量双路 Top-20"]
    L --> M["RRF 融合 Top-10"]
    M --> N["reranker 精排 Top-3"]
```
## 📊 评测结果（32 条评测集）

| 指标 | 数值 | 说明 |
|---|---|---|
| 来源命中率（Top-3） | **100%** | 正确文档进入候选上下文 |
| Top-1 准确率 | **95.5%** | 唯一误排为说明书本身缺信息的歧义样本 |
| 域外拒答准确率 | **100%** | 10 条域外问题全部正确拒答 |

优化记录：评测发现跨文档同构章节混淆（"用法用量"章节在多个药品间误排），
采用**文档上下文前缀注入**（向量化 + 重排双通道），Top-1 准确率由 90.9% 提升至 95.5%。

## 🚀 快速开始

```
bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置密钥（新建 .env）
echo SILICON_API_KEY=你的硅基流动API_KEY > .env

# 3. 放入医药文档（PDF/txt）到 data/ 目录

# 4. 启动后端（首次运行自动构建向量索引）
python api.py

# 5. 启动前端（新终端）
streamlit run app.py
```
打开 http://127.0.0.1:8501 开始问答；API 文档见 http://127.0.0.1:8000/docs。

## 🗂️ 项目结构

```

├── app.py           # Streamlit 前端（HTTP 调后端，前后端分离）
├── api.py           # FastAPI 服务层（/ask /health /knowledge）
├── agents.py        # LangGraph 五节点流水线 + 条件路由
├── retrieval.py     # 手写 BM25 / 余弦 / RRF + 混合检索器
├── rerank.py        # bge-reranker-v2-m3 交叉编码器精排
├── knowledge.py     # PDF 解析 + 父子块切分 + 索引持久化
├── compliance.py    # 三级合规护栏
├── evaluation.py    # 32 条评测集 + 三指标量化
├── config.py        # 配置中心
├── tests/           # pytest 单元测试（11 条）
├── data/            # 医药文档（5 份真实药品说明书 PDF）
└── requirements.txt
```
## 🛠️ 技术栈

LangGraph · LangChain · FastAPI · Streamlit · DeepSeek-V3 ·
BGE-large-zh · bge-reranker-v2-m3 · PyMuPDF · jieba · numpy · pytest

## 🔜 后续规划

- [ ] FAQ 高频问题缓存层（阈值调优曲线）
- [ ] 扫描版 PDF 的 OCR 接入
- [ ] 多轮对话上下文压缩
- [ ] Docker 容器化部署

## ⚠️ 免责声明

本项目为个人学习项目，知识库内容为公开药品说明书，仅用于技术演示。
AI 生成内容不构成任何用药建议，实际用药请遵医嘱。
```

### 项目至此全部完工 🎉

```
✅ 六层模块化架构     ✅ 真实医药文档（5份说明书）
✅ 手写 BM25/RRF/余弦  ✅ 父子块 + 上下文前缀注入
✅ reranker 完整漏斗   ✅ 三级合规护栏 + 拒答路由
✅ 32 条评测集（100/95.5/100）  ✅ 11 条单测全绿
✅ FastAPI + Streamlit 前后端分离  ✅ README + 评测报告
```