```markdown
PharmaDoc Agent —— 医药文献合规问答平台

![CI](https://github.com/Yourk-SH/pharma-knowledge-platform/actions/workflows/ci.yml/badge.svg)

基于 LangGraph + RAG 的医药文献智能问答系统，面向 GxP 合规场景设计：**答案必须带引用溯源，知识库外的问题宁可拒答也不编造**。

---

## ✨ 核心特性

| 特性 | 实现方式 |
|---|---|
| 🤖 Agent 流水线 | LangGraph 五节点编排：改写 → 检索 → 决策路由 → 生成 → 合规审查（含拒答分支） |
| 🔍 混合检索 + 精排 | BM25 + 向量双路召回 → RRF 融合 → bge-reranker 交叉编码器精排 → 父子块回溯 |
| 💬 多轮对话 | 对话历史注入 + LLM 指代消解（few-shot 加固）："阿莫西林呢" → "阿莫西林的主要成分有哪些" |
| ⚡ SSE 流式输出 | token 级逐字返回，改写/检索进度事件实时推送 |
| 🗄️ 三级缓存 | L1 Redis 热缓存 → L2 MySQL FAQ 语义匹配（余弦相似度 ≥ 0.85）→ L3 RAG 兜底；Redis/MySQL 双层降级保护 |
| 🛡️ 合规护栏 | 引用标注校验（防幻觉）+ 敏感词免责声明 + rerank 阈值拒答（宁缺毋错） |
| 👍 反馈闭环 | 点赞/点踩 → badcases.jsonl 落盘 → 脚本回流回归评测集（数据飞轮） |
| 📊 可观测性 | loguru 结构化日志 + 请求级 request_id 追踪 + 全链路节点耗时 trace |
| ✅ 质量保障 | pytest 24 例单测（合规规则 / 缓存降级 / API 契约 / 检索核心）+ GitHub Actions CI |
| 🐳 容器化 | docker-compose 三容器编排（redis / api / web），数据卷挂载 + 密钥 env_file 隔离 |

---

## 🏗️ 系统架构

~~~mermaid
graph TD
    U["用户"] --> W["Streamlit Web"]
    W -->|"SSE /ask_stream"| A["FastAPI"]
    A --> C{"三级缓存"}
    C -->|"L1 精确命中"| R["Redis"]
    C -->|"L2 语义命中"| M["MySQL FAQ 库"]
    C -->|"未命中"| G["LangGraph 流水线"]
    G --> RW["rewrite 多轮指代消解"]
    RW --> RT["retrieve BM25+向量+RRF+rerank"]
    RT -->|"Top1 分数达标"| GE["generate 流式生成"]
    RT -->|"分数过低"| RJ["reject 合规拒答"]
    GE --> CP["compliance 引用校验+免责声明"]
    CP --> U
    RJ --> U
~~~

---
```
## 🚀 快速开始

### Docker 一键部署（推荐）


bash

cp .env.example .env        # 填入 SILICON_API_KEY 与本机 MySQL 密码
docker compose up -d --build
```
| 服务 | 地址 |
|---|---|
| 前端界面 | http://127.0.0.1:8501 |
| API 文档（Swagger） | http://127.0.0.1:8000/docs |
| Redis | 127.0.0.1:6379 |

> L2 FAQ 库依赖宿主机 MySQL（容器经 `host.docker.internal` 访问）；MySQL 不可用时自动降级为内置 FAQ，服务不中断。

### 本地开发模式

```
bash

pip install -r requirements.txt

python api.py                 # 后端 :8000

streamlit run app.py          # 前端 :8501
```
---

## 🧪 测试与 CI

```
bash

pytest tests/ -v                      # 24 例单测，零 LLM 额度消耗

python scripts/collect_badcases.py    # 点踩样本回流回归评测集

python loadtest.py cache              # 缓存路径压测（QPS / P50 / P95）

python loadtest.py full               # 全链路压测
```
GitHub Actions 在每次 push 自动执行：单元测试 → Docker 镜像构建。

---

## 📈 评测结果

| 指标 | 得分 |
|---|---|
| 答案准确率 | 95.5% |
| 引用溯源正确率 | 100% |
| 域外问题拒答率 | 100% |

---
```

## 📁 项目结构

```

├── agents.py           # LangGraph 五节点流水线（多轮改写 + token 流式生成）
├── api.py              # FastAPI：/ask、/ask_stream(SSE)、/feedback、/health
├── app.py              # Streamlit 前端（流式渲染 + 多轮历史 + 👍👎 反馈）
├── cache.py            # 三级缓存（懒加载单例设计，单元测试友好）
├── knowledge.py        # PDF 解析 + 父子块切分 + 向量索引持久化
├── retrieval.py        # BM25 + 向量双路召回 + RRF 融合 + 父块回溯
├── rerank.py           # bge-reranker-v2-m3 交叉编码器精排
├── compliance.py       # GxP 合规护栏（引用校验 / 敏感词 / 来源清单）
├── logger.py           # 结构化日志 + request_id 请求追踪
├── loadtest.py         # 压测脚本（缓存 / 全链路双场景）
├── tests/              # pytest 单测 24 例
├── scripts/            # badcase 回流脚本
├── eval/               # 回归评测集（点踩样本自动回流）
├── data/               # 知识库 PDF 与向量索引（不入库）
├── logs/               # 运行日志与 badcases.jsonl（不入库）
└── docker-compose.yml  # 三容器编排（redis / api / web）
```
---

## 💡 关键设计决策

**为什么 217 条向量不用向量数据库？**
numpy 暴力检索在该规模是亚毫秒级，真正的瓶颈在 embedding API 的网络往返，引入 Milvus 是负优化。经测算扩展拐点在约 10 万向量（1000+ 份文档）：届时向量内存超 GB 级、暴力检索破百毫秒，再引入 HNSW 索引（Milvus/Qdrant）+ 增量索引 + 元数据过滤。

**为什么缓存键用"改写后的问题"？**
多轮追问的省略句（"阿莫西林呢"）语义不完整，直接做键会导致漏命中与跨会话误命中；改写为独立问题后命中率与正确性同时提升。

**流式输出与合规审查如何共存？**
正文逐 token 流出保证体验，引用来源清单与免责声明由合规节点计算完成后以末尾 SSE 事件补发——审查不因流式而缺位。

**医药场景的取舍：宁缺毋错。**
rerank Top1 低于阈值一律拒答，系统不因为用户语气随意或问题模糊而编造答案。

---

## 🛣️ 后续规划

- [ ] critique 反思自检环（引用-内容一致性校验，不合格自动重写）
- [ ] 会话持久化（MySQL 存储对话历史，刷新不丢失）
- [ ] Prometheus 指标 + Grafana 看板
- [ ] 文档量破百后评估向量数据库迁移与增量索引
```



## 推送

```powershell
cd D:\heima\PythonProject\pharma-knowledge-platform
git add README.md
git commit -m "docs: README v3 补全流式/多轮/测试/CI 特性与设计决策"
git push
```
