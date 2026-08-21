深度思考
```Push successful! The code is now on GitHub at https://github.com/Yourk-SH/pharma-knowledge-platform.

I should check if I can view the repo to verify it looks good and give README feedback. Let me try fetching the GitHub page. Actually, I can use WebFetch to look at the repo and verify the files are visible and that there are no secrets. Let's give it a try.

After that: next steps — build with Docker.

Let me fetch the repo page.
```
🎉 **推送成功！代码已在 GitHub 上线。** 我看一眼仓库页面。


**已获取网页内容** *https://github.com/Yourk-SH/pharma-knowledge-platform*
深度思考
```The repo is up and running. But the README is old — it was written before Stations 1–3 (logging/tracing, three-tier cache, feedback loop), and it lists as "后续规划 (future plans)" the things that have actually already been implemented (FAQ cache, Docker). Things missing from the README: logging/tracing, three-tier cache (Redis+MySQL), feedback loop, Docker deploy instructions, .env.example instructions. Need to deliver an updated README v2.

Also, the project structure section is outdated (cache.py, logger.py, loadtest.py, Dockerfile, docker-compose.yml, scripts.py are missing).

Note that "演示动图（录制后放这里）(demo GIF (to be placed here after recording))" is still a placeholder.

So, the next action: deliver an updated README.md (full version), plus a .env.example file, and instruct them to commit and push the update. This aligns with my promise to "polish the README + add .env.example."

Let me draft the updated README:

Sections to keep: title/intro, core features (adding three-tier cache, link tracing, feedback loop), architecture diagram (updated to include the cache layer), evaluation results, quick start (including docker compose deploy + .env.example), project structure (updated), tech stack (add redis/pymysql/loguru/docker), future plans (remove completed items, list remaining ones: streaming output, reflection loop, multi-turn, Milvus at scale), disclaimers.

Also add a "Deployment" section: docker compose up. Add a "Runbook / Engineering Practices" section listing observability, caching, feedback loop, CI plans, etc.

Keep the README reasonably concise. Also .env.example:

```
# SiliconFlow API key (https://cloud.siliconflow.cn)
SILICON_API_KEY=sk-your-key-here
# MySQL (L2 FAQ library; if not provided, falls back to local JSON)
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DB=pharma_kb
# Redis (if not provided, falls back to in-memory cache)
REDIS_URL=redis://127.0.0.1:6379/0
```

After that: git add README.md .env.example → commit → push.

And next: Docker build (WSL ready). Keep pushing forward on both tracks.
```
## ✅ 仓库上线成功！但 README 还是旧版——现在升级它

我看了仓库：18 个文件都推上去了，**但 README 还停留在三个站之前**——"后续规划"里写着"FAQ 缓存、Docker 部署"，而这些你已经做完了；新增的日志追踪、三级缓存、反馈闭环一个字没提。**README 是面试官的第一眼，必须同步。**

### 1️⃣ README.md 整个替换为 v2

```markdown
# 💊 PharmaDoc Agent · 医药文献合规问答平台

基于 LangGraph + 混合检索 + 三级缓存 + 合规护栏 的医药知识问答系统：
回答必带引用溯源，知识域外诚实拒答，高频问题缓存秒回，全链路可观测。

> 演示动图（录制后放这里）

## ✨ 核心特性

| 能力 | 实现 |
| --- | --- |
| 🤖 Agent 流水线 | LangGraph 节点编排：改写 → 检索 → 条件路由 → 生成 → 合规审查，节点级耗时追踪 |
| 🔍 混合检索 | 手写 BM25 + 稠密向量双路召回，RRF 融合，bge-reranker-v2-m3 精排 |
| 📄 父子块切分 | 子块（80字）检索、父块（300字）生成，向量化注入文档名前缀消除跨文档混淆 |
| ⚡ 三级缓存 | Redis 热缓存（精确匹配，TTL 7天）→ MySQL FAQ 库（语义匹配 ≥0.85）→ RAG 兜底；高频问题 6.5s → 0.6s |
| 🛡️ 合规护栏 | 引用标注校验（防幻觉）/ 敏感词免责声明 / 来源溯源清单 / 低分拒答路由（宁缺毋错） |
| 📊 量化评测 | 32 条自建评测集：来源命中率 100% / Top-1 95.5% / 域外拒答 100% |
| 👎 反馈闭环 | 前端点赞点踩 → badcase 落盘 → 回流评测集，数据飞轮驱动迭代 |
| 🔭 可观测性 | loguru 双通道日志 + 请求 ID 全链路追踪 + 节点耗时 trace |
| 🧪 质量保障 | pytest 11 条单测全绿（曾抓出敏感词表真实漏洞） |

## 🏗️ 系统架构

```
mermaid
graph LR
A["用户提问"] --> B["Streamlit 前端"]
B -->|HTTP| C["FastAPI 服务层"]
C --> D{"三级缓存"}
D -->|L1 命中| R1["Redis 热缓存"]
D -->|L2 命中| R2["MySQL FAQ 库"]
D -->|未命中| E["LangGraph Agent"]
E --> F["混合检索: BM25+向量 → RRF → reranker"]
F --> G{"rerank ≥ 阈值?"}
G -->|否| H["拒答节点"]
G -->|是| I["带引用生成"]
I --> J["合规审查"]
J --> K["回答+来源+警告"]
K -.回写.-> R1
```
## 📊 评测结果（32 条评测集）

| 指标 | 数值 | 说明 |
| --- | --- | --- |
| 来源命中率（Top-3） | 100% | 正确文档进入候选上下文 |
| Top-1 准确率 | 95.5% | 唯一误排为说明书本身缺信息的歧义样本 |
| 域外拒答准确率 | 100% | 10 条域外问题全部正确拒答 |

**优化记录**：评测发现跨文档同构章节混淆（"用法用量"在多药品间误排），
采用文档上下文前缀注入（向量化 + 重排双通道），Top-1 由 90.9% → 95.5%。

## 🚀 快速开始

### 方式一：Docker 一键部署（推荐）

```
bash
# 配置密钥：复制模板并填入自己的 Key
cp .env.example .env

# 构建并启动（redis + api + web 三容器）
docker compose up -d --build

# 访问
# 前端: http://127.0.0.1:8501
# API 文档: http://127.0.0.1:8000/docs
```
### 方式二：本地运行

```
bash
pip install -r requirements.txt
cp .env.example .env   # 填入硅基流动 API Key（MySQL/Redis 缺失时自动降级）
# 将医药文档（PDF/txt）放入 data/ 目录
python api.py          # 后端（首次运行自动构建索引）
streamlit run app.py   # 前端（新终端）
```
> 药品说明书 PDF 出于版权原因未包含在仓库中，需自行放入 `data/` 目录。

## 🗂️ 项目结构

```

├── app.py            # Streamlit 前端（点赞点踩反馈入口）
├── api.py            # FastAPI 服务层（/ask /feedback /health /knowledge）
├── agents.py         # LangGraph 流水线 + 拒答条件路由 + 节点计时
├── retrieval.py      # 手写 BM25 / 余弦 / RRF + 混合检索器
├── rerank.py         # bge-reranker-v2-m3 交叉编码器精排
├── knowledge.py      # PDF 解析 + 父子块切分 + 索引持久化
├── cache.py          # 三级缓存：Redis 热缓存 + MySQL FAQ 库（自动降级）
├── compliance.py     # 合规护栏（引用校验/敏感词/来源清单）
├── evaluation.py     # 32 条评测集 + 三指标量化
├── loadtest.py       # 并发压测（缓存/全链路双场景）
├── logger.py         # loguru 双通道日志 + 请求 ID 追踪
├── config.py         # 配置中心（锚定项目根目录加载 .env）
├── tests/            # pytest 单元测试（11 条）
├── Dockerfile        # 容器镜像定义
├── docker-compose.yml# 三容器编排（redis/api/web）
└── requirements.txt
```
## 🛠️ 技术栈

LangGraph · LangChain · FastAPI · Streamlit · DeepSeek-V3 · BGE-large-zh ·
bge-reranker-v2-m3 · Redis · MySQL · Docker · PyMuPDF · jieba · loguru · pytest

## 🔜 后续规划

- 流式输出（SSE），降低体感延迟
- 生成节点反思自检环（critique 节点）
- 多轮对话记忆（session 级上下文）
- 文档扩至百份级后评估引入向量数据库

## ⚠️ 免责声明

本项目为个人学习项目，知识库内容为公开药品说明书，仅用于技术演示。
AI 生成内容不构成任何用药建议，实际用药请遵医嘱。
```


### 2️⃣ 新建 `.env.example`（密钥模板，安全进仓库）

```text
# 硅基流动 API Key（必填，https://cloud.siliconflow.cn 获取）
SILICON_API_KEY=sk-your-key-here

# MySQL（L2 FAQ 库，缺失时自动降级为内置 FAQ）
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-mysql-password
MYSQL_DB=pharma_kb

# Redis（L1 热缓存，缺失时自动降级为内存缓存）
REDIS_URL=redis://127.0.0.1:6379/0
```


### 3️⃣ 提交推送

```powershell
git add README.md .env.example
git commit -m "README v2：补充三级缓存/可观测性/反馈闭环/Docker部署说明"
git push
```


---

README 推完，马上回来干正事：**WSL 好了 → 打开 Docker Desktop → `docker version` 看到 Server 段 → `docker compose up -d --build`**。第 4 站就差这最后一把火了。