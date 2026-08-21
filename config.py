import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"

# ===== 路径配置 =====
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# ===== 模型配置 =====
API_KEY = os.getenv("SILICON_API_KEY")
BASE_URL = "https://api.siliconflow.cn/v1"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"

# ===== 检索参数 =====
RETRIEVE_TOP_K = 2
SIM_THRESHOLD = 0.4
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

CANDIDATE_TOP_K = 10
RERANK_TOP_N = 3
RERANK_SCORE_THRESHOLD = 0.3
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# ===== 合规参数 =====
SENSITIVE_WORDS = ["服用", "剂量", "用药", "停药", "禁忌", "用量", "用法"]

# ===== 缓存配置 =====
FAQ_SIM_THRESHOLD = 0.85
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_TTL = 7 * 24 * 3600

MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "123456")
MYSQL_DB = os.getenv("MYSQL_DB", "pharma_kb")