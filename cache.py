"""三级缓存：L1 Redis 热缓存 → L2 MySQL FAQ 库（相似度≥0.85）→ 未命中走 RAG（v2：懒加载，可测试）"""
import hashlib
import json
import time

import numpy as np

import config
from knowledge import get_embeddings
from logger import logger
from retrieval import cosine_similarity_np

FAQ_VECTOR_PATH = config.DATA_DIR / "faq_vectors.npy"
FAQ_VECTOR_KEY_PATH = config.DATA_DIR / "faq_vectors.key"

DISCLAIMER = "\n\n⚠️ 合规提示：本内容由AI基于文献生成，仅供专业人士参考，不构成用药建议。"

# ===== FAQ 种子数据（人工审核制，上线前需对照说明书逐条核对）=====
FAQ_SEED = [
    {
        "question": "二甲双胍哪些人不能用",
        "answer": "以下人群禁用或不推荐使用二甲双胍：\n"
                  "1. 急性或慢性代谢性酸中毒（含糖尿病酮症酸中毒）患者禁用；\n"
                  "2. 严重肾功能衰竭（eGFR<45）者禁用；\n"
                  "3. 对二甲双胍过敏者禁用；\n"
                  "4. 孕妇不推荐使用；\n"
                  "5. 哺乳期妇女应停止哺乳或停止用药。\n\n"
                  "【引用来源】二甲双胍缓释片说明书.pdf" + DISCLAIMER,
    },
    {
        "question": "感冒吃什么药好",
        "answer": "知识库中未检索到足够相关的信息，无法回答，请核实后重新提问。（合规要求：宁缺毋错）",
    },
]


def _qkey(question: str) -> str:
    return "pharma:faq:" + hashlib.md5(question.strip().encode("utf-8")).hexdigest()


class RedisLayer:
    """L1：近期问题精确匹配热缓存；Redis 不可用时自动降级为内存缓存"""

    def __init__(self):
        self._mem = {}
        self._client = None
        try:
            import redis
            self._client = redis.from_url(config.REDIS_URL, socket_connect_timeout=2, decode_responses=True)
            self._client.ping()
            logger.info("L1 缓存：Redis 已连接")
        except Exception:
            self._client = None
            logger.warning("L1 缓存：Redis 不可用，降级为内存缓存（进程重启后失效）")

    def get(self, question: str):
        key = _qkey(question)
        if self._client:
            raw = self._client.get(key)
            return json.loads(raw) if raw else None
        item = self._mem.get(key)
        if item and item[1] > time.time():
            return item[0]
        return None

    def set(self, question: str, answer: str):
        key = _qkey(question)
        payload = json.dumps({"answer": answer}, ensure_ascii=False)
        if self._client:
            self._client.setex(key, config.REDIS_TTL, payload)
        else:
            self._mem[key] = ({"answer": answer}, time.time() + config.REDIS_TTL)


class FAQStore:
    """L2：FAQ 库（MySQL 持久化，不可用时回退内置列表），向量相似度匹配"""

    def __init__(self):
        self.faq = self._load_faq()
        self._vecs = self._ensure_vectors()
        logger.info(f"L2 FAQ 库就绪：{len(self.faq)} 条")

    def _load_faq(self):
        try:
            import pymysql
            conn = pymysql.connect(
                host=config.MYSQL_HOST, port=config.MYSQL_PORT,
                user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
                charset="utf8mb4",
            )
            with conn.cursor() as cur:
                cur.execute(
                    f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DB} DEFAULT CHARSET utf8mb4"
                )
            conn.commit()
            conn.close()

            conn = pymysql.connect(
                host=config.MYSQL_HOST, port=config.MYSQL_PORT,
                user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
                database=config.MYSQL_DB, charset="utf8mb4",
            )
            with conn.cursor() as cur:
                cur.execute(
                    "CREATE TABLE IF NOT EXISTS faq ("
                    "id INT AUTO_INCREMENT PRIMARY KEY, "
                    "question VARCHAR(255) NOT NULL, "
                    "answer TEXT NOT NULL, "
                    "UNIQUE KEY uq_question (question))"
                )
                for f in FAQ_SEED:
                    cur.execute(
                        "INSERT IGNORE INTO faq (question, answer) VALUES (%s, %s)",
                        (f["question"], f["answer"]),
                    )
                conn.commit()
                cur.execute("SELECT question, answer FROM faq")
                rows = cur.fetchall()
            conn.close()
            logger.info("L2 FAQ 库：MySQL 已连接")
            return [{"question": q, "answer": a} for q, a in rows]
        except Exception as e:
            logger.warning(f"L2 FAQ 库：MySQL 不可用（{e}），使用内置 FAQ")
            return list(FAQ_SEED)

    def _ensure_vectors(self):
        # 指纹校验：问题列表变了（改/删/换序）就重建，防止旧向量配新答案
        fingerprint = hashlib.md5(
            "|".join(f["question"] for f in self.faq).encode("utf-8")
        ).hexdigest()
        need_build = True
        if FAQ_VECTOR_PATH.exists() and FAQ_VECTOR_KEY_PATH.exists():
            saved = FAQ_VECTOR_KEY_PATH.read_text(encoding="utf-8").strip()
            if saved == fingerprint and len(np.load(FAQ_VECTOR_PATH)) == len(self.faq):
                need_build = False
        if need_build:
            vecs = np.array(
                get_embeddings().embed_documents([f["question"] for f in self.faq]),
                dtype=np.float32,
            )
            np.save(FAQ_VECTOR_PATH, vecs)
            FAQ_VECTOR_KEY_PATH.write_text(fingerprint, encoding="utf-8")
        return np.load(FAQ_VECTOR_PATH)

    def match(self, query: str):
        """返回 (答案, 相似度)；未达阈值返回 (None, 相似度)"""
        qv = np.array([get_embeddings().embed_query(query)], dtype=np.float32)
        sims = cosine_similarity_np(qv, self._vecs)[0]
        idx = int(np.argmax(sims))
        best = float(sims[idx])
        if best >= config.FAQ_SIM_THRESHOLD:
            return self.faq[idx]["answer"], best
        return None, best


# ===== 懒加载单例：导入不再触发 MySQL/embedding 调用，单元测试可安全 import =====
_redis_layer = None
_faq_store = None


def get_redis_layer() -> RedisLayer:
    global _redis_layer
    if _redis_layer is None:
        _redis_layer = RedisLayer()
    return _redis_layer


def get_faq_store() -> FAQStore:
    global _faq_store
    if _faq_store is None:
        _faq_store = FAQStore()
    return _faq_store


def warmup_cache():
    """服务启动时主动预热（保持原有的启动日志时机）"""
    get_redis_layer()
    get_faq_store()


def cache_lookup(question: str):
    """三级查询：命中返回 (答案, 来源标识)，未命中返回 (None, None)"""
    hit = get_redis_layer().get(question)
    if hit:
        return hit["answer"], "redis"
    answer, sim = get_faq_store().match(question)
    if answer:
        get_redis_layer().set(question, answer)
        return answer, f"faq:{sim:.4f}"
    return None, None


def cache_writeback(question: str, answer: str):
    """RAG 回答回写 L1：近期重复问题直接复用"""
    get_redis_layer().set(question, answer)
