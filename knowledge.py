"""知识库模块：文档解析、父子块切分、向量索引（v2：支持 PDF + 上下文前缀注入）"""
import json

import numpy as np
import pymupdf
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# ===== 内置示例知识库（data/ 目录为空时兜底） =====
PHARMA_KB = [
    {"source": "示例-二甲双胍缓释片说明书 §禁忌",
     "content": "严重肾功能不全（eGFR<30）、急性代谢性酸中毒、对本品过敏者禁用二甲双胍。"},
    {"source": "示例-GxP合规手册 §数据完整性",
     "content": "所有临床数据修改必须留痕，记录修改人、修改时间和修改原因，原始数据不可覆盖删除，即ALCOA+原则。"},
]

PARENTS_PATH = config.DATA_DIR / "kb_parents.json"
CHILDREN_PATH = config.DATA_DIR / "kb_children.json"
VECTOR_PATH = config.DATA_DIR / "kb_child_vectors.npy"

_embeddings = None

# 父块：300 字保留上下文；子块：80 字精准匹配
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=30,
    separators=["\n\n", "\n", "。", "；", "！", "？", " "],
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=80, chunk_overlap=0,
    separators=["。", "；", "，", "、", " "],
)


def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=config.EMBED_MODEL,
            openai_api_key=config.API_KEY,
            openai_api_base=config.BASE_URL,
            check_embedding_ctx_length=False,
        )
    return _embeddings


def extract_pdf_text(path) -> str:
    """PyMuPDF 提取 PDF 文本"""
    doc = pymupdf.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(p.strip() for p in pages if p.strip())


def _add_document(parents: list, children: list, text: str, source: str):
    """一个文档 → 父块 → 子块；子块的向量化文本注入文档名前缀"""
    doc_name = source.replace(".pdf", "").replace(".txt", "")
    for pchunk in parent_splitter.split_text(text):
        if len(pchunk.strip()) < 20:
            continue
        pid = len(parents)
        parents.append({"id": pid, "content": pchunk.strip(), "source": source})
        for cchunk in child_splitter.split_text(pchunk):
            cchunk = cchunk.strip()
            if len(cchunk) >= 10:
                children.append({
                    "content": cchunk,
                    "embed_text": f"出自《{doc_name}》：{cchunk}",
                    "parent_id": pid,
                })


def load_kb() -> tuple[list, list]:
    """加载全部文档：PDF + txt，无文档时回退内置示例"""
    parents, children = [], []
    config.DATA_DIR.mkdir(exist_ok=True)

    for pdf in sorted(config.DATA_DIR.glob("*.pdf")):
        text = extract_pdf_text(pdf)
        print(f"  📄 {pdf.name}: 提取 {len(text)} 字")
        if len(text) < 100:
            print(f"     ⚠️ 提取字数过少，可能是扫描版 PDF（需要 OCR）")
        _add_document(parents, children, text, pdf.name)

    for txt in sorted(config.DATA_DIR.glob("*.txt")):
        _add_document(parents, children, txt.read_text(encoding="utf-8"), txt.name)

    if not parents:
        for entry in PHARMA_KB:
            pid = len(parents)
            parents.append({"id": pid, "content": entry["content"], "source": entry["source"]})
            children.append({
                "content": entry["content"],
                "embed_text": f"出自《{entry['source']}》：{entry['content']}",
                "parent_id": pid,
            })
    return parents, children


def build_index() -> tuple[int, int]:
    """构建索引：向量建在带前缀的子块文本上，持久化到磁盘"""
    parents, children = load_kb()
    vecs = get_embeddings().embed_documents([c["embed_text"] for c in children])
    np.save(VECTOR_PATH, np.array(vecs, dtype=np.float32))
    PARENTS_PATH.write_text(json.dumps(parents, ensure_ascii=False), encoding="utf-8")
    CHILDREN_PATH.write_text(json.dumps(children, ensure_ascii=False), encoding="utf-8")
    return len(parents), len(children)


def load_index() -> tuple[list, list, np.ndarray]:
    """加载索引；不存在则自动构建"""
    if not (PARENTS_PATH.exists() and CHILDREN_PATH.exists() and VECTOR_PATH.exists()):
        build_index()
    parents = json.loads(PARENTS_PATH.read_text(encoding="utf-8"))
    children = json.loads(CHILDREN_PATH.read_text(encoding="utf-8"))
    vecs = np.load(VECTOR_PATH)
    return parents, children, vecs