"""Badcase 回流：从 logs/badcases.jsonl 提取点踩样本 → 回归评测集
用法：python scripts/collect_badcases.py"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
SRC = BASE_DIR / "logs" / "badcases.jsonl"
DST = BASE_DIR / "eval" / "badcases_regression.txt"


def main():
    if not SRC.exists():
        print("还没有 badcases.jsonl，先去前端点几个 👎")
        return

    seen = set()
    questions = []
    for line in SRC.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("rating") != "down":
            continue
        q = record.get("question", "").strip()
        if q and q not in seen:
            seen.add(q)
            questions.append(q)

    DST.parent.mkdir(exist_ok=True)
    DST.write_text("\n".join(questions), encoding="utf-8")
    print(f"已提取 {len(questions)} 条点踩样本 → {DST}")


if __name__ == "__main__":
    main()