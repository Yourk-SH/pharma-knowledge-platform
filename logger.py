"""结构化日志模块：控制台 + 文件双通道，带请求 ID 追踪"""
import sys
from contextvars import ContextVar

from loguru import logger as _logger

# 每个请求独立的追踪 ID（线程安全）
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def _log_format(record) -> str:
    rid = request_id_var.get()
    ts = record["time"].strftime("%Y-%m-%d %H:%M:%S")
    return f"{ts} | {record['level'].name:<7} | [{rid}] | {record['message']}\n"


_logger.remove()
_logger.add(sys.stdout, format=_log_format, level="INFO")
_logger.add(
    "logs/app.log",
    format=_log_format,
    level="DEBUG",           # 文件里记录更详细的 DEBUG 信息
    rotation="10 MB",        # 每 10MB 自动轮转
    retention="7 days",
    encoding="utf-8",
)

logger = _logger


def set_request_id(rid: str) -> None:
    request_id_var.set(rid)