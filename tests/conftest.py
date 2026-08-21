"""pytest 路径配置：保证能 import 项目根目录的模块"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))