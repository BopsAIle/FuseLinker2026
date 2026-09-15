"""Đưa thư mục nhiệm vụ + common lên sys.path (chạy được từ mọi CWD)."""
from pathlib import Path
import sys

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parent
COMMON_DIR = ROOT / "common"

# Task trước (model.py / model_base4 của nhiệm vụ), rồi common (paths, myutils, ...).
for _p in (TASK_DIR, COMMON_DIR):
    _s = str(_p)
    if _s in sys.path:
        sys.path.remove(_s)
    sys.path.insert(0, _s)
