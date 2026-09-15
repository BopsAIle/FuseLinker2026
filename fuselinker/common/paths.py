"""Đường dẫn gốc fuselinker/, dataset và checkpoint (ổn định theo CWD)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"


def resolve_data_dir(data: str) -> str:
    """
    Nhận tên dataset (vd hetionet) hoặc đường dẫn.
    Thứ tự: path tuyệt đối / tồn tại theo CWD → fuselinker/<data>.
    """
    p = Path(data)
    if p.is_absolute():
        return str(p)
    if p.exists():
        return str(p.resolve())
    alt = ROOT / data
    if alt.exists():
        return str(alt.resolve())
    return str(alt)


def resolve_path(path: str) -> str:
    """
    Resolve file/dir tương đối về dưới fuselinker/ (ROOT), không phụ thuộc CWD.

    - Path tuyệt đối: giữ nguyên.
    - Path tương đối đã tồn tại dưới CWD: dùng CWD (tiện đọc file cũ).
    - Còn lại: ROOT / path (chỗ ghi checkpoint mặc định).
    """
    p = Path(path)
    if p.is_absolute():
        return str(p)
    cwd_p = Path.cwd() / p
    root_p = ROOT / p
    if cwd_p.exists():
        return str(cwd_p.resolve())
    if root_p.exists():
        return str(root_p.resolve())
    return str(root_p.resolve())


def ensure_parent(path: str) -> str:
    """Tạo thư mục cha rồi trả lại path (str) để torch.save / ghi file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def default_checkpoint(
    dataset: str,
    embedding: str,
    variant: str,
    filename: str = "model_state.pth",
) -> str:
    """
    checkpoints/<dataset>/<embedding>/<variant>/<filename>

    variant ví dụ: model | modelbase | model_base2 | ssl_base | ssl_upgrade
    """
    return str(CHECKPOINTS_DIR / dataset / embedding / variant / filename)
