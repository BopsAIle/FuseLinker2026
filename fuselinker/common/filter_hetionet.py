"""
Lọc Hetionet cho drug-repurposing (dùng chung cho cả hai nhiệm vụ):
  - Giữ entity: Gene, Compound, Disease  (prefix trước '::')
  - Giữ 13 quan hệ metaedge giữa các entity đó
  - Đọc train.tsv / valid.tsv / test.tsv, ghi đè bản đã lọc vào cùng thư mục

Ví dụ:
  cd fuselinker
  python filter_hetionet.py --data_dir hetionet
  python filter_hetionet.py --data_dir hetionet --backup
  python filter_hetionet.py --data_dir path/to/raw_splits --out_dir hetionet
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

# 13 metaedge giữ lại (khớp relation2index.pkl hiện có của project).
DEFAULT_ALLOWED_RELS = frozenset({
    "Gr>G",  # Gene regulates Gene
    "GcG",   # Gene covaries Gene
    "GiG",   # Gene interacts Gene
    "CdG",   # Compound down-regulates Gene
    "CuG",   # Compound up-regulates Gene
    "CbG",   # Compound binds Gene
    "CrC",   # Compound resembles Compound
    "CtD",   # Compound treats Disease
    "CpD",   # Compound palliates Disease
    "DaG",   # Disease associates Gene
    "DdG",   # Disease down-regulates Gene
    "DuG",   # Disease up-regulates Gene
    "DrD",   # Disease resembles Disease
})

ALLOWED_ENTITY_TYPES = frozenset({"Gene", "Compound", "Disease"})
SPLIT_NAMES = ("train", "valid", "test")


def entity_type(name: str) -> str:
    if not isinstance(name, str):
        return ""
    if "::" in name:
        return name.split("::", 1)[0]
    return ""


def load_allowed_rels(data_dir: Path) -> frozenset[str]:
    """Ưu tiên đọc relation2index.pkl nếu có; không thì dùng DEFAULT_ALLOWED_RELS."""
    pkl = data_dir / "relation2index.pkl"
    if not pkl.is_file():
        return DEFAULT_ALLOWED_RELS
    with open(pkl, "rb") as f:
        relation2index = pickle.load(f)
    # key dạng ('head', 'GiG', 'tail') hoặc string 'GiG'
    rels = set()
    for key in relation2index.keys():
        if isinstance(key, tuple) and len(key) >= 2:
            rels.add(str(key[1]))
        else:
            rels.add(str(key))
    if not rels:
        return DEFAULT_ALLOWED_RELS
    print(f"[info] Loaded {len(rels)} allowed relations from {pkl.name}")
    return frozenset(rels)


def read_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, engine="python")
    if df.shape[1] < 3:
        raise ValueError(f"{path} must have >= 3 columns (head, relation, tail), got {df.shape[1]}")
    df = df.iloc[:, :3].copy()
    df.columns = ["head", "relation", "tail"]
    df["head"] = df["head"].astype(str).str.strip()
    df["relation"] = df["relation"].astype(str).str.strip()
    df["tail"] = df["tail"].astype(str).str.strip()
    return df


def filter_triples(df: pd.DataFrame, allowed_rels: frozenset[str]) -> pd.DataFrame:
    head_ok = df["head"].map(entity_type).isin(ALLOWED_ENTITY_TYPES)
    tail_ok = df["tail"].map(entity_type).isin(ALLOWED_ENTITY_TYPES)
    rel_ok = df["relation"].isin(allowed_rels)
    keep = head_ok & tail_ok & rel_ok
    out = df.loc[keep].drop_duplicates(subset=["head", "relation", "tail"]).reset_index(drop=True)
    return out


def summarize(name: str, df: pd.DataFrame) -> None:
    ents = pd.concat([df["head"], df["tail"]], ignore_index=True)
    type_counts = Counter(entity_type(x) for x in ents.unique())
    rel_counts = Counter(df["relation"].tolist())
    print(f"  [{name}] triples={len(df):,}  unique_entities={ents.nunique():,}")
    print(f"         entity_types={dict(sorted(type_counts.items()))}")
    print(f"         relations={len(rel_counts)}  top={rel_counts.most_common(5)}")


def backup_file(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if bak.exists():
        print(f"  [skip backup] already exists: {bak.name}")
        return
    shutil.copy2(path, bak)
    print(f"  [backup] {path.name} -> {bak.name}")


def main() -> None:
    # Chay: python common/filter_hetionet.py --data_dir hetionet
    import sys
    _common = Path(__file__).resolve().parent
    if str(_common) not in sys.path:
        sys.path.insert(0, str(_common))
    from paths import resolve_data_dir, resolve_path

    parser = argparse.ArgumentParser(
        description="Filter Hetionet train/valid/test to Gene/Compound/Disease + 13 relations."
    )
    parser.add_argument(
        "--data_dir",
        default="hetionet",
        help="Thu muc chua train/valid/test.tsv (ten dataset hoac path)",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Thu muc ghi file da loc (mac dinh = data_dir)",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Sao luu *.tsv thanh *.tsv.bak truoc khi ghi de (chi khi out_dir == data_dir)",
    )
    args = parser.parse_args()

    data_dir = Path(resolve_data_dir(args.data_dir))
    out_dir = Path(resolve_path(args.out_dir)) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = [s for s in SPLIT_NAMES if not (data_dir / f"{s}.tsv").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {data_dir.resolve()}: "
            + ", ".join(f"{s}.tsv" for s in missing)
            + "\nPlace unfiltered (or raw) train/valid/test TSV there, then re-run."
        )

    allowed_rels = load_allowed_rels(data_dir)
    print(f"Allowed entity types: {sorted(ALLOWED_ENTITY_TYPES)}")
    print(f"Allowed relations ({len(allowed_rels)}): {sorted(allowed_rels)}")
    print(f"Input : {data_dir.resolve()}")
    print(f"Output: {out_dir.resolve()}")

    filtered = {}
    for split in SPLIT_NAMES:
        src = data_dir / f"{split}.tsv"
        df = read_split(src)
        print(f"\nBefore filter — {split}.tsv")
        summarize(split, df)

        if args.backup and out_dir.resolve() == data_dir.resolve():
            backup_file(src)

        df_f = filter_triples(df, allowed_rels)
        print(f"After filter  — {split}.tsv")
        summarize(split, df_f)
        filtered[split] = df_f

    # Thống kê tổng sau lọc (giống paper: ~19930 nodes, 13 rels).
    all_df = pd.concat(filtered.values(), ignore_index=True)
    all_ents = pd.concat([all_df["head"], all_df["tail"]], ignore_index=True)
    print("\n=== TOTAL (train+valid+test, filtered) ===")
    print(f"  triples (with dup across splits ok): {len(all_df):,}")
    print(f"  unique triples: {all_df.drop_duplicates().shape[0]:,}")
    print(f"  unique entities: {all_ents.nunique():,}")
    print(f"  entity types: {dict(Counter(entity_type(x) for x in all_ents.unique()))}")
    print(f"  unique relations: {all_df['relation'].nunique()}")

    for split, df_f in filtered.items():
        dst = out_dir / f"{split}.tsv"
        df_f.to_csv(dst, sep="\t", header=False, index=False)
        print(f"[wrote] {dst}  ({len(df_f):,} triples)")

    print("\nDone. Co the chay task_degree_edge/train_upgrade.py --data", out_dir.as_posix())
    print("Nhac: embedding .npy phai co shape[0] == so entity sau khi Data() map id.")


if __name__ == "__main__":
    main()
