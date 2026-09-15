"""
inspect_dataset.py
------------------
Phan tich mot thu muc dataset dang KG y sinh (cau truc giong hetionet/)
va xuat ra cac thong tin can thiet de viet mo ta dataset trong bai bao.

Vi du su dung:
    python inspect_dataset.py --folder_path D:/Project/fuseLinker/FuseLinker/fuselinker/Adint

Script doc:
  - <folder>/train.tsv, valid.tsv, test.tsv   (moi file: head \t relation \t tail)
  - Cac file numpy *.npy trong folder          (text + domain embeddings)
  - Cac file *.pkl / *.csv mapping             (entity2index, relation2index, ...)
  - Cac file *.txt / *.md co the chua mo ta dataset

Va in/luu bao cao tong hop: so luong node, quan he, edge, phan bo,
mo ta embedding di kem, ...
"""

import argparse
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_read_tsv(path):
    """Doc file TSV 3 cot (head, relation, tail), bo qua loi neu co."""
    try:
        df = pd.read_csv(path, sep="\t", header=None, dtype=str,
                         on_bad_lines="skip", engine="python")
        if df.shape[1] < 3:
            return None
        df = df.iloc[:, :3]
        df.columns = ["head", "relation", "tail"]
        return df
    except Exception as e:
        print(f"[WARN] Khong doc duoc {path}: {e}")
        return None


def detect_entity_type(entity_id: str):
    """Thu doan loai thuc the tu tien to (prefix:: hoac pattern UMLS / DB / hsa / HSA: / GO: ...)."""
    if not isinstance(entity_id, str):
        return "<non-string>"
    if "::" in entity_id:
        # Dang 'Type::id' (vi du: Compound::DB00495, Gene::2831)
        return entity_id.split("::", 1)[0]
    # Cac pattern thuong gap
    patterns = [
        (r"^[A-Z]\d{6}_", "UMLS"),       # C0003250_aapp
        (r"^DC\d+_", "UMLS_DRUG"),
        (r"^hsa\d+", "KEGG_PATHWAY"),
        (r"^HSA:", "KEGG_GENE"),
        (r"^D\d+", "KEGG_DRUG"),
        (r"^H\d+", "KEGG_DISEASE"),
        (r"^N\d+", "KEGG_NETWORK"),
        (r"^GO:\d+", "GO"),
        (r"^UBERON:", "UBERON"),
        (r"^DB\d+", "DRUGBANK"),
    ]
    for pat, name in patterns:
        if re.match(pat, entity_id):
            return name
    # UMLS cui cung (C...)
    if re.match(r"^C\d{7}$", entity_id):
        return "UMLS_RAW"
    return "OTHER"


def summarize_text_embeddings(folder: Path):
    """Tim cac file .npy co the la text embedding, in shape + dtype."""
    rows = []
    for f in sorted(folder.glob("*.npy")):
        name_lower = f.name.lower()
        # Heuristic: ten co chua 'embedding' hoac la public bert/pubmed/llama/flan/t5
        is_text_emb = any(k in name_lower for k in [
            "embedding", "bert", "pubmed", "llama", "flan", "t5", "medllama", "pmcllama"
        ])
        is_domain_emb = "poincare" in name_lower or "domain" in name_lower or "ontology" in name_lower
        kind = "unknown"
        if is_domain_emb and not is_text_emb:
            kind = "domain (Poincare/ontology?)"
        elif is_text_emb:
            kind = "text (LLM encoder?)"
        try:
            arr = np.load(f, mmap_mode="r")
            shape = tuple(arr.shape)
            dtype = str(arr.dtype)
            size_mb = f.stat().st_size / (1024 * 1024)
        except Exception as e:
            shape = ("<unreadable>",)
            dtype = str(e)
            size_mb = f.stat().st_size / (1024 * 1024)
        rows.append({
            "file": f.name,
            "shape": shape,
            "dtype": dtype,
            "size_MB": round(size_mb, 2),
            "guess_kind": kind,
        })
    return rows


def try_load_pickle(path):
    """Thu load pickle an toan."""
    import pickle
    try:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    except Exception as e:
        return f"<error: {e}>"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Inspect a biomedical KG dataset folder.")
    parser.add_argument("--folder_path", required=True,
                        help="Duong dan toi folder dataset (vd: .../fuselinker/Adint)")
    parser.add_argument("--out", default=None,
                        help="(optional) Luu bao cao Markdown ra file nay")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="So luong mau in ra cho moi muc (default: 5)")
    args = parser.parse_args()

    folder = Path(args.folder_path)
    if not folder.is_dir():
        raise SystemExit(f"[ERROR] Folder khong ton tai: {folder}")

    print("=" * 78)
    print(f"INSPECT DATASET FOLDER: {folder}")
    print("=" * 78)

    # 1) Liet ke toan bo file trong folder ------------------------------------
    print("\n[1] Cau truc folder:")
    for p in sorted(folder.iterdir()):
        if p.is_file():
            size = p.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.2f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.2f} KB"
            else:
                size_str = f"{size} B"
            print(f"   - {p.name:50s}  {size_str}")
        elif p.is_dir():
            print(f"   + {p.name}/   (subdir)")

    # 2) Doc README/Mo ta neu co ---------------------------------------------
    print("\n[2] Mo ta dataset (neu co file .md / .txt / README):")
    for cand in ["README.md", "README.txt", "readme.md", "description.txt", "DESCRIPTION.md"]:
        f = folder / cand
        if f.exists():
            print(f"   --- {f.name} ---")
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                print(content[:2000] + ("\n   ...(truncated)..." if len(content) > 2000 else ""))
            except Exception as e:
                print(f"   [WARN] Khong doc duoc {f}: {e}")

    # 3) Doc cac file TSV -----------------------------------------------------
    print("\n[3] Thong ke cac file TSV:")
    splits = {}
    for split in ["train", "valid", "test"]:
        for ext in [".tsv", ".txt"]:
            p = folder / f"{split}{ext}"
            if p.exists():
                df = safe_read_tsv(p)
                if df is not None:
                    splits[split] = df
                    print(f"   {split}{ext}: {len(df):,} rows  "
                          f"[cols={list(df.columns)}]")
                    print(f"      Vi du 3 dong dau:")
                    for _, row in df.head(3).iterrows():
                        print(f"         {row['head']!r:35s} | {row['relation']!r:25s} | {row['tail']!r}")
                    break

    if not splits:
        print("   [WARN] Khong tim thay train/valid/test (.tsv hoac .txt)")
        return

    train_df = splits.get("train")
    valid_df = splits.get("valid")
    test_df = splits.get("test")
    graph_df = pd.concat([d for d in [train_df, valid_df, test_df] if d is not None],
                         ignore_index=True)

    # 4) So luong node, relation, edge ---------------------------------------
    print("\n[4] Thong ke tong quan ( tren graph = train+valid+test ):")
    num_edges = len(graph_df)
    all_heads = set(graph_df["head"].dropna().unique())
    all_tails = set(graph_df["tail"].dropna().unique())
    all_entities = all_heads | all_tails
    all_relations = set(graph_df["relation"].dropna().unique())
    print(f"   Tong so canh (edges)        : {num_edges:,}")
    print(f"   Tong so thuc the (entities) : {len(all_entities):,}")
    print(f"   Tong so quan he (relations) : {len(all_relations):,}")

    for split_name, df in splits.items():
        e = set(df["head"].dropna().unique()) | set(df["tail"].dropna().unique())
        r = set(df["relation"].dropna().unique())
        print(f"      - {split_name:5s}: {len(df):>9,} edges | "
              f"{len(e):>7,} entities | {len(r):>3,} relations")

    # 5) Phan bo quan he ------------------------------------------------------
    print("\n[5] Phan bo quan he (top 15 theo so canh):")
    rel_counter = Counter(graph_df["relation"].dropna())
    for rel, cnt in rel_counter.most_common(15):
        print(f"   {cnt:>10,}  {rel}")
    if len(rel_counter) > 15:
        print(f"   ... va {len(rel_counter) - 15} quan he khac.")

    # 6) Phan loai thuc the ---------------------------------------------------
    print("\n[6] Phan loai thuc the (top 15 theo so luong):")
    type_counter = Counter(detect_entity_type(e) for e in all_entities)
    for t, cnt in type_counter.most_common(15):
        print(f"   {cnt:>10,}  {t}")
    if len(type_counter) > 15:
        print(f"   ... va {len(type_counter) - 15} loai khac.")

    # 7) Entity overlap giua cac split ----------------------------------------
    if train_df is not None and valid_df is not None and test_df is not None:
        print("\n[7] Entity overlap giua cac split (transductive?):")
        tr_e = set(train_df["head"].dropna()) | set(train_df["tail"].dropna())
        va_e = set(valid_df["head"].dropna()) | set(valid_df["tail"].dropna())
        te_e = set(test_df["head"].dropna()) | set(test_df["tail"].dropna())
        print(f"   train entities: {len(tr_e):,}")
        print(f"   valid entities: {len(va_e):,}")
        print(f"   test  entities: {len(te_e):,}")
        print(f"   valid \\ train  (chi trong valid): {len(va_e - tr_e):,}")
        print(f"   test  \\ train  (chi trong test) : {len(te_e - tr_e):,}")
        print(f"   valid ∩ test  : {len(va_e & te_e):,}")

    # 8) Edge overlap (mot triple co xuat hien o nhieu split khong?) ----------
    print("\n[8] Kiem tra leakage: triple trung giua cac split:")
    def triple_set(df):
        return set(zip(df["head"], df["relation"], df["tail"]))
    if train_df is not None and valid_df is not None:
        tr_t = triple_set(train_df)
        va_t = triple_set(valid_df)
        print(f"   train ∩ valid : {len(tr_t & va_t):,} triples")
    if train_df is not None and test_df is not None:
        tr_t = triple_set(train_df)
        te_t = triple_set(test_df)
        print(f"   train ∩ test  : {len(tr_t & te_t):,} triples")
    if valid_df is not None and test_df is not None:
        va_t = triple_set(valid_df)
        te_t = triple_set(test_df)
        print(f"   valid ∩ test  : {len(va_t & te_t):,} triples")

    # 9) Mat do do thi (trung binh) -------------------------------------------
    print("\n[9] Mat do do thi:")
    avg_deg = (2 * num_edges) / max(len(all_entities), 1)
    print(f"   Bac trung binh (avg degree, vo huong): {avg_deg:.2f}")

    # 10) Kiem tra cac file pickle mapping ------------------------------------
    print("\n[10] Mapping files (.pkl / .csv):")
    for f in sorted(folder.glob("*2index.*")) + sorted(folder.glob("*2relation*")):
        print(f"   {f.name}: ", end="")
        if f.suffix == ".pkl":
            obj = try_load_pickle(f)
            if isinstance(obj, dict):
                print(f"dict voi {len(obj)} entries. Vi du 3 key dau: {list(obj.keys())[:3]}")
            elif isinstance(obj, list):
                print(f"list co {len(obj)} phan tu. Vi du 3 phan tu dau: {obj[:3]}")
            else:
                print(f"type={type(obj).__name__}")
        elif f.suffix == ".csv":
            try:
                df_map = pd.read_csv(f)
                print(f"csv shape={df_map.shape}, columns={list(df_map.columns)}")
                print(f"      Vi du 3 dong dau:\n{df_map.head(3).to_string(index=False)}")
            except Exception as e:
                print(f"[WARN] Khong doc duoc: {e}")

    # 11) Embedding files -----------------------------------------------------
    print("\n[11] Embedding files (.npy) di kem:")
    emb_rows = summarize_text_embeddings(folder)
    if emb_rows:
        # In bang don gian
        print(f"   {'file':45s} {'shape':25s} {'dtype':10s} {'size_MB':>8s}  guess_kind")
        for r in emb_rows:
            shape_str = "x".join(str(s) for s in r["shape"])
            print(f"   {r['file']:45s} {shape_str:25s} {r['dtype']:10s} "
                  f"{r['size_MB']:>8.2f}  {r['guess_kind']}")
    else:
        print("   (khong co file .npy nao)")

    # 12) Tong hop cho nguoi viet ---------------------------------------------
    print("\n[12] TOM TAT NHANH (de copy vao bai bao):")
    print(f"   - Ten dataset    : {folder.name}")
    print(f"   - So thuc the    : {len(all_entities):,}")
    print(f"   - So quan he     : {len(all_relations):,}")
    print(f"   - So canh (graph): {num_edges:,}")
    for split_name, df in splits.items():
        print(f"      + {split_name}: {len(df):,}")
    print(f"   - So loai entity : {len(type_counter)} "
          f"(top 3: {[t for t, _ in type_counter.most_common(3)]})")
    print(f"   - Embedding di kem:")
    for r in emb_rows:
        shape_str = "x".join(str(s) for s in r["shape"])
        print(f"      + {r['file']:40s} shape={shape_str:25s}  ({r['guess_kind']})")

    print("\n" + "=" * 78)
    print("HOAN TAT.")
    print("=" * 78)


if __name__ == "__main__":
    main()
