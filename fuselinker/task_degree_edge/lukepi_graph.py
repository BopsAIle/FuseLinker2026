"""
lukepi_graph.py - Sửa đổi đồ thị kiểu LukePi (edge masking + negative) cho FuseLinker.

Khác với generate_sampled_graph_and_labels (task_node_mask DistMult):
  - Mask một tỷ lệ cạnh (mask_rate), XÓA chúng khỏi đồ thị message-passing.
  - Positive = cạnh bị mask, nhãn = relation id thật.
  - Negative = cặp (src, dst) không tồn tại trong batch, nhãn = num_rels (lớp fake).
  - GNN encode trên đồ thị còn lại; head edge chỉ predict trên mask + negative.

API trả về tương thích pipeline model_base4 / main3:
  g, uniq_v, rel, norm, samples, edge_labels
  - samples: [N, 3] (src, rel, dst) đã relabel theo node subgraph
  - edge_labels: [N] long, lớp trong [0, num_rels] (num_rels = fake)
"""
import numpy as np

from myutils import (
    sample_edge_uniform,
    sample_edge_neighborhood,
    build_graph_from_triples,
)


def _edge_key_set(src, dst):
    """Tập cạnh vô hướng theo hướng (src, dst) để lọc negative."""
    return set(zip(src.tolist(), dst.tolist()))

## num_neg: số lượng negative edges cần lấy 
# Sinh ra các cạnh ngẫu nhiên bằng cách chọn ngẫu nhiên 2 node và nối với nhau
def _sample_negative_edges(num_nodes, edge_set, num_neg, rng):
    """
    Sample cặp (src, dst) không nằm trong edge_set (giống Negativesub LukePi).
    Trả về arrays src_neg, dst_neg độ dài ~num_neg.
    """
    src_neg = []
    dst_neg = []
    # Thử nhiều lần hơn target để tránh kẹt khi đồ thị dày.
    max_trials = max(num_neg * 20, num_neg + 100)
    trials = 0
    seen = set()  # Lưu các cạnh negative đã sinh ra 
    while len(src_neg) < num_neg and trials < max_trials:
        trials += 1
        s = int(rng.randint(0, num_nodes)) #Lấy ngẫu nhiên node nguồn
        d = int(rng.randint(0, num_nodes)) # Lấy ngẫu nhiên node đích
        if s == d:
            continue
        key = (s, d)
        if key in edge_set or key in seen:
            continue
        seen.add(key)
        src_neg.append(s)
        dst_neg.append(d)

    return (
        np.asarray(src_neg, dtype=np.int64),
        np.asarray(dst_neg, dtype=np.int64),
    )
## Trả về 2 mảng . Mảng 1 là chứa node nguồn src_neg ; Mảng 2 là chứa node đích dst_neg

def generate_lukepi_masked_graph_and_labels(
    triplets,
    sample_size,
    mask_rate,
    num_rels,
    adj_list,
    degrees,
    negative_rate=1,
    sampler="uniform",
    rng=None,
):
    """
    Lấy mẫu subgraph rồi mask cạnh kiểu LukePi.

    Args:
        triplets: np.ndarray [E, 3] (src, rel, dst) toàn train.
        sample_size: số cạnh lấy vào batch (graph_batch_size).
        mask_rate: tỷ lệ cạnh bị mask (vd 0.2) — phần này dùng cho loss_edge,
                   phần còn lại build đồ thị encode.
        num_rels: số loại quan hệ thật (lớp fake = num_rels).
        adj_list, degrees: dùng khi sampler='neighbor'.
        negative_rate: số negative / số positive (masked). Mặc định 1:1.
        sampler: 'uniform' | 'neighbor'.
        rng: np.random.RandomState (tuỳ chọn).

    Returns:
        g: DGLGraph — chỉ chứa cạnh KHÔNG bị mask (đã thêm reverse trong build).
        uniq_v: np.ndarray — map local id -> original entity id.
        rel: np.ndarray — relation id trên cạnh của g (kể cả reverse).
        norm: np.ndarray — node norm.
        samples: np.ndarray [N, 3] — (src, rel, dst) local id; neg có rel=0 placeholder.
        edge_labels: np.ndarray [N] int64 — nhãn đa lớp [0, num_rels].
    """
    if rng is None:
        rng = np.random

    if not (0.0 < mask_rate < 1.0):
        raise ValueError(f"mask_rate must be in (0, 1), got {mask_rate}")

    sample_size = min(int(sample_size), len(triplets))
    if sampler == "uniform":
        edge_ids = sample_edge_uniform(len(triplets), sample_size)
    elif sampler == "neighbor":
        edge_ids = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    edges = triplets[edge_ids]
    src, rel, dst = edges.transpose()

    # Relabel node về [0, n_sub)
    uniq_v, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
    n_half = len(src)
    src = inv[:n_half].astype(np.int64)
    dst = inv[n_half:].astype(np.int64)
    rel = rel.astype(np.int64)
    num_sub_nodes = len(uniq_v)

    # ---- Mask edges (LukePi) ----
    n_mask = max(1, int(sample_size * mask_rate))
    n_mask = min(n_mask, sample_size - 1)  # giữ ít nhất 1 cạnh cho message passing
    perm = rng.permutation(sample_size)
    mask_ids = perm[:n_mask] ## Tập các cạnh bị mask
    keep_ids = perm[n_mask:] # Tập các cạnh không bị mask

    # Positive: cạnh bị mask — nhãn = relation thật
    pos_src = src[mask_ids]
    pos_dst = dst[mask_ids]
    pos_rel = rel[mask_ids]
    pos_labels = pos_rel.copy()

    # Đồ thị encode: chỉ cạnh còn lại
    keep_src = src[keep_ids]
    keep_dst = dst[keep_ids]
    keep_rel = rel[keep_ids]
    g, graph_rel, norm = build_graph_from_triples(
        num_sub_nodes, num_rels, (keep_src, keep_rel, keep_dst)
    )

    # ---- Negative edges ----
    # Tập cạnh đã biết trong batch (cả mask lẫn keep) để tránh sample trùng cạnh thật.
    known = _edge_key_set(src, dst)
    # Thêm hướng ngược để negative không trùng cạnh 2 chiều đã có trong KG batch.
    known |= set((d, s) for s, d in known)

    n_neg = int(n_mask * negative_rate)
    neg_src, neg_dst = _sample_negative_edges(num_sub_nodes, known, n_neg, rng)
    # rel placeholder (không dùng khi nhãn = fake); vẫn điền 0 cho shape [N,3]
    neg_rel = np.zeros(len(neg_src), dtype=np.int64)
    neg_labels = np.full(len(neg_src), num_rels, dtype=np.int64)

    samples = np.stack(
        [
            np.concatenate([pos_src, neg_src]),
            np.concatenate([pos_rel, neg_rel]),
            np.concatenate([pos_dst, neg_dst]),
        ],
        axis=1,
    ).astype(np.int64)
    edge_labels = np.concatenate([pos_labels, neg_labels]).astype(np.int64)

    return g, uniq_v, graph_rel, norm, samples, edge_labels
