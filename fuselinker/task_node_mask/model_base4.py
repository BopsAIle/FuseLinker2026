"""
Task 1 (mask entity / DistMult) — encoder UPGRADE.

Bản viết lại: gộp những gì thực sự có ích từ model.py, model_base2.py và hai
bản model_base4.py (task_node_mask + task_degree_edge), bỏ code chết, và đưa
MỌI module đáng ngờ ra thành công tắc để chạy ablation.

Vì sao bản upgrade cũ thua base — 4 nguyên nhân tìm được trong code
--------------------------------------------------------------------
1. Lệch scale train/eval ở nhánh PPR. Train dùng neighbor sampling
   (fanout=50, dropout bật), eval dùng full graph trên CPU. `PPRGraphConv` cũ
   aggregate bằng SUM nên độ lớn h_ppr phụ thuộc số hàng xóm lấy được →
   embedding lúc test không cùng scale với lúc train. DistMult là tích ba
   chiều, cực nhạy với scale.
   → Sửa: `ppr_agg="mean"` (chuẩn hóa theo tổng trọng số PPR) + `ppr_fanout=-1`
     mặc định (Gppr đã cắt top-k nên lấy hết vẫn rẻ, và khớp đúng eval).

2. BatchNorm trong TextEmbeddingAutoencoder. Running-stats được ước lượng từ
   subgraph ~250 cạnh lúc train, rồi áp lên toàn bộ N node lúc eval. Bản
   task_degree_edge đã đổi sang LayerNorm vì đúng lý do này; bản task_node_mask
   thì chưa.
   → Sửa: `text_norm="layer"` mặc định (vẫn chọn lại được "batch").

3. Regularization phạt chính phần PPR cộng thêm. `regularization_loss` tính
   trên embedding ĐÃ fuse, mà fuse là `h_main + g*h_aux` nên mọi đóng góp của
   PPR đều làm tăng ||h||² → reg_param đẩy gate về 0. Cộng với init bias -3
   (gate ≈ 0.047 lúc đầu), nhánh PPR gần như chỉ có thể lỗ chứ không có lãi.
   → Sửa: `reg_on="rgcn"` mặc định — phạt nhánh backbone như FuseLinker gốc,
     không phạt phần bổ sung.

4. Decoder của autoencoder không bao giờ được dùng. `forward` chỉ lấy `encoded`,
   `decoded` bị vứt, không có reconstruction loss ở bất kỳ đâu → nửa số tham số
   của module đó là tham số chết (grad = 0). "Autoencoder" thực chất chỉ là một
   MLP encoder.
   → Sửa: `text_encoder="mlp"` mặc định (đúng phần đang thực sự chạy). Muốn
     autoencoder thật thì `--text_encoder ae --text_recon_weight 0.1`.

Giữ lại gì / bỏ gì
------------------
GIỮ  Dựng Gppr (PPR sparse + cache)      — tiền xử lý đúng, tốn kém, có cache.
GIỮ  min-max toàn ma trận cho 2 embedding — đặc trưng FuseLinker gốc.
GIỮ  Poincaré → Euclidean bằng Linear.
GIỮ  Fusion theo w cố định (mặc định)     — base đang thắng bằng đúng cái này.
GIỮ  RelGraphConv 'bdd' + num_bases.
GIỮ  DistMult scorer + relation_weights   — để checkpoint/eval cũ dùng lại được.
GIỮ  ResidualPPRFusion (gate init ~0)     — đúng về nguyên tắc "không tệ hơn base".
GIỮ  FusionGate (từ model_base2)          — nhưng chỉ khi bật cờ, không mặc định.
BỎ   MLPProjector/FusionGate treo lơ lửng không ai gọi (code chết ở bản cũ).
BỎ   Decoder mặc định (xem mục 4).
BỎ   Nhân đôi TextEmbeddingAutoencoder giữa 2 task — nay là 1 lớp có tham số.

Công tắc ablation
-----------------
Dùng `add_model_arguments(parser)` trong train script rồi
`EncoderConfig.from_args(args)`; hoặc truyền thẳng `config=EncoderConfig(...)`.
Mọi cờ đều có thể tắt về đúng hành vi model.py gốc:

    --use_ppr false --text_norm batch --text_encoder ae --reg_on fused
      → tương đương base (model.py) + decoder chết, để đối chứng.

Tương thích ngược: chữ ký `LinkPredict(...)` và `forward(...)` không đổi, nên
train_upgrade.py / eval_auroc.py hiện tại chạy được ngay.
"""
import argparse
import copy
import random
from dataclasses import dataclass, fields
from pathlib import Path

import dgl
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_global_seed(seed):
    """
    Cố định seed cho random/numpy/torch/dgl. Không có bước này thì hai run cùng
    cấu hình vẫn lệch nhau, và mọi kết luận ablation đều vô nghĩa.
    """
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)


# ================================================================
# Generating Auxiliary Network (PPR) theo mô tả ở đầu file.
# ================================================================
def _keep_topk(indices, values, topk):
    """
    Giữ top-k phần tử có giá trị lớn nhất.
    """
    if topk is None or topk <= 0 or values.shape[0] <= topk:
        return indices, values
    keep = np.argpartition(values, values.shape[0] - topk)[-topk:]
    keep = keep[np.argsort(values[keep])[::-1]]
    return indices[keep], values[keep]


def _compute_ppr_streaming_sparse(M, c, epsilon, num_iter, topk):
    """
    Tính PPR theo từng node, không tạo ma trận dense NxN.
    Mỗi vòng lặp đều prune theo epsilon/topk để khống chế bộ nhớ.
    """
    M = M.tocsr()
    n = M.shape[0]
    indptr = M.indptr
    indices = M.indices
    data = M.data

    rows = []
    cols = []
    vals = []
    max_state = (topk + 1) if (topk is not None and topk > 0) else None

    for center in range(n):
        # Khởi tạo state của random walk tại chính node center.
        state_idx = np.array([center], dtype=np.int64)
        state_val = np.array([1.0], dtype=np.float32)

        for _ in range(num_iter):
            acc = {}
            for src_node, src_weight in zip(state_idx, state_val):
                start = indptr[src_node]
                end = indptr[src_node + 1]
                neigh = indices[start:end]
                neigh_w = data[start:end]
                for dst_node, trans_w in zip(neigh, neigh_w):
                    key = int(dst_node)
                    acc[key] = acc.get(key, 0.0) + float(src_weight * trans_w)

            if not acc:
                acc = {center: 0.0}

            # s <- c * (sM) + (1-c) * e_center
            for key in list(acc.keys()):
                acc[key] = c * acc[key]
            acc[center] = acc.get(center, 0.0) + (1.0 - c)

            next_idx = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
            next_val = np.fromiter(acc.values(), dtype=np.float32, count=len(acc))

            keep_mask = (np.abs(next_val) >= epsilon) | (next_idx == center)
            next_idx = next_idx[keep_mask]
            next_val = next_val[keep_mask]

            if next_idx.shape[0] == 0:
                next_idx = np.array([center], dtype=np.int64)
                next_val = np.array([1.0 - c], dtype=np.float32)

            if max_state is not None and next_idx.shape[0] > max_state:
                self_mask = (next_idx == center)
                self_idx = next_idx[self_mask]
                self_val = next_val[self_mask]
                other_idx = next_idx[~self_mask]
                other_val = next_val[~self_mask]
                keep_other = max(max_state - self_idx.shape[0], 0)
                other_idx, other_val = _keep_topk(other_idx, other_val, keep_other)
                next_idx = np.concatenate([self_idx, other_idx]).astype(np.int64, copy=False)
                next_val = np.concatenate([self_val, other_val]).astype(np.float32, copy=False)

            state_idx = next_idx
            state_val = next_val

        final_mask = (state_idx != center) & (np.abs(state_val) >= epsilon)
        row_idx = state_idx[final_mask]
        row_val = state_val[final_mask]
        row_idx, row_val = _keep_topk(row_idx, row_val, topk)

        if row_idx.shape[0] > 0:
            rows.append(np.full(row_idx.shape[0], center, dtype=np.int64))
            cols.append(row_idx.astype(np.int64, copy=False))
            vals.append(row_val.astype(np.float32, copy=False))

    if len(vals) == 0:
        return sp.coo_matrix((n, n), dtype=np.float32)

    out_rows = np.concatenate(rows)
    out_cols = np.concatenate(cols)
    out_vals = np.concatenate(vals)
    return sp.coo_matrix((out_vals, (out_rows, out_cols)), shape=(n, n))


def _resolve_ppr_device(device):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif not isinstance(device, torch.device):
        device = torch.device(device)
    return device


def _auto_ppr_batch_size(num_nodes, num_edges, device, requested=None):
    if requested is not None and requested > 0:
        return int(requested)
    if device.type != "cuda":
        return 64
    try:
        free_bytes, _ = torch.cuda.mem_get_info(device)
    except Exception:
        free_bytes = 2 * 1024 ** 3
    # Message tensor [E, B] dominates. 4GB laptop GPUs need a small batch.
    bytes_per_col = 4.0 * (2.0 * max(int(num_edges), 1) + 4.0 * max(int(num_nodes), 1))
    batch = int(0.12 * free_bytes / max(bytes_per_col, 1.0))
    return max(4, min(32, batch))


def _transition_to_dgl(M, device):
    """Chuyển M scipy -> DGL graph; trọng số cạnh lưu trên graph (đi cùng permute format)."""
    M_coo = M.tocoo()
    # (M S)_i = sum_j M_ij S_j  => cạnh j -> i với weight M_ij.
    src = torch.from_numpy(np.asarray(M_coo.col, dtype=np.int64).copy())
    dst = torch.from_numpy(np.asarray(M_coo.row, dtype=np.int64).copy())
    w = torch.from_numpy(np.asarray(M_coo.data, dtype=np.float32).copy()).view(-1, 1)
    g = dgl.graph((src, dst), num_nodes=M.shape[0])
    g.edata["ppr_w"] = w
    g = g.to(device)
    if hasattr(g, "create_formats_"):
        g.create_formats_()
    return g


def _sparsify_ppr_batch(S, seeds, epsilon, topk):
    """
    S: [N, B] PPR của batch walker. Trả src/dst/weight (1D) sau khi bỏ self + epsilon + topk.
    """
    n, batch = S.shape
    col = torch.arange(batch, device=S.device)
    S[seeds, col] = 0.0
    S.masked_fill_(S.abs() < epsilon, 0.0)

    if topk is not None and topk > 0:
        k = min(int(topk), n)
        vals, dst = torch.topk(S, k=k, dim=0)  # [k, B]
        src = seeds.unsqueeze(0).expand(k, -1)
        keep = vals.abs() >= epsilon
        return src[keep], dst[keep], vals[keep]

    dst, local = torch.nonzero(S, as_tuple=True)
    return seeds[local], dst, S[dst, local]


def _compute_ppr_batched_dgl(M, c, epsilon, num_iter, topk, device, batch_size):
    """
    Power iteration GPU/CPU:
        S <- c * M S + (1 - c) E
    theo từng batch cột để không cấp phát dense NxN.
    """
    n = M.shape[0]
    g = _transition_to_dgl(M, device)
    batch_size = _auto_ppr_batch_size(n, g.number_of_edges(), device, batch_size)
    print(
        f"Computing PPR on {device} "
        f"(iterative, N={n}, E={g.number_of_edges()}, batch={batch_size}, iter={num_iter})."
    )

    all_src, all_dst, all_w = [], [], []
    n_batch = (n + batch_size - 1) // batch_size
    log_every = max(1, n_batch // 5)

    with torch.inference_mode():
        for b_idx, start in enumerate(range(0, n, batch_size)):
            end = min(start + batch_size, n)
            seeds = torch.arange(start, end, device=device, dtype=torch.int64)
            batch = seeds.shape[0]
            E = torch.zeros(n, batch, device=device, dtype=torch.float32)
            E[seeds, torch.arange(batch, device=device)] = 1.0
            S = E
            for _ in range(num_iter):
                with g.local_scope():
                    g.ndata["h"] = S
                    g.update_all(fn.u_mul_e("h", "ppr_w", "msg"), fn.sum("msg", "h_new"))
                    S = c * g.ndata["h_new"] + (1.0 - c) * E
            src, dst, w = _sparsify_ppr_batch(S, seeds, epsilon, topk)
            if src.numel() > 0:
                all_src.append(src.detach().cpu())
                all_dst.append(dst.detach().cpu())
                all_w.append(w.detach().cpu())
            if b_idx % log_every == 0 or end == n:
                print(f"  PPR batch {b_idx + 1}/{n_batch} (nodes {start}:{end}/{n})")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not all_w:
        return sp.coo_matrix((n, n), dtype=np.float32)

    rows = torch.cat(all_src).numpy().astype(np.int64, copy=False)
    cols = torch.cat(all_dst).numpy().astype(np.int64, copy=False)
    vals = torch.cat(all_w).numpy().astype(np.float32, copy=False)
    return sp.coo_matrix((vals, (rows, cols)), shape=(n, n))


def _compute_ppr_dense_gpu(M, c, epsilon, device):
    """Closed-form S = (1-c)(I - cM)^{-1} trên GPU (chỉ dùng khi N nhỏ)."""
    n = M.shape[0]
    print(f"Computing PPR on {device} (dense inverse, N={n}).")
    M_dense = torch.as_tensor(M.toarray(), dtype=torch.float32, device=device)
    I = torch.eye(n, device=device, dtype=torch.float32)
    S = (1.0 - c) * torch.linalg.inv(I - c * M_dense)
    S.fill_diagonal_(0.0)
    S.masked_fill_(S.abs() < epsilon, 0.0)
    src, dst = torch.nonzero(S, as_tuple=True)
    w = S[src, dst]
    return sp.coo_matrix(
        (
            w.detach().cpu().numpy().astype(np.float32, copy=False),
            (
                src.detach().cpu().numpy().astype(np.int64, copy=False),
                dst.detach().cpu().numpy().astype(np.int64, copy=False),
            ),
        ),
        shape=(n, n),
    )


def compute_ppr_sparse(num_nodes, train_triples, c=0.15, epsilon=1e-4,
                        add_self_loop=True, use_iterative=False, num_iter=50,
                        topk=50, device=None, batch_size=None):
    """
    Dựng auxiliary network Gppr theo công thức HGDC:
        S = (1 - c) (I - c M)^{-1},  M = D^{-1/2} A D^{-1/2}
    rồi cắt ngưỡng |S_ij| < epsilon để ra ma trận thưa ~S.

    Args:
        num_nodes: số node.
        train_triples: np.ndarray [E, 3] = (src, rel, dst). Ta chỉ dùng (src, dst).
        c: damping factor.
        epsilon: ngưỡng cắt các giá trị nhỏ trong S.
        add_self_loop: cộng self-loop vào A trước khi chuẩn hóa (ổn định hơn).
        use_iterative: True -> ép dùng iterative (không tạo dense NxN).
        num_iter: số lần lặp khi dùng iterative.
        topk: số cạnh tối đa giữ lại cho mỗi node sau khi sparse PPR.
        device: 'cuda' / 'cpu' / torch.device. None -> CUDA nếu có.
        batch_size: số walker / batch khi iterative trên GPU. None -> tự chọn.

    Returns:
        ppr_graph: dgl.DGLGraph (đồ thị thuần, không relation type).
        edge_weight: torch.FloatTensor [num_edges] - giá trị ~S_ij.
    """
    src = train_triples[:, 0].astype(np.int64)
    dst = train_triples[:, 2].astype(np.int64)

    # A đối xứng (không trọng số)
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(len(rows), dtype=np.float32)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    A.data[:] = 1.0  # loại trùng lặp cạnh -> unweighted
    A = A.maximum(A.T)
    if add_self_loop:
        A = A + sp.eye(num_nodes, format='csr')

    # D^{-1/2} A D^{-1/2}
    deg = np.asarray(A.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = sp.diags(deg_inv_sqrt)
    M = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()

    # Khi có top-k thì dùng iterative để tránh cấp phát dense NxN gây tràn RAM.
    # Nhánh dense chỉ giữ lại để tương thích khi cần chạy đúng công thức closed-form.
    use_streaming = use_iterative or (topk is not None and topk > 0)
    device = _resolve_ppr_device(device)
    S_sparse = None

    if device.type == "cuda":
        cur_bs = _auto_ppr_batch_size(num_nodes, M.nnz, device, batch_size)
        try:
            if use_streaming:
                while True:
                    try:
                        S_sparse = _compute_ppr_batched_dgl(
                            M, c, epsilon, num_iter, topk, device, cur_bs
                        )
                        break
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower() or cur_bs <= 8:
                            raise
                        next_bs = max(8, cur_bs // 2)
                        print(
                            f"PPR GPU OOM at batch={cur_bs}; retry with batch={next_bs}."
                        )
                        torch.cuda.empty_cache()
                        cur_bs = next_bs
            else:
                S_sparse = _compute_ppr_dense_gpu(M, c, epsilon, device)
        except Exception as exc:
            print(f"PPR GPU failed ({exc}); falling back to CPU.")
            torch.cuda.empty_cache()
            S_sparse = None

    if S_sparse is None:
        cpu_dev = torch.device("cpu")
        if use_streaming:
            try:
                S_sparse = _compute_ppr_batched_dgl(
                    M, c, epsilon, num_iter, topk, cpu_dev, batch_size or 64
                )
            except Exception as exc:
                print(f"PPR batched CPU failed ({exc}); falling back to sparse streaming.")
                S_sparse = _compute_ppr_streaming_sparse(M, c, epsilon, num_iter, topk)
        else:
            print("Computing PPR on CPU (dense inverse).")
            I = sp.eye(num_nodes, format='csr')
            S_dense = (1.0 - c) * np.linalg.inv((I - c * M).toarray())
            S_dense = S_dense.astype(np.float32)
            S_dense[np.abs(S_dense) < epsilon] = 0.0
            np.fill_diagonal(S_dense, 0.0)
            S_sparse = sp.coo_matrix(S_dense)
            del S_dense

    S_sparse = S_sparse.tocoo()
    src_t = torch.from_numpy(S_sparse.row.astype(np.int64))
    dst_t = torch.from_numpy(S_sparse.col.astype(np.int64))
    w_t = torch.from_numpy(S_sparse.data.astype(np.float32))

    ppr_graph = dgl.graph((src_t, dst_t), num_nodes=num_nodes)
    ppr_graph.edata["ppr_w"] = w_t.view(-1, 1)
    if hasattr(ppr_graph, "create_formats_"):
        ppr_graph.create_formats_()
    # Keep the auxiliary graph on CPU. Training samples a neighborhood onto GPU
    # each step so a 4GB card is not filled by the full Gppr.
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ppr_graph, w_t


def load_or_compute_ppr_sparse(num_nodes, train_triples, cache_dir=None, **kwargs):
    """
    compute_ppr_sparse + cache file under cache_dir so Hetionet PPR is not
    rebuilt every run (first build can take several minutes on a 4GB GPU).
    """
    c = kwargs.get("c", 0.15)
    epsilon = kwargs.get("epsilon", 1e-4)
    num_iter = kwargs.get("num_iter", 50)
    topk = kwargs.get("topk", 50)
    n_edges = int(train_triples.shape[0])
    fname = (
        f"ppr_N{num_nodes}_E{n_edges}_c{c}_eps{epsilon}_it{num_iter}_k{topk}.bin"
    )
    cache_path = Path(cache_dir) / fname if cache_dir else None
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached PPR: {cache_path}")
        graphs, extra = dgl.load_graphs(str(cache_path))
        g = graphs[0]
        w = extra["w"]
        if "ppr_w" not in g.edata:
            g.edata["ppr_w"] = w.view(-1, 1)
        return g, w

    g, w = compute_ppr_sparse(num_nodes, train_triples, **kwargs)
    g = g.cpu()
    w = w.cpu()
    if "ppr_w" not in g.edata:
        g.edata["ppr_w"] = w.view(-1, 1)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving PPR cache: {cache_path}")
        dgl.save_graphs(str(cache_path), [g], {"w": w})
    return g, w


# ================================================================
# Cấu hình ablation
# ================================================================
@dataclass
class EncoderConfig:
    """
    Mỗi trường = 1 module bật/tắt được. Giá trị mặc định là bản đã sửa 4 lỗi
    nêu ở docstring; chú thích ghi giá trị tái hiện hành vi cũ.
    """
    # ---- nguồn embedding đầu vào ----
    use_text: bool = True               # tắt để đo đóng góp của text embedding
    use_domain: bool = True             # tắt để đo đóng góp của Poincaré
    text_encoder: str = "mlp"           # mlp | ae | linear      (cũ: ae)
    text_norm: str = "layer"            # layer | batch | none   (cũ: batch)
    text_recon_weight: float = 0.0      # >0 mới thực sự dùng decoder của ae
    input_fusion: str = "fixed"         # fixed | gate           (cũ: fixed)
    input_norm: bool = False            # LayerNorm sau khi fuse
    node_id_embedding: bool = False     # residual id embedding (ý từ base2)

    # ---- backbone R-GCN ----
    rgcn_block: str = "plain"           # plain | residual       (cũ: plain)
    rgcn_regularizer: str = "bdd"       # bdd | basis | none

    # ---- nhánh phụ PPR ----
    use_ppr: bool = True
    ppr_num_layers: int = 2
    ppr_fanout: int = -1                # -1 = hết hàng xóm      (cũ: 50)
    ppr_agg: str = "mean"               # mean | sum             (cũ: sum)
    ppr_layer_norm: bool = True
    ppr_dropout: float = None           # None = theo --dropout
    ppr_fusion: str = "residual"        # residual | gate | scalar | replace
    ppr_fusion_init_bias: float = -3.0
    ppr_fusion_norm: bool = False       # LayerNorm sau fusion
    ppr_eval_device: str = "cpu"        # cpu | same

    # ---- loss ----
    reg_on: str = "rgcn"                # rgcn | fused | none    (cũ: fused)
    relation_minmax: bool = True        # min-max relation pretrained (như model.py)

    @classmethod
    def from_args(cls, args):
        kwargs = {}
        for f in fields(cls):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        return cls(**kwargs)

    def describe(self):
        return " | ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


def _bool_flag(x):
    return str(x).lower() in ("1", "true", "yes", "y", "t")


def add_model_arguments(parser):
    """
    Thêm toàn bộ công tắc ablation vào argparse của train script.
    Tên cờ trùng tên trường trong EncoderConfig để from_args() map thẳng.
    """
    g = parser.add_argument_group("ablation (encoder)")
    g.add_argument("--use_text", type=_bool_flag, default=True)
    g.add_argument("--use_domain", type=_bool_flag, default=True)
    g.add_argument("--text_encoder", choices=["mlp", "ae", "linear"], default="mlp")
    g.add_argument("--text_norm", choices=["layer", "batch", "none"], default="layer")
    g.add_argument("--text_recon_weight", type=float, default=0.0)
    g.add_argument("--input_fusion", choices=["fixed", "gate"], default="fixed")
    g.add_argument("--input_norm", type=_bool_flag, default=False)
    g.add_argument("--node_id_embedding", type=_bool_flag, default=False)
    g.add_argument("--rgcn_block", choices=["plain", "residual"], default="plain")
    g.add_argument("--rgcn_regularizer", choices=["bdd", "basis", "none"], default="bdd")

    p = parser.add_argument_group("ablation (PPR)")
    p.add_argument("--use_ppr", type=_bool_flag, default=True)
    p.add_argument("--ppr_num_layers", type=int, default=2)
    p.add_argument("--ppr_fanout", type=int, default=-1)
    p.add_argument("--ppr_agg", choices=["mean", "sum"], default="mean")
    p.add_argument("--ppr_layer_norm", type=_bool_flag, default=True)
    p.add_argument("--ppr_dropout", type=float, default=None)
    p.add_argument("--ppr_fusion",
                   choices=["residual", "gate", "scalar", "replace"],
                   default="residual")
    p.add_argument("--ppr_fusion_init_bias", type=float, default=-3.0)
    p.add_argument("--ppr_fusion_norm", type=_bool_flag, default=False)
    p.add_argument("--ppr_eval_device", choices=["cpu", "same"], default="cpu")

    l = parser.add_argument_group("ablation (loss)")
    l.add_argument("--reg_on", choices=["rgcn", "fused", "none"], default="rgcn")
    l.add_argument("--relation_minmax", type=_bool_flag, default=True)
    return parser


# ================================================================
# Khối dùng chung
# ================================================================
def _make_norm(kind, dim):
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class TextEncoder(nn.Module):
    """
    Thay cho TextEmbeddingAutoencoder cũ.

    kind="mlp"    : đúng phần encoder mà bản cũ thực sự chạy (decoder bị vứt).
    kind="ae"     : dựng thêm decoder; chỉ có ý nghĩa khi text_recon_weight > 0,
                    khi đó reconstruction loss mới được cộng vào get_loss().
    kind="linear" : 1 Linear thuần, để đo xem MLP 2 lớp có đáng không.
    """

    def __init__(self, input_dim, hidden_dim, kind="mlp", norm="layer", dropout=0.2):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.encoder = nn.Linear(input_dim, hidden_dim)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                _make_norm(norm, hidden_dim * 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                _make_norm(norm, hidden_dim),
            )
        self.decoder = None
        if kind == "ae":
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                _make_norm(norm, hidden_dim * 2),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, input_dim),
                nn.ReLU(True),
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded) if self.decoder is not None else None
        return encoded, decoded


class FusionGate(nn.Module):
    """
    Gated fusion (lấy từ model_base2). So với weighted average cố định, gate
    được tính từ [a, b, |a-b|, a*b] nên "biết" hai nguồn đang mâu thuẫn hay
    đồng thuận ở từng chiều. w (nếu có) làm prior mềm để không phá ý tưởng cũ.
    """

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, a, b, w=None):
        stats = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)
        gate = self.gate(stats)
        if w is not None:
            gate = 0.5 * gate + 0.5 * w
        return gate * a + (1.0 - gate) * b


class PPRGraphConv(nn.Module):
    """
    GCN có trọng số cạnh trên Gppr:

        agg="sum"  : h_v' = sum_u w_uv * (W h_u)                  (bản cũ)
        agg="mean" : h_v' = sum_u w_uv * (W h_u) / sum_u w_uv

    "mean" là điểm sửa quan trọng nhất: độ lớn đầu ra không còn phụ thuộc số
    hàng xóm lấy được, nên subgraph lúc train và full graph lúc eval cho cùng
    một thang đo — điều kiện bắt buộc để DistMult không lệch giữa train/test.
    """

    def __init__(self, in_dim, out_dim, activation=True, use_norm=True,
                 agg="mean", dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.act = nn.ReLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.agg = agg

    def forward(self, g, x, edge_weight):
        with g.local_scope():
            # srcdata/dstdata: dùng được cho cả full graph lẫn DGL block.
            g.srcdata["h"] = self.linear(x)
            g.edata["w"] = edge_weight.reshape(-1, 1)
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h_new"))
            h = g.dstdata["h_new"]
            if self.agg == "mean":
                g.update_all(fn.copy_e("w", "mw"), fn.sum("mw", "w_sum"))
                h = h / g.dstdata["w_sum"].clamp(min=1e-6)
        return self.dropout(self.act(self.norm(h)))


class PPRBranch(nn.Module):
    """
    Nhánh phụ trên Gppr, ăn cùng feature đầu vào với R-GCN (sau EmbeddingLayer)
    để hai không gian biểu diễn tương thích. Lớp cuối không ReLU.
    """

    def __init__(self, hidden_dim, num_layers=2, agg="mean",
                 use_norm=True, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            PPRGraphConv(
                hidden_dim, hidden_dim,
                activation=(i < num_layers - 1),
                use_norm=use_norm,
                agg=agg,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

    def forward(self, ppr_graph, x, edge_weight):
        h = x
        for layer in self.layers:
            h = layer(ppr_graph, h, edge_weight)
        return h


class PPRFusion(nn.Module):
    """
    Trộn backbone (h_main) với nhánh PPR (h_aux).

    mode="residual" : h_main + g * h_aux, g = sigmoid(fc([h_main, h_aux])),
                      fc khởi tạo 0 + bias âm nên lúc đầu g ~ 0, output ĐÚNG
                      BẰNG base. Lựa chọn an toàn: upgrade không thể tệ hơn
                      base ngay từ iteration đầu.
    mode="scalar"   : h_main + alpha * h_aux, alpha là 1 scalar học được init 0.
                      Ít tham số nhất và dễ đọc: alpha sau train cho biết ngay
                      model có thèm dùng PPR hay không.
    mode="gate"     : g*h_main + (1-g)*h_aux (kiểu FusionGate, init ~0.5 nên
                      pha loãng backbone ngay lập tức).
    mode="replace"  : chỉ dùng h_aux — để xem PPR một mình mạnh cỡ nào.
    """

    def __init__(self, hidden_dim, mode="residual", init_bias=-3.0, post_norm=False):
        super().__init__()
        self.mode = mode
        if mode in ("residual", "gate"):
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
            if mode == "residual":
                nn.init.zeros_(self.fc.weight)
                nn.init.constant_(self.fc.bias, init_bias)
        elif mode == "scalar":
            self.alpha = nn.Parameter(torch.zeros(1))
        self.post_norm = nn.LayerNorm(hidden_dim) if post_norm else nn.Identity()

    def forward(self, h_main, h_aux):
        if self.mode == "replace":
            out = h_aux
        elif self.mode == "scalar":
            out = h_main + self.alpha * h_aux
        else:
            gate = torch.sigmoid(self.fc(torch.cat([h_main, h_aux], dim=-1)))
            if self.mode == "residual":
                out = h_main + gate * h_aux
            else:
                out = gate * h_main + (1.0 - gate) * h_aux
        return self.post_norm(out)

    def gate_report(self):
        """Giá trị để in ra log: model có thực sự dùng PPR không."""
        if self.mode == "scalar":
            return float(self.alpha.detach().abs().item())
        if self.mode in ("residual", "gate"):
            return float(torch.sigmoid(self.fc.bias.detach()).mean().item())
        return 1.0


class ResidualRelGraphConvBlock(nn.Module):
    """
    Pre-norm residual block quanh RelGraphConv (ý tưởng từ model_base2).
    API giữ đúng dạng layer(graph, x, rel_ids, norm) của BaseRGCN.
    """

    def __init__(self, hidden_dim, num_rels, num_bases=None, regularizer="bdd",
                 dropout=0.2, use_self_loop=True, activation=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv = RelGraphConv(
            in_feat=hidden_dim, out_feat=hidden_dim, num_rels=num_rels,
            regularizer=regularizer, num_bases=num_bases,
            activation=None, self_loop=use_self_loop, dropout=0.0,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, graph, x, rel_ids, norm):
        h = self.conv(graph, self.norm1(x), rel_ids, norm)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return self.act(x)


# ================================================================
# Embedding đầu vào
# ================================================================
class EmbeddingLayer(nn.Module):
    """
    Giữ đúng FuseLinker gốc khi để mặc định:
      - min-max toàn ma trận về [0, 1]
      - domain: Linear Poincaré -> Euclidean
      - text: encoder MLP 768 -> hidden
      - fusion: (1 - w) * domain + w * text

    Khác bản cũ: norm mặc định là LayerNorm (BatchNorm nhiễm running-stats giữa
    subgraph train và full graph eval), và decoder chỉ được dựng khi thật sự có
    reconstruction loss.
    """

    def __init__(self, num_nodes, hidden_dim,
                 pretrained_text_embeddings, pretrained_domain_embeddings,
                 freeze=False, w=0.5, dropout=0.2, config=None):
        super().__init__()
        self.cfg = config or EncoderConfig()
        self.w = w
        self.last_recon_loss = None

        if not self.cfg.use_text and not self.cfg.use_domain:
            raise ValueError("use_text và use_domain không thể cùng False.")

        # ---- DOMAIN ----
        if pretrained_domain_embeddings is not None:
            if pretrained_domain_embeddings.shape[0] != num_nodes:
                raise ValueError(
                    f"Domain embedding rows={pretrained_domain_embeddings.shape[0]} "
                    f"!= num_nodes={num_nodes}. Row i must match entity id i."
                )
            domain = torch.from_numpy(pretrained_domain_embeddings).float()
            domain = (domain - domain.min()) / (domain.max() - domain.min())
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(domain, freeze=freeze)
            self.poincare_to_euclidean = nn.Linear(
                pretrained_domain_embeddings.shape[1], hidden_dim
            )
            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.poincare_to_euclidean = nn.Linear(hidden_dim, hidden_dim)
            print("Initialized random domain embeddings.")

        # ---- TEXT ----
        if pretrained_text_embeddings is not None:
            if pretrained_text_embeddings.shape[0] != num_nodes:
                raise ValueError(
                    f"Text embedding rows={pretrained_text_embeddings.shape[0]} "
                    f"!= num_nodes={num_nodes}. Row i must match entity id i."
                )
            text = torch.from_numpy(pretrained_text_embeddings).float()
            text = (text - text.min()) / (text.max() - text.min())
            self.norm_text_embeddings = nn.Embedding.from_pretrained(text, freeze=freeze)
            text_in = pretrained_text_embeddings.shape[1]
            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            self.norm_text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            text_in = hidden_dim
            print("Initialized random text embeddings.")

        if self.cfg.text_encoder == "ae" and self.cfg.text_recon_weight <= 0:
            print("CẢNH BÁO: text_encoder='ae' nhưng text_recon_weight=0 -> decoder "
                  "không có gradient (đúng như bản cũ). Dùng --text_encoder mlp.")
        self.text_encoder = TextEncoder(
            text_in, hidden_dim, kind=self.cfg.text_encoder,
            norm=self.cfg.text_norm, dropout=dropout,
        )

        # ---- FUSION ----
        self.fusion_gate = (
            FusionGate(hidden_dim, dropout) if self.cfg.input_fusion == "gate" else None
        )
        self.input_norm = nn.LayerNorm(hidden_dim) if self.cfg.input_norm else nn.Identity()

        if self.cfg.node_id_embedding:
            self.node_id_embedding = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.node_id_embedding.weight)
        else:
            self.node_id_embedding = None

    def forward(self, graph, node_ids, rel_ids, norm):
        ids = node_ids.squeeze()
        raw_text = self.norm_text_embeddings(ids)

        text_x, decoded = self.text_encoder(raw_text)
        domain_x = self.poincare_to_euclidean(self.norm_domain_embeddings(ids))

        if decoded is not None and self.cfg.text_recon_weight > 0:
            self.last_recon_loss = F.mse_loss(decoded, raw_text)
        else:
            self.last_recon_loss = None

        if not self.cfg.use_text:
            fused = domain_x
        elif not self.cfg.use_domain:
            fused = text_x
        elif self.fusion_gate is not None:
            fused = self.fusion_gate(
                text_x, domain_x, w=torch.full_like(text_x, float(self.w))
            )
        else:
            fused = (1 - self.w) * domain_x + self.w * text_x

        if self.node_id_embedding is not None:
            fused = fused + self.node_id_embedding(ids)
        return self.input_norm(fused)


# ================================================================
# R-GCN
# ================================================================
class BaseRGCN(nn.Module):
    """Giữ nguyên form cũ để train script không phải sửa."""

    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 freeze=False, w=0.5, config=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.pretrained_text_embeddings = pretrained_text_embeddings
        self.pretrained_domain_embeddings = pretrained_domain_embeddings
        self.freeze = freeze
        self.w = w
        self.cfg = config or EncoderConfig()
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)
        for idx in range(self.num_hidden_layers):
            self.layers.append(self.build_hidden_layer(idx))
        output_layer = self.build_output_layer()
        if output_layer is not None:
            self.layers.append(output_layer)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, graph, node_ids, rel_ids, norm):
        for layer in self.layers:
            node_ids = layer(graph, node_ids, rel_ids, norm)
        return node_ids


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(
            self.num_nodes, self.hidden_dim,
            self.pretrained_text_embeddings, self.pretrained_domain_embeddings,
            self.freeze, self.w, dropout=self.dropout, config=self.cfg,
        )

    def build_hidden_layer(self, idx):
        last = idx == self.num_hidden_layers - 1
        regularizer = None if self.cfg.rgcn_regularizer == "none" else self.cfg.rgcn_regularizer
        num_bases = self.num_bases if regularizer is not None else None
        if self.cfg.rgcn_block == "residual":
            return ResidualRelGraphConvBlock(
                self.hidden_dim, self.num_relations, num_bases=num_bases,
                regularizer=regularizer or "basis", dropout=self.dropout,
                use_self_loop=self.use_self_loop, activation=not last,
            )
        return RelGraphConv(
            in_feat=self.hidden_dim, out_feat=self.hidden_dim,
            num_rels=self.num_relations, regularizer=regularizer, num_bases=num_bases,
            activation=None if last else F.relu,
            self_loop=self.use_self_loop, dropout=self.dropout,
        )

    def encode_input(self, node_ids):
        """
        Chỉ chạy EmbeddingLayer (layer 0) — dùng cho nhánh PPR. Nếu chọn
        text_norm='batch' thì giữ BatchNorm ở eval trong lần gọi này để không
        nhiễm running-stats từ tập node PPR.
        """
        bn_state = []
        for module in self.layers[0].modules():
            if isinstance(module, nn.BatchNorm1d):
                bn_state.append((module, module.training))
                module.eval()
        try:
            return self.layers[0](None, node_ids, None, None)
        finally:
            for module, was_training in bn_state:
                module.train(was_training)


# ================================================================
# Link prediction
# ================================================================
class LinkPredict(nn.Module):
    """
    DistMult trên embedding của R-GCN, cộng nhánh phụ PPR tùy chọn.
    Chữ ký giữ nguyên bản cũ; mọi tùy chọn mới nằm trong `config`.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_cuda=False,
                 regularization_param=0.0,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 pretrained_relation_embeddings=None, freeze=False, w=0.5,
                 use_ppr=None, ppr_num_layers=None, ppr_fanout=None,
                 config=None):
        super().__init__()
        cfg = copy.deepcopy(config) if config is not None else EncoderConfig()
        # Cờ kiểu cũ (train script hiện tại truyền vào) vẫn được tôn trọng.
        if use_ppr is not None:
            cfg.use_ppr = bool(use_ppr)
        if ppr_num_layers is not None:
            cfg.ppr_num_layers = int(ppr_num_layers)
        if ppr_fanout is not None:
            cfg.ppr_fanout = int(ppr_fanout)
        self.cfg = cfg

        # Giống FuseLinker gốc: positional arg thứ 8 là use_cuda nhưng gắn vào
        # use_self_loop của BaseRGCN (khi GPU: self_loop=True).
        self.rgcn = RGCN(
            input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
            num_hidden_layers, dropout, use_cuda,
            pretrained_text_embeddings=pretrained_text_embeddings,
            pretrained_domain_embeddings=pretrained_domain_embeddings,
            freeze=freeze, w=w, config=cfg,
        )

        self.use_ppr = cfg.use_ppr
        if cfg.use_ppr:
            ppr_dropout = dropout if cfg.ppr_dropout is None else cfg.ppr_dropout
            self.ppr_branch = PPRBranch(
                hidden_dim, num_layers=cfg.ppr_num_layers, agg=cfg.ppr_agg,
                use_norm=cfg.ppr_layer_norm, dropout=ppr_dropout,
            )
            self.ppr_fusion = PPRFusion(
                hidden_dim, mode=cfg.ppr_fusion,
                init_bias=cfg.ppr_fusion_init_bias, post_norm=cfg.ppr_fusion_norm,
            )
            self.ppr_fanout = int(cfg.ppr_fanout)

        self.regularization_param = regularization_param
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self._last_rgcn_embeddings = None

        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            if cfg.relation_minmax:
                # model.py gốc có bước này; bản model_base4 cũ làm mất.
                rw = self.relation_weights
                self.relation_weights.data.copy_((rw - rw.min()) / (rw.max() - rw.min()))
            print("Loaded pretrained relation embeddings.")
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain("relu"))
            print("Initialized random relation embeddings.")

        print(f"Encoder config: {cfg.describe()}")

    # ---------------- scorer ----------------
    def calculate_score(self, embeddings, triplets):
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        return torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)

    # ---------------- nhánh PPR ----------------
    def _ensure_ppr_weight(self, ppr_graph, ppr_edge_weight):
        if "ppr_w" in ppr_graph.edata:
            return ppr_graph
        if ppr_edge_weight is None:
            raise ValueError("PPR graph thiếu edata['ppr_w'] và ppr_edge_weight là None.")
        ppr_graph.edata["ppr_w"] = ppr_edge_weight.detach().to(ppr_graph.device).reshape(-1, 1)
        return ppr_graph

    def _ppr_sampled(self, ppr_graph, seed_ids, device):
        """
        Neighbor sampling trên CPU, GCN trên GPU. Gppr đã cắt top-k nên
        ppr_fanout=-1 (mặc định) vừa rẻ vừa khớp chính xác với đường eval.
        """
        g_cpu = ppr_graph if ppr_graph.device.type == "cpu" else ppr_graph.cpu()
        frontier = seed_ids.detach().cpu().long().view(-1)
        fanout = int(self.ppr_fanout)
        if fanout == 0:
            fanout = 1
        blocks = []
        for _ in range(len(self.ppr_branch.layers)):
            sg = dgl.sampling.sample_neighbors(g_cpu, frontier, fanout, edge_dir="in")
            block = dgl.to_block(sg, dst_nodes=frontier)
            frontier = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        src_ids = blocks[0].srcdata[dgl.NID].to(device)
        h = self.rgcn.encode_input(src_ids.view(-1, 1))
        for block, layer in zip(blocks, self.ppr_branch.layers):
            block = block.to(device)
            h = layer(block, h, block.edata["ppr_w"].reshape(-1))
        return h

    def _ppr_full(self, ppr_graph, ppr_edge_weight, all_node_ids, ids, device):
        """
        Full-graph PPR encode. Mặc định chạy trên CPU (card 4GB) bằng một bản
        deepcopy nhỏ của ppr_branch, nên tham số/Adam state trên GPU không bị
        đụng tới. ppr_eval_device="same" thì chạy thẳng trên device của model.
        """
        target = torch.device("cpu") if self.cfg.ppr_eval_device == "cpu" else device
        n = ppr_graph.number_of_nodes()
        ids_all = all_node_ids.to(device)
        chunks = []
        for start in range(0, n, 4096):
            chunks.append(
                self.rgcn.encode_input(ids_all[start:start + 4096]).detach().to(target)
            )
        x = torch.cat(chunks, dim=0)

        g = ppr_graph.to(target)
        if "ppr_w" in g.edata:
            ew = g.edata["ppr_w"].to(target).reshape(-1)
        else:
            ew = ppr_edge_weight.detach().to(target).reshape(-1)

        if target == device:
            branch = self.ppr_branch
            was_training = branch.training
            branch.eval()
            with torch.no_grad():
                h_full = branch(g, x, ew)
            branch.train(was_training)
        else:
            branch = copy.deepcopy(self.ppr_branch).to(target).eval()
            with torch.no_grad():
                h_full = branch(g, x, ew)
            del branch
        return h_full[ids.detach().to(target)].to(device)

    def _ppr_features(self, ppr_graph, ppr_edge_weight, node_ids, all_node_ids):
        ids = node_ids.squeeze()
        device = ids.device
        g = self._ensure_ppr_weight(ppr_graph, ppr_edge_weight)
        if self.training:
            return self._ppr_sampled(g, ids, device)
        return self._ppr_full(g, ppr_edge_weight, all_node_ids, ids, device)

    # ---------------- forward / loss ----------------
    def forward(self, graph, node_ids, rel_ids, norm,
                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        h_rgcn = self.rgcn(graph, node_ids, rel_ids, norm)
        self._last_rgcn_embeddings = h_rgcn

        if self.use_ppr and ppr_graph is not None and all_node_ids is not None:
            h_ppr = self._ppr_features(ppr_graph, ppr_edge_weight, node_ids, all_node_ids)
            return self.ppr_fusion(h_rgcn, h_ppr)
        return h_rgcn

    def regularization_loss(self, embeddings):
        """
        reg_on="rgcn"  : phạt embedding của backbone (mặc định). Không phạt phần
                         PPR cộng thêm — nếu phạt, reg_param sẽ chủ động đóng
                         gate lại và nhánh phụ chỉ còn là nhiễu.
        reg_on="fused" : hành vi cũ, phạt embedding sau fusion.
        reg_on="none"  : chỉ phạt relation_weights.
        """
        rel_term = torch.mean(self.relation_weights.pow(2))
        if self.cfg.reg_on == "none":
            return rel_term
        if self.cfg.reg_on == "rgcn" and self._last_rgcn_embeddings is not None:
            embeddings = self._last_rgcn_embeddings
        return torch.mean(embeddings.pow(2)) + rel_term

    def get_loss(self, graph, embeddings, triplets, labels):
        score = self.calculate_score(embeddings, triplets)
        loss = F.binary_cross_entropy_with_logits(score, labels)
        loss = loss + self.regularization_param * self.regularization_loss(embeddings)

        recon = getattr(self.rgcn.layers[0], "last_recon_loss", None)
        if recon is not None and self.cfg.text_recon_weight > 0:
            loss = loss + self.cfg.text_recon_weight * recon
        return loss

    def ppr_gate_value(self):
        """In giá trị này mỗi vài nghìn iteration: gate ~ 0 nghĩa là PPR vô dụng."""
        if not self.use_ppr:
            return 0.0
        return self.ppr_fusion.gate_report()

    # ---------------- checkpoint / eval ----------------
    @classmethod
    def load_checkpoint(cls, checkpoint_path, device=None, **init_kwargs):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(**init_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as exc:
            keys = " ".join(checkpoint["state_dict"].keys())
            hint = ""
            if "autoencoder" in keys:
                hint = (" Checkpoint cũ dùng TextEmbeddingAutoencoder -> nạp bằng "
                        "--model_module model / model_base2, hoặc train lại.")
            elif "ppr_branch" not in keys and "fusion_gate" in keys:
                hint = " Kiểm tra --model_module (model_base2 vs model_base4)."
            raise RuntimeError(f"{exc}{hint}") from exc
        model.to(device)
        return model, checkpoint.get("iteration", None)

    def get_final_embeddings(self, graph, node_ids, rel_ids, norm,
                             ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(
                graph, node_ids, rel_ids, norm,
                ppr_graph=ppr_graph, ppr_edge_weight=ppr_edge_weight,
                all_node_ids=all_node_ids,
            )
        if was_training:
            self.train()
        return embeddings

    def evaluate_auroc(self, embeddings, test_triplets, total_data,
                       batch_size=4096, seed=None, return_pairs=False):
        from calc_auroc import calc_auroc
        return calc_auroc(
            embeddings, self.relation_weights, test_triplets, total_data,
            batch_size=batch_size, seed=seed, return_pairs=return_pairs,
        )

    def evaluate_auroc_on_graph(self, graph, node_ids, rel_ids, norm,
                                test_triplets, total_data,
                                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None,
                                batch_size=4096, seed=None, return_pairs=False):
        embeddings = self.get_final_embeddings(
            graph, node_ids, rel_ids, norm,
            ppr_graph=ppr_graph, ppr_edge_weight=ppr_edge_weight,
            all_node_ids=all_node_ids,
        )
        return self.evaluate_auroc(
            embeddings, test_triplets, total_data,
            batch_size=batch_size, seed=seed, return_pairs=return_pairs,
        )
