###Trong file này tôi muốn cài đặt thêm Generating Auxiliary Network
# Generating auxiliary network
# HGDC applies the PPR [39] to generate an auxiliary network. The
# PPR is a random walk-based method which can measure the node
# proximity (i.e. the closeness among multiple nodes) in a graph.
# Taking a network G with N genes as an example, the iteration
# equation of PPR is written as follows:
# −→
# s i = cM−→
# s i + (1 − c) −→
# e i (1)
# PPR
# (−→
# s i
# )
# = (1 − c) (IN − cM)−1−→
# e i (2)
# where −→
# s i is a N-dimensional probability vector, which represents
# the state of i-th walker (i.e. gene i); −→
# e i is a one-hot vector and
# its i-th nonzero entry indicates that the i-th walker can return to itself with probability 1-c, while c is the damping factor; M is the
# transition matrix derived by M = D−1/2AD1/2, where A and D are
# the adjacency matrix and the diagonal degree matrix of graph G,
# respectively.
# The PPR defines an iterative process in which a walker ran-
# domly walks along the edges following the Markov process with
# probability c and returns to itself with probability 1-c. The iterative
# process of PPR converges when the values in −→
# s i are no longer
# significantly updated, and the final PPR
# (−→
# s i
# )
# can be obtained by
# Equation (2) (i.e. the solution of −→
# s i in Equation (1)). The value
# of j-th element in PPR
# (−→
# s i
# )
# measures the connection strengthen
# of gene j to gene i. As the PPR
# (−→
# s i
# )
# is derived by considering
# the overall structures of a network, the PPR naturally reflects
# the structural similarity or proximity among multiple genes in a
# biomolecular network [40]. After replacing the −→
# e i in Equation (2)
# by the identity matrix IN, we can obtain the structural similarity
# matrix SN×N = (1 − c) (IN − cT)−1, in which Si,j measures the struc-
# tural similarity or proximity between gene i and gene j.
# However, the structural similarity matrix S is not sparse.
# Directly applying S to generate the auxiliary network will result
# in a dense network, bringing unaffordable computing burdens.
# Thus, it is necessary to sparse S. Previous study [41] shows that
# the entry values of S are typically highly localized, allowing us
# to truncate small values of S with a threshold ε to obtain the
# sparse matrix ∼
# S, which still retains most of the information of
# S. That is, the elements in S below ε are set to zero. The sparse
# matrix ∼
# S is used to construct the auxiliary network Gppr, in which
# an edge represents the structural similarity or proximity among
# genes. We utilize the auxiliary network Gppr as the complement
# to the biomolecular network to capture more useful structural
# similarities of genes
import copy
from pathlib import Path

import dgl
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class PPRGraphConv(nn.Module):
    """
    GCN layer có trọng số cạnh cho auxiliary network Gppr:
        h_v' = ReLU( LayerNorm( sum_{u in N(v)} w_{u,v} * (W h_u) ) )
    Không có relation type vì Gppr là đồ thị thuần.

    use_norm=False (dùng cho lớp cuối): BỎ LayerNorm để giữ lại magnitude của tổng
    trọng số PPR trên neighbor. Magnitude này tương quan với bậc/độ liên kết toàn cục
    của node và với "độ gần cấu trúc" của cặp node — chính là tín hiệu mà LayerNorm
    (chuẩn hóa var=1 theo từng node) sẽ xóa mất. Giữ lại giúp task degree + edge.
    """

    def __init__(self, in_dim, out_dim, activation=True, use_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_norm else nn.Identity()
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, g, x, edge_weight):
        with g.local_scope():
            # srcdata/dstdata: works for both full graphs and sampled DGL blocks.
            g.srcdata['h'] = self.linear(x)
            g.edata['w'] = edge_weight.reshape(-1, 1)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
            h = g.dstdata['h_new']
        h = self.norm(h)
        h = self.act(h)
        return h


class PPRBranch(nn.Module):
    """
    Nhánh phụ chạy trên toàn bộ num_nodes với auxiliary network Gppr.
    Dùng feature đầu vào giống với RGCN (sau EmbeddingLayer) để không gian biểu diễn tương thích.

    Lớp cuối bỏ cả activation lẫn LayerNorm để magnitude cấu trúc toàn cục (đại diện cho
    bậc node và độ gần của cặp node) được truyền nguyên vẹn vào fusion residual.
    """

    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            PPRGraphConv(
                hidden_dim, hidden_dim,
                activation=(i < num_layers - 1),
                use_norm=(i < num_layers - 1),
            )
            for i in range(num_layers)
        ])

    def forward(self, ppr_graph, x, edge_weight):
        h = x
        for layer in self.layers:
            h = layer(ppr_graph, h, edge_weight)
        return h

#Hàm biến đổi text embedding về chiều cùng chiều với domain embedding
class TextEmbeddingAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(TextEmbeddingAutoencoder, self).__init__()
        # LayerNorm thay cho BatchNorm1d: encoder được gọi cả trên subgraph (nhánh RGCN)
        # lẫn trên toàn bộ N node (nhánh PPR) trong cùng một forward, nên BatchNorm sẽ
        # nhiễm running-stats giữa 2 phân phối batch và lệch train/eval. LayerNorm chuẩn
        # hóa theo từng sample -> nhất quán, không phụ thuộc kích thước batch.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.LayerNorm(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.LayerNorm(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.LayerNorm(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class MLPProjector(nn.Module):
    """
    Ý tưởng lấy từ file đầu tiên:
    projector riêng cho từng nguồn embedding trước khi fusion
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(True),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionGate(nn.Module):
    """
    Trộn 2 nguồn bằng trọng số học được (flex), thay cho w cố định của model.py.
    gate = sigmoid(Linear([a, b]));  h = gate * a + (1 - gate) * b
    Nếu truyền w, neo nhẹ gate về prior đó.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, a, b, w=None):
        gate = torch.sigmoid(self.fc(torch.cat([a, b], dim=-1)))
        if w is not None:
            gate = 0.5 * gate + 0.5 * w
        return gate * a + (1.0 - gate) * b


class ResidualPPRFusion(nn.Module):
    """
    Fusion residual cho nhánh phụ (PPR):  fused = h_main + g * h_aux
    với g = sigmoid(fc([h_main, h_aux])).

    Khác với FusionGate (gate*a + (1-gate)*b, khởi tạo gate~0.5 -> pha loãng ngay
    50% tín hiệu backbone), ở đây fc được khởi tạo 0 và bias âm nên g ~ 0 lúc đầu:
    output bắt đầu bằng đúng h_main (tương đương base), rồi model mới học cách
    "cộng thêm" thông tin cấu trúc từ PPR. Nhờ vậy upgrade không tệ hơn base ngay
    từ đầu và chỉ có thể cải thiện ở degree/edge.
    """

    def __init__(self, hidden_dim, init_bias=-3.0):
        super().__init__()
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, h_main, h_aux):
        gate = torch.sigmoid(self.fc(torch.cat([h_main, h_aux], dim=-1)))
        return h_main + gate * h_aux


class BaseRGCN(nn.Module):
    """
    Giữ nguyên form của model.py cũ để main.py không phải sửa.
    """

    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 freeze=False, w=0.5):
        super(BaseRGCN, self).__init__()
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

        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)

        for idx in range(self.num_hidden_layers):
            hidden_layer = self.build_hidden_layer(idx)
            self.layers.append(hidden_layer)

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
        """
        Forward tuần tự như model.py / BaseRGCN: không Jumping Knowledge.
        """
        for layer in self.layers:
            node_ids = layer(graph, node_ids, rel_ids, norm)
        return node_ids


class EmbeddingLayer(nn.Module):
    """
    Bản tương thích với main.py + tinh thần file đầu tiên:
    - projector riêng cho text/domain
    - fusion gate học được
    - vẫn giữ parameter w từ code cũ
    """

    def __init__(self, num_nodes, hidden_dim,
                 pretrained_text_embeddings,
                 pretrained_domain_embeddings,
                 freeze=False, w=0.5, dropout=0.2):
        super(EmbeddingLayer, self).__init__()
        self.w = w
        self.hidden_dim = hidden_dim

        # ---- DOMAIN ----
        if pretrained_domain_embeddings is not None:
            if pretrained_domain_embeddings.shape[0] != num_nodes:
                raise ValueError(
                    f"Domain embedding rows={pretrained_domain_embeddings.shape[0]} "
                    f"!= num_nodes={num_nodes}. Row i must match entity id i "
                    f"(appearance order in train+valid+test). "
                    f"Mismatch causes CUDA indexSelectLargeIndex / "
                    f"`srcIndex < srcSelectDimSize`."
                )
            domain_embeddings = torch.from_numpy(pretrained_domain_embeddings).float()
            self.domain_embeddings = nn.Embedding.from_pretrained(
                domain_embeddings, freeze=freeze
            )
            self.domain_projector = MLPProjector(
                pretrained_domain_embeddings.shape[1], hidden_dim, dropout
            )
            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            self.domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.domain_projector = MLPProjector(hidden_dim, hidden_dim, dropout)
            print("Initialized random domain embeddings.")

        # ---- TEXT ----
        if pretrained_text_embeddings is not None:
            if pretrained_text_embeddings.shape[0] != num_nodes:
                raise ValueError(
                    f"Text embedding rows={pretrained_text_embeddings.shape[0]} "
                    f"!= num_nodes={num_nodes}. Row i must match entity id i "
                    f"(appearance order in train+valid+test). "
                    f"Mismatch causes CUDA indexSelectLargeIndex / "
                    f"`srcIndex < srcSelectDimSize`."
                )
            text_embeddings = torch.from_numpy(pretrained_text_embeddings).float()
            self.text_embeddings = nn.Embedding.from_pretrained(
                text_embeddings, freeze=freeze
            )
            self.autoencoder = TextEmbeddingAutoencoder(
                pretrained_text_embeddings.shape[1], hidden_dim, dropout_rate=dropout
            )
            self.text_post = nn.LayerNorm(hidden_dim)
            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            self.text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.autoencoder = TextEmbeddingAutoencoder(
                hidden_dim, hidden_dim, dropout_rate=dropout
            )
            self.text_post = nn.LayerNorm(hidden_dim)
            print("Initialized random text embeddings.")

        self.fusion_gate = FusionGate(hidden_dim)

    def forward(self, graph, node_ids, rel_ids, norm):
        node_ids = node_ids.squeeze()

        # text branch
        raw_text = self.text_embeddings(node_ids)
        text_x, _ = self.autoencoder(raw_text)
        text_x = self.text_post(text_x)

        # domain branch
        raw_domain = self.domain_embeddings(node_ids)
        domain_x = self.domain_projector(raw_domain)

        # prior từ w để không mất hẳn ý tưởng weighted fusion ban đầu
        w_prior = torch.full_like(text_x, float(self.w))

        fused = self.fusion_gate(text_x, domain_x, w=w_prior)
        return fused


class RGCN(BaseRGCN):
    """
    Hidden layer giữ như BaseRGCN / model.py: RelGraphConv thuần (bdd).
    EmbeddingLayer vẫn dùng fusion flex của model_base4.
    """

    def build_input_layer(self):
        return EmbeddingLayer(
            self.num_nodes,
            self.hidden_dim,
            self.pretrained_text_embeddings,
            self.pretrained_domain_embeddings,
            self.freeze,
            self.w,
            self.dropout
        )

    def build_hidden_layer(self, idx):
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(
            in_feat=self.hidden_dim,
            out_feat=self.hidden_dim,
            num_rels=self.num_relations,
            regularizer='bdd',
            num_bases=self.num_bases,
            activation=activation,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
        )

    def encode_input(self, node_ids):
        """
        Chỉ chạy EmbeddingLayer (layer đầu tiên) để lấy feature khởi tạo cho node_ids.
        Dùng cho nhánh PPR (cần encode toàn bộ num_nodes).
        """
        return self.layers[0](None, node_ids, None, None)


class EdgeTypeClassifier(nn.Module):
    """
    Head SSL cho tác vụ dự đoán loại cạnh (LukePi: linear_pred_edges).
    Nhận embedding của 2 node (src, dst), nối lại rồi phân loại vào
    (num_rels + 1) lớp: num_rels loại quan hệ thật + 1 lớp "negative/fake edge".
    """

    def __init__(self, hidden_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, h_src, h_dst):
        return self.net(torch.cat([h_src, h_dst], dim=-1))


class DegreeClassifier(nn.Module):
    """
    Head SSL cho tác vụ dự đoán bậc của node (LukePi: linear_pred_nodes).
    Nhận embedding của node rồi phân loại vào num_bins bucket bậc.
    """

    def __init__(self, hidden_dim, num_bins, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, h):
        return self.net(h)


class LinkPredict(nn.Module):
    """
    API giữ nguyên hoàn toàn để main.py chạy được ngay.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_cuda=False,
                 regularization_param=0.0,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 pretrained_relation_embeddings=None, freeze=False, w=0.5,
                 use_ppr=True, ppr_num_layers=2, ppr_fanout=15):
        super(LinkPredict, self).__init__()

        # giữ đúng convention cũ: num_relations * 2 cho graph conv
        self.rgcn = RGCN(
            input_dim,
            hidden_dim,
            hidden_dim,
            num_relations * 2,
            num_bases,
            num_hidden_layers,
            dropout,
            use_self_loop=True,
            use_cuda=use_cuda,
            pretrained_text_embeddings=pretrained_text_embeddings,
            pretrained_domain_embeddings=pretrained_domain_embeddings,
            freeze=freeze,
            w=w
        )

        # Nhánh auxiliary PPR (HGDC style) - song song với RGCN
        self.use_ppr = use_ppr
        if use_ppr:
            self.ppr_branch = PPRBranch(hidden_dim, num_layers=ppr_num_layers)
            # Fusion residual: bắt đầu ~ base (h_rgcn), học cách cộng thêm tín hiệu PPR.
            self.ppr_fusion = ResidualPPRFusion(hidden_dim)
            self.ppr_fanout = int(ppr_fanout)
            
        self.regularization_param = regularization_param
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations

        # Giữ relation_weights để main.py và calc_mrr dùng được như cũ
        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            print("Loaded pretrained relation embeddings.")
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))
            print("Initialized random relation embeddings.")

    def calculate_score(self, embeddings, triplets):
        """
        Giữ DistMult scorer để tương thích hoàn toàn với pipeline đánh giá hiện tại.
        """
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
        return score

    def _ensure_ppr_weight(self, ppr_graph, ppr_edge_weight):
        if "ppr_w" in ppr_graph.edata:
            return ppr_graph
        w = ppr_edge_weight
        if w is None:
            raise ValueError("PPR graph is missing edata['ppr_w'] and ppr_edge_weight is None.")
        ppr_graph.edata["ppr_w"] = w.detach().to(ppr_graph.device).reshape(-1, 1)
        return ppr_graph

    def _ppr_sampled(self, ppr_graph, seed_ids, device):
        """
        Neighbor-sampled PPR GCN for the current seed nodes.
        Keeps the full Gppr on CPU and only moves a small block to GPU.
        """
        g_cpu = ppr_graph if ppr_graph.device.type == "cpu" else ppr_graph.cpu()
        frontier = seed_ids.detach().cpu().long().view(-1)
        fanout = max(int(self.ppr_fanout), 1)
        blocks = []
        for _ in range(len(self.ppr_branch.layers)):
            sg = dgl.sampling.sample_neighbors(
                g_cpu, frontier, fanout, edge_dir="in"
            )
            block = dgl.to_block(sg, dst_nodes=frontier)
            frontier = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        src_ids = blocks[0].srcdata[dgl.NID].to(device)
        h = self.rgcn.encode_input(src_ids.view(-1, 1))
        for block, layer in zip(blocks, self.ppr_branch.layers):
            block = block.to(device)
            ew = block.edata["ppr_w"].reshape(-1)
            h = layer(block, h, ew)
        return h

    def _ppr_full_cpu(self, ppr_graph, ppr_edge_weight, all_node_ids, ids, device):
        """
        Full-graph PPR encode on CPU (eval). Does not move training params off GPU
        — uses a tiny deepcopy of ppr_branch so Adam state stays valid.
        """
        n = ppr_graph.number_of_nodes()
        ids_all = all_node_ids.to(device)
        chunks = []
        bs = 4096
        for start in range(0, n, bs):
            chunks.append(self.rgcn.encode_input(ids_all[start:start + bs]).detach().cpu())
        x = torch.cat(chunks, dim=0)
        g_cpu = ppr_graph.cpu()
        if "ppr_w" in g_cpu.edata:
            ew = g_cpu.edata["ppr_w"].cpu().reshape(-1)
        else:
            ew = ppr_edge_weight.detach().cpu().reshape(-1)
        branch_cpu = copy.deepcopy(self.ppr_branch).cpu().eval()
        with torch.no_grad():
            h_full = branch_cpu(g_cpu, x, ew)
        del branch_cpu
        return h_full[ids.detach().cpu()].to(device)

    def _ppr_features(self, ppr_graph, ppr_edge_weight, node_ids, all_node_ids):
        ids = node_ids.squeeze()
        device = ids.device
        g = self._ensure_ppr_weight(ppr_graph, ppr_edge_weight)
        if self.training:
            return self._ppr_sampled(g, ids, device)
        return self._ppr_full_cpu(g, ppr_edge_weight, all_node_ids, ids, device)

    def forward(self, graph, node_ids, rel_ids, norm,
                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        """
        - graph, node_ids, rel_ids, norm: nhánh RGCN (subgraph huấn luyện hoặc full test graph).
        - ppr_graph, ppr_edge_weight, all_node_ids: nhánh PPR.
          Train: sample neighborhood trên CPU rồi GCN trên GPU.
          Eval: PPR GCN full graph trên CPU để khỏi OOM card 4GB.
          Nếu để None, model trượt về hành vi cũ (chỉ RGCN).
        """
        h_rgcn = self.rgcn(graph, node_ids, rel_ids, norm)

        if self.use_ppr and ppr_graph is not None and all_node_ids is not None:
            h_ppr = self._ppr_features(
                ppr_graph, ppr_edge_weight, node_ids, all_node_ids
            )
            return self.ppr_fusion(h_rgcn, h_ppr)

        return h_rgcn

    def regularization_loss(self, embeddings):
        return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))

    def get_loss(self, graph, embeddings, triplets, labels):
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)
        return prediction_loss + self.regularization_param * reg_loss

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device=None, **init_kwargs):
        """
        Load trọng số đã train từ file .pth.
        init_kwargs phải khớp cấu hình lúc train (n_hidden, num_hidden_layers, w, ...).
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(**init_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except RuntimeError as exc:
            keys = " ".join(checkpoint["state_dict"].keys())
            hint = ""
            if "poincare_to_euclidean" in keys or "norm_text_embeddings" in keys:
                hint = " Checkpoint này train bằng model.py → dùng --model_module model hoặc auto."
            elif "ppr_branch" not in keys and ("fusion_gate" in keys or "domain_projector" in keys):
                hint = " Kiểm tra --model_module (model_base2 vs model_base4)."
            raise RuntimeError(f"{exc}{hint}") from exc
        model.to(device)
        iteration = checkpoint.get("iteration", None)
        return model, iteration

    def get_final_embeddings(self, graph, node_ids, rel_ids, norm,
                             ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        """
        Đưa embedding cho trước (text/domain .npy) qua mô hình đã train
        → embedding node cuối cùng dùng cho MR/MRR/AUROC.
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(
                graph, node_ids, rel_ids, norm,
                ppr_graph=ppr_graph,
                ppr_edge_weight=ppr_edge_weight,
                all_node_ids=all_node_ids,
            )
        if was_training:
            self.train()
        return embeddings

    def evaluate_auroc(self, embeddings, test_triplets, total_data,
                       batch_size=4096, seed=None, return_pairs=False):
        """
        Tính AUROC từ embedding cuối và relation_weights.
        """
        from calc_auroc import calc_auroc
        return calc_auroc(
            embeddings,
            self.relation_weights,
            test_triplets,
            total_data,
            batch_size=batch_size,
            seed=seed,
            return_pairs=return_pairs,
        )

    def evaluate_auroc_on_graph(self, graph, node_ids, rel_ids, norm,
                                test_triplets, total_data,
                                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None,
                                batch_size=4096, seed=None, return_pairs=False):
        """
        Luồng đầy đủ: forward đồ thị test → embedding cuối → AUROC.
        """
        embeddings = self.get_final_embeddings(
            graph, node_ids, rel_ids, norm,
            ppr_graph=ppr_graph,
            ppr_edge_weight=ppr_edge_weight,
            all_node_ids=all_node_ids,
        )
        return self.evaluate_auroc(
            embeddings, test_triplets, total_data,
            batch_size=batch_size, seed=seed, return_pairs=return_pairs,
        )