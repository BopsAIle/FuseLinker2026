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


def compute_ppr_sparse(num_nodes, train_triples, c=0.15, epsilon=1e-4,
                        add_self_loop=True, use_iterative=False, num_iter=50,
                        topk=50):
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
        use_iterative: True -> ép dùng iterative streaming (không tạo dense NxN).
        num_iter: số lần lặp khi dùng iterative streaming.
        topk: số cạnh tối đa giữ lại cho mỗi node sau khi sparse PPR.

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

    # Khi có top-k thì dùng iterative streaming để tránh cấp phát dense NxN gây tràn RAM.
    # Nhánh dense chỉ giữ lại để tương thích khi cần chạy đúng công thức closed-form.
    use_streaming = use_iterative or (topk is not None and topk > 0)
    if use_streaming:
        S_sparse = _compute_ppr_streaming_sparse(M, c, epsilon, num_iter, topk)
    else:
        I = sp.eye(num_nodes, format='csr')
        S_dense = (1.0 - c) * np.linalg.inv((I - c * M).toarray())
        S_dense = S_dense.astype(np.float32)
        S_dense[np.abs(S_dense) < epsilon] = 0.0
        np.fill_diagonal(S_dense, 0.0)
        S_sparse = sp.coo_matrix(S_dense)
        del S_dense

    src_t = torch.from_numpy(S_sparse.row.astype(np.int64))
    dst_t = torch.from_numpy(S_sparse.col.astype(np.int64))
    w_t = torch.from_numpy(S_sparse.data.astype(np.float32))

    ppr_graph = dgl.graph((src_t, dst_t), num_nodes=num_nodes)
    return ppr_graph, w_t


class PPRGraphConv(nn.Module):
    """
    GCN layer có trọng số cạnh cho auxiliary network Gppr:
        h_v' = ReLU( LayerNorm( sum_{u in N(v)} w_{u,v} * (W h_u) ) )
    Không có relation type vì Gppr là đồ thị thuần.
    """

    def __init__(self, in_dim, out_dim, dropout=0.2, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_weight):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.edata['w'] = edge_weight.view(-1, 1)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
            h = g.ndata['h_new']
        h = self.norm(h)
        h = self.act(h)
        return self.dropout(h)


class PPRBranch(nn.Module):
    """
    Nhánh phụ chạy trên toàn bộ num_nodes với auxiliary network Gppr.
    Dùng feature đầu vào giống với RGCN (sau EmbeddingLayer) để không gian biểu diễn tương thích.
    """

    def __init__(self, hidden_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([
            PPRGraphConv(
                hidden_dim, hidden_dim,
                dropout=dropout,
                activation=(i < num_layers - 1),
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
    """
    Giữ lại đúng tinh thần code cũ:
    - dùng encoder để giảm chiều text embedding
    - decoder vẫn tồn tại để giữ cấu trúc quen thuộc, nhưng forward của model chỉ dùng encoded
    """
    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super(TextEmbeddingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class FusionGate(nn.Module):
    """
    Gated fusion học được thay cho weighted average cố định.
    Vẫn có thể giữ w như bias mềm để không phá hẳn ý tưởng cũ.
    """
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        fusion_in = hidden_dim * 4

        self.gate = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        stats = torch.cat(
            [text_x, domain_x, torch.abs(text_x - domain_x), text_x * domain_x],
            dim=-1,
        )
        gate = self.gate(stats)

        # Nếu có w, dùng w làm prior mềm để vẫn giữ tinh thần file model.py cũ
        if w is not None:
            gate = 0.5 * gate + 0.5 * w

        fused = gate * text_x + (1.0 - gate) * domain_x
        fused = fused + self.residual(torch.cat([text_x, domain_x], dim=-1))
        return self.norm(fused)


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
        h = node_ids
        layer_outputs = []

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                h = layer(graph, h, rel_ids, norm)
            else:
                h = layer(graph, h, rel_ids, norm)
                layer_outputs.append(h)

        # Jumping Knowledge nhẹ: lấy mean các hidden states
        if len(layer_outputs) > 1:
            stacked = torch.stack(layer_outputs, dim=0)
            h = torch.mean(stacked, dim=0)

        return h


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

        self.fusion_gate = FusionGate(hidden_dim, dropout)
        self.node_id_embedding = nn.Embedding(num_nodes, hidden_dim)
        nn.init.xavier_uniform_(self.node_id_embedding.weight)
        self.dropout = nn.Dropout(dropout)

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

        # residual node-id embedding giúp ổn định hơn
        fused = fused + self.node_id_embedding(node_ids)

        return self.dropout(fused)


class ResidualRelGraphConvBlock(nn.Module):
    """
    Lấy ý tưởng từ residual block của file đầu tiên nhưng dùng DGL RelGraphConv.
    API giữ nguyên: forward(graph, x, rel_ids, norm)
    """

    def __init__(self, hidden_dim, num_rels, num_bases=None,
                 dropout=0.2, use_self_loop=True, activation=True):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv = RelGraphConv(
            in_feat=hidden_dim,
            out_feat=hidden_dim,
            num_rels=num_rels,
            regularizer='bdd',
            num_bases=num_bases,
            activation=None,
            self_loop=use_self_loop,
            dropout=0.0
        )

        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.use_activation = activation
        self.act = nn.ReLU()

    def forward(self, graph, x, rel_ids, norm):
        h = self.norm1(x)
        h = self.conv(graph, h, rel_ids, norm)
        x = x + self.dropout(h)

        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        x = x + self.dropout(h2)

        if self.use_activation:
            x = self.act(x)

        return x


class RGCN(BaseRGCN):
    """
    Bản RGCN mới:
    - giữ API cũ
    - input fusion kiểu file đầu tiên
    - hidden block residual kiểu file đầu tiên
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
        activation = idx < self.num_hidden_layers - 1
        return ResidualRelGraphConvBlock(
            hidden_dim=self.hidden_dim,
            num_rels=self.num_relations,
            num_bases=self.num_bases,
            dropout=self.dropout,
            use_self_loop=self.use_self_loop,
            activation=activation
        )

    def encode_input(self, node_ids):
        """
        Chỉ chạy EmbeddingLayer (layer đầu tiên) để lấy feature khởi tạo cho node_ids.
        Dùng cho nhánh PPR (cần encode toàn bộ num_nodes).
        """
        return self.layers[0](None, node_ids, None, None)


class LinkPredict(nn.Module):
    """
    API giữ nguyên hoàn toàn để main.py chạy được ngay.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_cuda=False,
                 regularization_param=0.0,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 pretrained_relation_embeddings=None, freeze=False, w=0.5,
                 use_ppr=True, ppr_num_layers=2):
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
            self.ppr_branch = PPRBranch(hidden_dim, num_layers=ppr_num_layers, dropout=dropout)
            self.ppr_fusion = FusionGate(hidden_dim, dropout)
            
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

    def forward(self, graph, node_ids, rel_ids, norm,
                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        """
        - graph, node_ids, rel_ids, norm: nhánh RGCN (subgraph huấn luyện hoặc full test graph).
        - ppr_graph, ppr_edge_weight, all_node_ids: nhánh PPR trên toàn bộ num_nodes.
          Nếu để None, model trượt về hành vi cũ (chỉ RGCN).
        """
        h_rgcn = self.rgcn(graph, node_ids, rel_ids, norm)

        if self.use_ppr and ppr_graph is not None and all_node_ids is not None:
            # Encode toàn bộ num_nodes bằng EmbeddingLayer rồi propagate qua Gppr
            x_full = self.rgcn.encode_input(all_node_ids)
            h_ppr_full = self.ppr_branch(ppr_graph, x_full, ppr_edge_weight)

            # Gather về đúng các node nằm trong subgraph hiện tại
            ids = node_ids.squeeze()
            h_ppr = h_ppr_full[ids]

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
        model.load_state_dict(checkpoint["state_dict"])
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