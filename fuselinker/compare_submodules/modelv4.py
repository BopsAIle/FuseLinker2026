"""
Semantic multi-source R-GCN for biomedical knowledge-graph node embeddings.

Design goal
===========
Each node already has two pretrained semantic views:
1) text/name embedding;
2) medical/domain-knowledge embedding.

The model first projects and aligns those two semantic views, fuses them without
using a free node-ID lookup table, and only then lets graph structure refine the
representation through a lightweight LukePi-style residual R-GCN.  A relation-agnostic PPR branch is used
as a conservative auxiliary structural signal.

Important changes from model_base4.py / model_base4_best.py
------------------------------------------------------------
- No trainable node-ID shortcut: better transfer to weak/cold-start nodes.
- Separate LayerNorm + MLP projectors for text and medical embeddings.
- Feature-wise gated fusion learned directly from data; no fixed 0.5/w prior.
- Optional weak cross-modal alignment loss for corresponding text/domain views.
- R-GCN kept close to model.py (BDD regularizer and original activation pattern).
- LukePi-style learnable residual alpha at each R-GCN layer.
- No R-GCN FFN, LayerNorm, graph-scale, Jumping Knowledge, or final LayerNorm.
- GPU-only blockwise PPR power iteration with torch.sparse.mm.
- Final PPR graph keeps at most top-50 strongest PPR neighbours PER center node.
- PPR edge direction is neighbour -> center, so message passing for a center
  actually aggregates features from that center's own top-PPR neighbours.
- PPR remains a small auxiliary signal by default because it ignores relation
  types, while R-GCN remains the main relation-aware encoder.

The public LinkPredict API and DistMult scorer are intentionally preserved so
existing training/evaluation code can continue to use relation_weights.
"""

import dgl
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# CUDA PPR
# ============================================================================

@torch.no_grad()
def _build_normalized_sparse_adjacency_cuda(
    num_nodes,
    train_triples,
    device,
    add_self_loop=False,
):
    """Build M = D^{-1/2} A D^{-1/2} directly on CUDA as sparse COO."""
    triples = torch.as_tensor(train_triples, dtype=torch.long, device=device)
    if triples.ndim != 2 or triples.shape[1] < 3:
        raise ValueError("train_triples must have shape [E, 3] or wider.")

    src = triples[:, 0]
    dst = triples[:, 2]

    # Structural PPR is relation-agnostic and uses an undirected topology.
    row = torch.cat([src, dst], dim=0)
    col = torch.cat([dst, src], dim=0)

    if add_self_loop:
        diag = torch.arange(num_nodes, device=device, dtype=torch.long)
        row = torch.cat([row, diag], dim=0)
        col = torch.cat([col, diag], dim=0)

    indices = torch.stack([row, col], dim=0)
    values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)

    # Coalesce duplicate KG edges, then make the topology binary again.
    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()

    indices = adjacency.indices()
    values = torch.ones_like(adjacency.values())

    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.index_add_(0, indices[0], values)

    deg_inv_sqrt = deg.clamp_min(1.0).rsqrt()
    norm_values = (
        deg_inv_sqrt[indices[0]]
        * values
        * deg_inv_sqrt[indices[1]]
    )

    transition = torch.sparse_coo_tensor(
        indices,
        norm_values,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()

    return transition


@torch.no_grad()
def compute_ppr_sparse(
    num_nodes,
    train_triples,
    c=0.85,
    epsilon=0.0,
    add_self_loop=False,
    use_iterative=True,
    num_iter=30,
    topk=50,
    device=None,
    batch_size=128,
):
    """
    Compute a truncated Personalized PageRank auxiliary graph ON CUDA.

    Power iteration for seed i:
        s_i <- c M s_i + (1-c) e_i

    where M = D^{-1/2} A D^{-1/2} and A is the undirected KG topology.

    Memory strategy
    ---------------
    We never materialize an N x N dense PPR matrix. Seeds are processed in
    blocks. For a block of B seeds the largest dense state is N x B and all
    sparse-dense multiplications happen on GPU through torch.sparse.mm.

    Top-50 semantics
    ----------------
    During every power-iteration step we retain the center plus only its top-k
    non-self states. At the end, the center itself is removed and the final
    auxiliary graph keeps at most `topk` strongest PPR neighbours for EACH node.
    With the default topk=50 this means <= 50 PPR neighbours per center node.

    Edge orientation is deliberately:
        neighbour -> center
    so DGL message passing aggregates each center node FROM its own top-PPR
    neighbours. This fixes the reversed aggregation direction in older code.

    Parameters
    ----------
    num_nodes : int
    train_triples : array-like or Tensor [E, >=3]
    c : float
        Walk/damping probability. Restart probability is (1-c).
    epsilon : float
        Final minimum PPR weight. Keep 0.0 if you want pure top-k behavior.
    add_self_loop : bool
        Usually False because PPR already has an explicit restart-to-self term.
    use_iterative : bool
        Kept only for API compatibility. CUDA implementation is always iterative.
    num_iter : int
        Number of power iterations.
    topk : int
        Maximum number of non-self PPR neighbours kept per center node.
    device : torch.device or str, optional
        CUDA device. Defaults to current CUDA device.
    batch_size : int
        Number of PPR seed nodes processed together. Lower this if GPU memory is
        limited; increase it for more throughput on large-memory GPUs.

    Returns
    -------
    ppr_graph : dgl.DGLGraph on CUDA
    edge_weight : FloatTensor [E_ppr] on CUDA
    """
    del use_iterative  # compatibility argument; this implementation is iterative.

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA PPR was requested but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)

    if device.type != "cuda":
        raise ValueError(
            "compute_ppr_sparse in this model is GPU-only. Pass a CUDA device, "
            "for example device='cuda:0'."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current PyTorch runtime.")
    if not (0.0 < float(c) < 1.0):
        raise ValueError("c must satisfy 0 < c < 1.")
    if int(topk) <= 0:
        raise ValueError("topk must be a positive integer.")
    if int(num_iter) <= 0:
        raise ValueError("num_iter must be a positive integer.")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")

    num_nodes = int(num_nodes)
    topk = min(int(topk), max(num_nodes - 1, 1))
    batch_size = int(batch_size)

    M = _build_normalized_sparse_adjacency_cuda(
        num_nodes,
        train_triples,
        device=device,
        add_self_loop=add_self_loop,
    )

    all_src = []  # PPR neighbour -> center
    all_dst = []
    all_weight = []

    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        seeds = torch.arange(start, end, device=device, dtype=torch.long)
        bsz = seeds.numel()
        col_ids = torch.arange(bsz, device=device, dtype=torch.long)

        # state[:, j] is the PPR state for seed seeds[j].
        state = torch.zeros(
            (num_nodes, bsz),
            device=device,
            dtype=torch.float32,
        )
        state[seeds, col_ids] = 1.0

        for _ in range(int(num_iter)):
            next_state = float(c) * torch.sparse.mm(M, state)
            next_state[seeds, col_ids] += (1.0 - float(c))

            # Exclude self from neighbour top-k, but keep it separately because
            # the restart mass must survive to the next PPR iteration.
            self_values = next_state[seeds, col_ids].clone()
            next_state_t = next_state.transpose(0, 1)
            next_state_t[col_ids, seeds] = -torch.inf

            keep_values, keep_nodes = torch.topk(
                next_state_t,
                k=topk,
                dim=1,
                largest=True,
                sorted=False,
            )

            # Reuse the old dense state buffer instead of allocating a third
            # N x B matrix. Only center + top-k neighbours remain non-zero.
            state.zero_()
            state[seeds, col_ids] = self_values
            state.scatter_(
                0,
                keep_nodes.transpose(0, 1),
                keep_values.transpose(0, 1),
            )

        # Final exact top-k from the truncated PPR state, excluding self.
        state_t = state.transpose(0, 1)
        state_t[col_ids, seeds] = -torch.inf
        values, neighbours = torch.topk(
            state_t,
            k=topk,
            dim=1,
            largest=True,
            sorted=True,
        )

        valid = torch.isfinite(values) & (values > float(epsilon))

        centers = seeds[:, None].expand_as(neighbours)

        # DGL edge direction neighbour -> center so center receives the message.
        all_src.append(neighbours[valid])
        all_dst.append(centers[valid])
        all_weight.append(values[valid])

    if len(all_weight) == 0:
        empty_idx = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        ppr_graph = dgl.graph(
            (empty_idx, empty_idx),
            num_nodes=num_nodes,
            device=device,
        )
        return ppr_graph, empty_w

    ppr_src = torch.cat(all_src, dim=0).long()
    ppr_dst = torch.cat(all_dst, dim=0).long()
    edge_weight = torch.cat(all_weight, dim=0).float()

    ppr_graph = dgl.graph(
        (ppr_src, ppr_dst),
        num_nodes=num_nodes,
        device=device,
    )

    return ppr_graph, edge_weight


# ============================================================================
# Semantic feature encoders
# ============================================================================

class PoincareLogMap0(nn.Module):
    """Optional log-map from the unit Poincare ball to tangent space at 0."""

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
        max_norm = 1.0 - self.eps

        scale = torch.clamp(
            max_norm / norm.clamp_min(self.eps),
            max=1.0,
        )
        x_ball = x * scale

        ball_norm = torch.linalg.vector_norm(
            x_ball,
            dim=-1,
            keepdim=True,
        ).clamp(max=max_norm)

        factor = torch.atanh(ball_norm) / ball_norm.clamp_min(self.eps)
        factor = torch.where(
            ball_norm > self.eps,
            factor,
            torch.ones_like(factor),
        )
        return x_ball * factor


class MLPProjector(nn.Module):
    """
    Lightweight projector for the medical/domain embedding.

    model.py does not apply BatchNorm/Dropout on the domain branch, so this
    projector intentionally contains neither. LayerNorm is kept because it is
    deterministic normalization and does not inject batch statistics/noise.
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.0):
        super().__init__()
        del dropout  # API compatibility only; no extra dropout here.
        self.in_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.in_norm(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.out_norm(x)


class TextEmbeddingAutoencoder(nn.Module):
    """
    Keep the BatchNorm + Dropout pattern from model.py ONLY on the text branch.

    The decoder is retained for API compatibility with model.py. The main model
    uses the encoded representation, exactly as in the original code path.
    """

    def __init__(self, input_dim, encoding_dim, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.BatchNorm1d(encoding_dim * 2),
            nn.ReLU(True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU(True),
        )

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class FusionGate(nn.Module):
    """
    Fully learnable feature-wise fusion.

    There is NO fixed weighted-average prior, NO added 0.5, and NO `w` term in
    the fusion equation. The gate is predicted directly from the two semantic
    views of the same node:

        g = sigmoid(MLP([a, b, |a-b|, a*b]))
        fused = g*a + (1-g)*b

    The final gate layer uses Xavier initialization rather than zero
    initialization, so the model is not forced to start from g=0.5 either.
    """

    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        del dropout  # API compatibility only; model.py has no dropout here.
        self.left_norm = nn.LayerNorm(hidden_dim)
        self.right_norm = nn.LayerNorm(hidden_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

        nn.init.xavier_uniform_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, left_x, right_x, w=None):
        # `w` is accepted only so old callers do not break; it is intentionally
        # ignored and has zero effect on the fusion result.
        del w

        left_x = self.left_norm(left_x)
        right_x = self.right_norm(right_x)

        gate_features = torch.cat(
            [
                left_x,
                right_x,
                torch.abs(left_x - right_x),
                left_x * right_x,
            ],
            dim=-1,
        )

        gate = torch.sigmoid(self.gate_net(gate_features))
        fused = gate * left_x + (1.0 - gate) * right_x
        return self.out_norm(fused)


class EmbeddingLayer(nn.Module):
    """
    Initial node representation from:
        node-name/text embedding + medical/domain-knowledge embedding.

    BatchNorm/Dropout policy:
    - Text branch: keeps the same Autoencoder BatchNorm + Dropout as model.py.
    - Domain branch: no BatchNorm/Dropout, matching model.py's domain path.
    - Fusion/output: no extra BatchNorm/Dropout.

    `w` remains in the constructor ONLY for compatibility with existing main.py;
    it is no longer used as a fixed weighted-average prior.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        pretrained_text_embeddings,
        pretrained_domain_embeddings,
        freeze=False,
        w=None,
        dropout=0.0,
        domain_is_poincare=False,
    ):
        super().__init__()
        self.w = w  # compatibility only; FusionGate ignores it.
        self.hidden_dim = hidden_dim
        self.last_alignment_loss = None

        # Medical/domain view -------------------------------------------------
        if pretrained_domain_embeddings is not None:
            domain_tensor = torch.as_tensor(
                pretrained_domain_embeddings,
                dtype=torch.float32,
            )
            self.domain_embeddings = nn.Embedding.from_pretrained(
                domain_tensor,
                freeze=freeze,
            )
            domain_input_dim = domain_tensor.shape[1]
            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            self.domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.domain_embeddings.weight)
            domain_input_dim = hidden_dim
            print("Initialized random domain embeddings.")

        self.domain_logmap = (
            PoincareLogMap0() if domain_is_poincare else nn.Identity()
        )
        self.domain_projector = MLPProjector(
            domain_input_dim,
            hidden_dim,
        )

        # Text/name view ------------------------------------------------------
        if pretrained_text_embeddings is not None:
            text_tensor = torch.as_tensor(
                pretrained_text_embeddings,
                dtype=torch.float32,
            )
            self.text_embeddings = nn.Embedding.from_pretrained(
                text_tensor,
                freeze=freeze,
            )
            text_input_dim = text_tensor.shape[1]
            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            self.text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.text_embeddings.weight)
            text_input_dim = hidden_dim
            print("Initialized random text embeddings.")

        # This is the ONLY explicit BN/Dropout feature encoder, mirroring model.py.
        # Exactly like model.py: TextEmbeddingAutoencoder keeps its own
        # dropout_rate=0.2; R-GCN dropout is a separate hyperparameter.
        self.text_autoencoder = TextEmbeddingAutoencoder(
            text_input_dim,
            hidden_dim,
        )

        self.fusion_gate = FusionGate(hidden_dim)

    @staticmethod
    def _semantic_alignment_loss(text_x, domain_x):
        """
        Weak positive-pair alignment between the two semantic views.
        """
        text_n = F.normalize(text_x, p=2, dim=-1)
        domain_n = F.normalize(domain_x, p=2, dim=-1)
        return (1.0 - (text_n * domain_n).sum(dim=-1)).mean()

    def forward(self, graph, node_ids, rel_ids, norm):
        del graph, rel_ids, norm
        node_ids = node_ids.reshape(-1).long()

        raw_text = self.text_embeddings(node_ids)
        text_x, _ = self.text_autoencoder(raw_text)

        raw_domain = self.domain_embeddings(node_ids)
        raw_domain = self.domain_logmap(raw_domain)
        domain_x = self.domain_projector(raw_domain)

        self.last_alignment_loss = self._semantic_alignment_loss(
            text_x,
            domain_x,
        )

        # No fixed 0.5 and no w-prior: gate is learned entirely from data.
        return self.fusion_gate(text_x, domain_x)


# ============================================================================
# Relation-aware graph encoder
# ============================================================================

class LukePiRelGraphConvBlock(nn.Module):
    """
    R-GCN layer kept deliberately close to model.py, with one small
    LukePi-inspired improvement: a learnable residual mixing coefficient.

    Let h_new be the output of RelGraphConv and x be the representation from
    the previous layer.  The block returns

        alpha * h_new + (1 - alpha) * x

    where alpha = sigmoid(alpha_logit) is learned separately for each R-GCN
    layer.  This lets training decide how much relation-aware graph information
    should replace the previous semantic representation.

    Important: unlike V3, this block has no added LayerNorm, FFN, GELU,
    graph_scale, ffn_scale, or extra Dropout.  Dropout exists only inside
    RelGraphConv, exactly as in model.py.
    """

    def __init__(
        self,
        hidden_dim,
        num_rels,
        num_bases=None,
        dropout=0.0,
        use_self_loop=True,
        activation=None,
    ):
        super().__init__()

        self.conv = RelGraphConv(
            in_feat=hidden_dim,
            out_feat=hidden_dim,
            num_rels=num_rels,
            regularizer="bdd",
            num_bases=num_bases,
            activation=activation,
            self_loop=use_self_loop,
            dropout=dropout,
        )

        # One learnable scalar per R-GCN layer.  Zero logit means a neutral
        # starting point; it is NOT a fixed 0.5 weighted average.
        self.alpha_logit = nn.Parameter(torch.zeros(1))

    def forward(self, graph, x, rel_ids, norm):
        h_new = self.conv(graph, x, rel_ids, norm)
        alpha = torch.sigmoid(self.alpha_logit)
        return alpha * h_new + (1.0 - alpha) * x

    @property
    def alpha(self):
        """Current learned residual weight in [0, 1], useful for logging."""
        return torch.sigmoid(self.alpha_logit)


# Backward-compatible alias in case any external code imports the old name.
ResidualRelGraphConvBlock = LukePiRelGraphConvBlock


class BaseRGCN(nn.Module):
    """
    R-GCN base kept close to model.py.

    V4 intentionally removes V3's Jumping Knowledge and final LayerNorm.  The
    input embedding is followed sequentially by R-GCN layers, and the output of
    the last layer is the R-GCN representation.
    """

    def __init__(
        self,
        num_nodes,
        hidden_dim,
        output_dim,
        num_relations,
        num_bases=-1,
        num_hidden_layers=1,
        dropout=0.0,
        use_self_loop=False,
        use_cuda=False,
        pretrained_text_embeddings=None,
        pretrained_domain_embeddings=None,
        freeze=False,
        w=None,
        domain_is_poincare=False,
    ):
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
        self.domain_is_poincare = domain_is_poincare

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

    @property
    def input_encoder(self):
        return self.layers[0]

    def forward(self, graph, node_ids, rel_ids, norm):
        h = self.input_encoder(graph, node_ids, rel_ids, norm)

        # Same sequential structure as model.py; the only added behavior lives
        # inside LukePiRelGraphConvBlock's learnable residual alpha.
        for layer in self.layers[1:]:
            h = layer(graph, h, rel_ids, norm)

        return h


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(
            self.num_nodes,
            self.hidden_dim,
            self.pretrained_text_embeddings,
            self.pretrained_domain_embeddings,
            freeze=self.freeze,
            w=self.w,
            dropout=self.dropout,
            domain_is_poincare=self.domain_is_poincare,
        )

    def build_hidden_layer(self, idx):
        # Match model.py: ReLU on every R-GCN layer except the last one.
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return LukePiRelGraphConvBlock(
            hidden_dim=self.hidden_dim,
            num_rels=self.num_relations,
            num_bases=self.num_bases,
            dropout=self.dropout,
            use_self_loop=self.use_self_loop,
            activation=activation,
        )

    def encode_input(self, node_ids):
        return self.input_encoder(None, node_ids, None, None)


# ============================================================================
# PPR auxiliary branch
# ============================================================================

class PPRGraphConv(nn.Module):
    """
    Weighted residual propagation on the relation-agnostic PPR graph.

    PPR edges are neighbour -> center. No BatchNorm/Dropout is added here.
    """

    def __init__(self, hidden_dim, dropout=0.0, activation=True):
        super().__init__()
        del dropout  # API compatibility only; no extra dropout in this branch.
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.activation = activation
        self.message_scale = nn.Parameter(torch.tensor(0.25))

    def forward(self, g, x, edge_weight):
        if edge_weight is None:
            raise ValueError("ppr_edge_weight is required when PPR is enabled.")
        if g.device != x.device:
            raise RuntimeError(
                f"PPR graph is on {g.device}, but node features are on {x.device}. "
                "Create PPR with compute_ppr_sparse(..., device=x.device)."
            )

        with g.local_scope():
            h = self.linear(self.norm(x))
            g.ndata["h"] = h
            g.edata["w"] = edge_weight.reshape(-1, 1).to(
                device=x.device,
                dtype=x.dtype,
            )

            g.update_all(
                fn.u_mul_e("h", "w", "m"),
                fn.sum("m", "h_sum"),
            )
            g.update_all(
                fn.copy_e("w", "m_w"),
                fn.sum("m_w", "w_sum"),
            )

            msg = g.ndata["h_sum"] / g.ndata["w_sum"].clamp_min(1e-6)

        if self.activation:
            msg = F.gelu(msg)

        return self.out_norm(x + self.message_scale * msg)


class PPRBranch(nn.Module):
    def __init__(self, hidden_dim, num_layers=1, dropout=0.0):
        super().__init__()
        del dropout  # API compatibility only.
        num_layers = max(int(num_layers), 1)
        self.layers = nn.ModuleList(
            [
                PPRGraphConv(
                    hidden_dim,
                    activation=(i < num_layers - 1),
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, ppr_graph, x, edge_weight):
        h = x
        for layer in self.layers:
            h = layer(ppr_graph, h, edge_weight)
        return h


# ============================================================================
# Link prediction / training wrapper
# ============================================================================

class LinkPredict(nn.Module):
    """
    Main pipeline:
        text/name embedding
                         \\
                          gated semantic fusion -> relation-aware R-GCN -> node z
                         /
        medical embedding

    plus a conservative CUDA-PPR structural branch.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_relations,
        num_bases=-1,
        num_hidden_layers=1,
        dropout=0.0,
        use_cuda=False,
        regularization_param=0.0,
        pretrained_text_embeddings=None,
        pretrained_domain_embeddings=None,
        pretrained_relation_embeddings=None,
        freeze=False,
        w=None,
        use_ppr=True,
        ppr_num_layers=1,
        domain_is_poincare=False,
        ppr_prior=0.10,
        semantic_alignment_weight=0.01,
    ):
        super().__init__()

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
            w=w,
            domain_is_poincare=domain_is_poincare,
        )

        self.use_ppr = use_ppr
        self.ppr_prior = float(ppr_prior)  # compatibility only; not used by FusionGate
        self.semantic_alignment_weight = float(semantic_alignment_weight)
        self.regularization_param = regularization_param
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self._last_semantic_alignment_loss = None

        if use_ppr:
            self.ppr_branch = PPRBranch(
                hidden_dim,
                num_layers=ppr_num_layers,
                dropout=dropout,
            )
            self.ppr_fusion = FusionGate(hidden_dim)

        if pretrained_relation_embeddings is not None:
            relation_tensor = torch.as_tensor(
                pretrained_relation_embeddings,
                dtype=torch.float32,
            )
            if relation_tensor.ndim != 2:
                raise ValueError(
                    "pretrained_relation_embeddings must have shape "
                    "[num_relations, hidden_dim]."
                )
            if relation_tensor.shape != (num_relations, hidden_dim):
                raise ValueError(
                    "pretrained_relation_embeddings shape must equal "
                    f"({num_relations}, {hidden_dim}), got "
                    f"{tuple(relation_tensor.shape)}."
                )
            self.relation_weights = nn.Parameter(relation_tensor.clone())
            print("Loaded pretrained relation embeddings.")
        else:
            self.relation_weights = nn.Parameter(
                torch.empty(num_relations, hidden_dim)
            )
            nn.init.xavier_uniform_(self.relation_weights)
            print("Initialized random relation embeddings.")

    def calculate_score(self, embeddings, triplets):
        """DistMult retained for compatibility with the current evaluators."""
        subject = embeddings[triplets[:, 0]]
        relation = self.relation_weights[triplets[:, 1]]
        obj = embeddings[triplets[:, 2]]
        return torch.sum(subject * relation * obj, dim=1)

    def forward(
        self,
        graph,
        node_ids,
        rel_ids,
        norm,
        ppr_graph=None,
        ppr_edge_weight=None,
        all_node_ids=None,
    ):
        h_rgcn = self.rgcn(graph, node_ids, rel_ids, norm)

        # Save the alignment term from THIS training subgraph before encode_input
        # is called again for all nodes by the optional PPR branch.
        self._last_semantic_alignment_loss = (
            self.rgcn.input_encoder.last_alignment_loss
        )

        ppr_ready = (
            self.use_ppr
            and ppr_graph is not None
            and ppr_edge_weight is not None
            and all_node_ids is not None
        )
        if not ppr_ready:
            return h_rgcn

        x_full = self.rgcn.encode_input(all_node_ids)
        h_ppr_full = self.ppr_branch(
            ppr_graph,
            x_full,
            ppr_edge_weight,
        )

        ids = node_ids.reshape(-1).long()
        h_ppr = h_ppr_full[ids]

        # No fixed PPR/RGCN ratio: fusion is learned directly from data.
        return self.ppr_fusion(h_ppr, h_rgcn)

    def regularization_loss(self, embeddings):
        return (
            torch.mean(embeddings.pow(2))
            + torch.mean(self.relation_weights.pow(2))
        )

    def get_loss(self, graph, embeddings, triplets, labels):
        del graph
        score = self.calculate_score(embeddings, triplets)
        labels = labels.to(dtype=score.dtype)

        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)

        loss = prediction_loss + self.regularization_param * reg_loss

        if (
            self.semantic_alignment_weight > 0.0
            and self._last_semantic_alignment_loss is not None
        ):
            loss = (
                loss
                + self.semantic_alignment_weight
                * self._last_semantic_alignment_loss
            )

        return loss

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path,
        device=None,
        strict=True,
        **init_kwargs,
    ):
        if device is None:
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        model = cls(**init_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=strict)
        model.to(device)
        return model, checkpoint.get("iteration", None)

    def get_final_embeddings(
        self,
        graph,
        node_ids,
        rel_ids,
        norm,
        ppr_graph=None,
        ppr_edge_weight=None,
        all_node_ids=None,
    ):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(
                graph,
                node_ids,
                rel_ids,
                norm,
                ppr_graph=ppr_graph,
                ppr_edge_weight=ppr_edge_weight,
                all_node_ids=all_node_ids,
            )
        if was_training:
            self.train()
        return embeddings

    def evaluate_auroc(
        self,
        embeddings,
        test_triplets,
        total_data,
        batch_size=4096,
        seed=None,
        return_pairs=False,
    ):
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

    def evaluate_auroc_on_graph(
        self,
        graph,
        node_ids,
        rel_ids,
        norm,
        test_triplets,
        total_data,
        ppr_graph=None,
        ppr_edge_weight=None,
        all_node_ids=None,
        batch_size=4096,
        seed=None,
        return_pairs=False,
    ):
        embeddings = self.get_final_embeddings(
            graph,
            node_ids,
            rel_ids,
            norm,
            ppr_graph=ppr_graph,
            ppr_edge_weight=ppr_edge_weight,
            all_node_ids=all_node_ids,
        )
        return self.evaluate_auroc(
            embeddings,
            test_triplets,
            total_data,
            batch_size=batch_size,
            seed=seed,
            return_pairs=return_pairs,
        )
