"""
model.py with optional model_base4 modules.

All flags default to False. With every flag off, this file is the model.py
pipeline: min-max text/domain embeddings, TextEmbeddingAutoencoder, a linear
domain map, fixed weighted fusion, plain RelGraphConv, and DistMult.

Turn one flag on to add exactly one model_base4 module:

- use_projector: MLPProjector instead of Linear on the domain view
- use_adaptive_fusion: FusionGate instead of (1-w)*domain + w*text
- use_node_id: residual trainable node-id embedding after fusion
- use_residual_rgcn: ResidualRelGraphConvBlock instead of RelGraphConv
- use_jumping_knowledge: mean of hidden-layer states (needs num_hidden_layers > 1)
- use_ppr: PPR auxiliary branch fused with the R-GCN output

Min-max normalization and relation-embedding normalization stay as in model.py
on every variant. LinkPredict still passes use_cuda into RGCN in the same
positional slot as model.py, so the baseline self-loop setting matches model.py.
"""
import sys
from pathlib import Path

import dgl
import dgl.function as fn
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

_COMPARE_DIR = Path(__file__).resolve().parent
if str(_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPARE_DIR))
from fusion_variants import build_fusion  # noqa: E402


@torch.no_grad()
def _build_normalized_sparse_adjacency_cuda(
    num_nodes,
    train_triples,
    device,
    add_self_loop=True,
):
    """Build M = D^{-1/2} A D^{-1/2} on CUDA as sparse COO."""
    triples = torch.as_tensor(train_triples, dtype=torch.long, device=device)
    if triples.ndim != 2 or triples.shape[1] < 3:
        raise ValueError("train_triples must have shape [E, 3] or wider.")

    src = triples[:, 0]
    dst = triples[:, 2]
    row = torch.cat([src, dst], dim=0)
    col = torch.cat([dst, src], dim=0)

    if add_self_loop:
        diag = torch.arange(num_nodes, device=device, dtype=torch.long)
        row = torch.cat([row, diag], dim=0)
        col = torch.cat([col, diag], dim=0)

    indices = torch.stack([row, col], dim=0)
    values = torch.ones(indices.shape[1], device=device, dtype=torch.float32)
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
    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices,
        norm_values,
        size=(num_nodes, num_nodes),
        device=device,
        dtype=torch.float32,
    ).coalesce()


@torch.no_grad()
def compute_ppr_sparse(num_nodes, train_triples, c=0.15, epsilon=1e-4,
                        add_self_loop=True, use_iterative=True, num_iter=50,
                        topk=50, device=None, batch_size=128):
    """
    CUDA blockwise PPR for the model_base4 auxiliary graph.

    Power iteration, same recurrence as the previous CPU implementation:
        s <- c M s + (1 - c) e
    M = D^{-1/2} A D^{-1/2}. Seeds are processed in blocks so the dense state
    is N x batch_size, not N x N. Each center keeps at most `topk` neighbours.
    Edge direction stays center -> neighbour, matching model_base4.
    """
    del use_iterative

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA PPR was requested but torch.cuda.is_available() is False."
            )
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("compute_ppr_sparse runs on CUDA. Pass device='cuda'.")
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

    all_src = []
    all_dst = []
    all_weight = []
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        seeds = torch.arange(start, end, device=device, dtype=torch.long)
        bsz = seeds.numel()
        col_ids = torch.arange(bsz, device=device, dtype=torch.long)
        state = torch.zeros((num_nodes, bsz), device=device, dtype=torch.float32)
        state[seeds, col_ids] = 1.0

        for _ in range(int(num_iter)):
            next_state = float(c) * torch.sparse.mm(M, state)
            next_state[seeds, col_ids] += (1.0 - float(c))
            self_values = next_state[seeds, col_ids].clone()
            next_state_t = next_state.transpose(0, 1)
            next_state_t[col_ids, seeds] = -torch.inf
            keep_values, keep_nodes = torch.topk(
                next_state_t, k=topk, dim=1, largest=True, sorted=False,
            )
            state.zero_()
            state[seeds, col_ids] = self_values
            state.scatter_(
                0,
                keep_nodes.transpose(0, 1),
                keep_values.transpose(0, 1),
            )

        state_t = state.transpose(0, 1)
        state_t[col_ids, seeds] = -torch.inf
        values, neighbours = torch.topk(
            state_t, k=topk, dim=1, largest=True, sorted=True,
        )
        valid = torch.isfinite(values) & (values.abs() >= float(epsilon))
        centers = seeds[:, None].expand_as(neighbours)
        # center -> neighbour, same direction as the CPU model_base4 graph.
        all_src.append(centers[valid])
        all_dst.append(neighbours[valid])
        all_weight.append(values[valid])

    if len(all_weight) == 0:
        empty_idx = torch.empty(0, device=device, dtype=torch.long)
        empty_w = torch.empty(0, device=device, dtype=torch.float32)
        return dgl.graph((empty_idx, empty_idx), num_nodes=num_nodes, device=device), empty_w

    ppr_src = torch.cat(all_src, dim=0).long()
    ppr_dst = torch.cat(all_dst, dim=0).long()
    edge_weight = torch.cat(all_weight, dim=0).float()
    ppr_graph = dgl.graph((ppr_src, ppr_dst), num_nodes=num_nodes, device=device)
    return ppr_graph, edge_weight


class PPRGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_weight):
        with g.local_scope():
            g.ndata["h"] = self.linear(x)
            g.edata["w"] = edge_weight.view(-1, 1)
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h_new"))
            h = g.ndata["h_new"]
        h = self.norm(h)
        h = self.act(h)
        return self.dropout(h)


class PPRBranch(nn.Module):
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


class TextEmbeddingAutoencoder(nn.Module):
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
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        # sigmoid(log(0.75 / 0.25)) = 0.75 while the weight is still zero.
        nn.init.constant_(self.gate.bias, torch.log(torch.tensor(0.75 / 0.25)).item())

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        fused = gate * text_x + (1.0 - gate) * domain_x
        return fused


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, hidden_dim, pretrained_text_embeddings,
                 pretrained_domain_embeddings, freeze=False, w=0.5, dropout=0.2,
                 use_projector=False, use_adaptive_fusion=False, use_node_id=False,
                 fusion_variant="node_channel"):
        super(EmbeddingLayer, self).__init__()
        self.w = w
        self.use_projector = use_projector
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_node_id = use_node_id
        self.fusion_variant = fusion_variant

        if pretrained_domain_embeddings is not None:
            domain_embeddings = torch.from_numpy(pretrained_domain_embeddings).float()
            norm_domain_embeddings = (domain_embeddings - domain_embeddings.min()) / (
                domain_embeddings.max() - domain_embeddings.min()
            )
            domain_dim = pretrained_domain_embeddings.shape[1]
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(
                norm_domain_embeddings, freeze=freeze
            )
            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            domain_dim = hidden_dim
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            print("Initialized random domain embeddings.")

        if use_projector:
            self.domain_projector = MLPProjector(domain_dim, hidden_dim, dropout)
        else:
            self.poincare_to_euclidean = nn.Linear(domain_dim, hidden_dim)

        if pretrained_text_embeddings is not None:
            text_embeddings = torch.from_numpy(pretrained_text_embeddings).float()
            norm_text_embeddings = (text_embeddings - text_embeddings.min()) / (
                text_embeddings.max() - text_embeddings.min()
            )
            text_dim = pretrained_text_embeddings.shape[1]
            self.norm_text_embeddings = nn.Embedding.from_pretrained(
                norm_text_embeddings, freeze=freeze
            )
            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            text_dim = hidden_dim
            self.norm_text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            print("Initialized random text embeddings.")

        # Autoencoder dropout stays at the model.py default (0.2).
        self.autoencoder = TextEmbeddingAutoencoder(text_dim, hidden_dim)

        if use_adaptive_fusion:
            self.fusion_gate = build_fusion(fusion_variant, hidden_dim, dropout)
        if use_node_id:
            self.node_id_embedding = nn.Embedding(num_nodes, hidden_dim)
            nn.init.xavier_uniform_(self.node_id_embedding.weight)

    def forward(self, graph, node_ids, rel_ids, norm):
        ids = node_ids.squeeze()
        text_x, _ = self.autoencoder(self.norm_text_embeddings(ids))
        raw_domain = self.norm_domain_embeddings(ids)
        if self.use_projector:
            domain_x = self.domain_projector(raw_domain)
        else:
            domain_x = self.poincare_to_euclidean(raw_domain)

        if self.use_adaptive_fusion:
            fused = self.fusion_gate(text_x, domain_x, self.w)
        else:
            fused = (1 - self.w) * domain_x + self.w * text_x

        if self.use_node_id:
            fused = fused + self.node_id_embedding(ids)
        return fused


class ResidualRelGraphConvBlock(nn.Module):
    def __init__(self, hidden_dim, num_rels, num_bases=None,
                 dropout=0.2, use_self_loop=False, activation=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv = RelGraphConv(
            in_feat=hidden_dim,
            out_feat=hidden_dim,
            num_rels=num_rels,
            regularizer="bdd",
            num_bases=num_bases,
            activation=None,
            self_loop=use_self_loop,
            dropout=0.0,
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


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 freeze=False, w=0.5, use_projector=False, use_adaptive_fusion=False,
                 use_node_id=False, use_residual_rgcn=False, use_jumping_knowledge=False,
                 fusion_variant="node_channel"):
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
        self.use_projector = use_projector
        self.use_adaptive_fusion = use_adaptive_fusion
        self.use_node_id = use_node_id
        self.use_residual_rgcn = use_residual_rgcn
        self.use_jumping_knowledge = use_jumping_knowledge
        self.fusion_variant = fusion_variant
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
        if not self.use_jumping_knowledge:
            for layer in self.layers:
                node_ids = layer(graph, node_ids, rel_ids, norm)
            return node_ids

        h = node_ids
        layer_outputs = []
        for idx, layer in enumerate(self.layers):
            h = layer(graph, h, rel_ids, norm)
            if idx != 0:
                layer_outputs.append(h)
        if len(layer_outputs) > 1:
            h = torch.mean(torch.stack(layer_outputs, dim=0), dim=0)
        return h


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(
            self.num_nodes,
            self.hidden_dim,
            self.pretrained_text_embeddings,
            self.pretrained_domain_embeddings,
            self.freeze,
            self.w,
            dropout=self.dropout,
            use_projector=self.use_projector,
            use_adaptive_fusion=self.use_adaptive_fusion,
            use_node_id=self.use_node_id,
            fusion_variant=self.fusion_variant,
        )

    def build_hidden_layer(self, idx):
        if self.use_residual_rgcn:
            return ResidualRelGraphConvBlock(
                hidden_dim=self.hidden_dim,
                num_rels=self.num_relations,
                num_bases=self.num_bases,
                dropout=self.dropout,
                use_self_loop=self.use_self_loop,
                activation=idx < self.num_hidden_layers - 1,
            )
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(
            in_feat=self.hidden_dim,
            out_feat=self.hidden_dim,
            num_rels=self.num_relations,
            regularizer="bdd",
            num_bases=self.num_bases,
            activation=activation,
            self_loop=self.use_self_loop,
            dropout=self.dropout,
        )

    def encode_input(self, node_ids):
        return self.layers[0](None, node_ids, None, None)


class LinkPredict(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                 num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
                 pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                 pretrained_relation_embeddings=None, freeze=False, w=0.5,
                 use_projector=False, use_adaptive_fusion=False, use_node_id=False,
                 use_residual_rgcn=False, use_jumping_knowledge=False,
                 use_ppr=False, ppr_num_layers=2, fusion_variant="node_channel"):
        super(LinkPredict, self).__init__()
        # Same positional call as model.py: use_cuda lands in RGCN.use_self_loop.
        self.rgcn = RGCN(
            input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
            num_hidden_layers, dropout, use_cuda,
            pretrained_text_embeddings=pretrained_text_embeddings,
            pretrained_domain_embeddings=pretrained_domain_embeddings,
            freeze=freeze, w=w,
            use_projector=use_projector,
            use_adaptive_fusion=use_adaptive_fusion,
            use_node_id=use_node_id,
            use_residual_rgcn=use_residual_rgcn,
            use_jumping_knowledge=use_jumping_knowledge,
            fusion_variant=fusion_variant,
        )
        self.use_ppr = use_ppr
        if use_ppr:
            self.ppr_branch = PPRBranch(hidden_dim, num_layers=ppr_num_layers, dropout=dropout)
            self.ppr_fusion = FusionGate(hidden_dim)

        self.regularization_param = regularization_param
        if pretrained_relation_embeddings is not None:
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            normalized_relations = (self.relation_weights - self.relation_weights.min()) / (
                self.relation_weights.max() - self.relation_weights.min()
            )
            self.relation_weights.data.copy_(normalized_relations)
            print("Loaded pretrained relation embeddings.")
        else:
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain("relu"))
            print("Initialized random relation embeddings.")

    def calculate_score(self, embeddings, triplets):
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        return torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)

    def forward(self, graph, node_ids, rel_ids, norm,
                ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
        h_rgcn = self.rgcn(graph, node_ids, rel_ids, norm)
        if not (self.use_ppr and ppr_graph is not None and all_node_ids is not None):
            return h_rgcn
        x_full = self.rgcn.encode_input(all_node_ids)
        h_ppr_full = self.ppr_branch(ppr_graph, x_full, ppr_edge_weight)
        h_ppr = h_ppr_full[node_ids.reshape(-1)]
        return self.ppr_fusion(h_rgcn, h_ppr)

    def regularization_loss(self, embeddings):
        return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))

    def get_loss(self, graph, embeddings, triplets, labels):
        score = self.calculate_score(embeddings, triplets)
        prediction_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embeddings)
        return prediction_loss + self.regularization_param * reg_loss

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device=None, **init_kwargs):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(**init_kwargs)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model, checkpoint.get("iteration", None)

    def get_final_embeddings(self, graph, node_ids, rel_ids, norm,
                             ppr_graph=None, ppr_edge_weight=None, all_node_ids=None):
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
