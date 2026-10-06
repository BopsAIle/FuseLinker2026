import sys
from pathlib import Path

import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

_COMPARE_DIR = Path(__file__).resolve().parent
if str(_COMPARE_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPARE_DIR))
from fusion_v1_v10 import build_fusion  # noqa: E402


# reduce dimensions by Autoencoder
# Đây là hàm giảm chiều text_embedding khi đi vào RGCN(dành cho text_embedding)
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
    """Domain MLP Projector. Replaces the linear map on domain embeddings."""

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


class BaseRGCN(nn.Module):
    """
    Base class for Relational Graph Convolutional Network (R-GCN) model.
    This class initializes the model and defines the base layers.
    """

    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False, pretrained_text_embeddings=None,
                pretrained_domain_embeddings=None, freeze=False, w=0.5,                 fusion_variant=None,
                use_domain_mlp_projector=False, use_residual_rgcn=False):
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
        self.fusion_variant = fusion_variant
        self.use_domain_mlp_projector = use_domain_mlp_projector
        self.use_residual_rgcn = use_residual_rgcn

        # Create RGCN layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # Input to hidden layer
        input_layer = self.build_input_layer()
        if input_layer is not None:
            self.layers.append(input_layer)
        # Hidden to hidden layers
        for idx in range(self.num_hidden_layers):
            hidden_layer = self.build_hidden_layer(idx)
            self.layers.append(hidden_layer)
        # Hidden to output layer (if necessary)
        output_layer = self.build_output_layer()
        if output_layer is not None:
            self.layers.append(output_layer)

    def build_input_layer(self):
        # Override in subclass
        return None

    def build_hidden_layer(self, idx):
        # Override in subclass
        raise NotImplementedError

    def build_output_layer(self):
        # Override in subclass
        return None

    def forward(self, graph, node_ids, rel_ids, norm):
        """
        Forward pass through the RGCN layers.
        """
        for layer in self.layers:
            node_ids = layer(graph, node_ids, rel_ids, norm)
        return node_ids


class EmbeddingLayer(nn.Module):
    """
    Embedding layer to initialize node features with two pretrained embeddings,
    one of which will be linearly transformed to match dimensions, and each is normalized before weighted averaging.
    """

    def __init__(self, num_nodes, hidden_dim, pretrained_text_embeddings, pretrained_domain_embeddings, freeze=False,w=0.5, fusion_variant=None, use_domain_mlp_projector=False, dropout=0.2):
        super(EmbeddingLayer, self).__init__()
        self.w = w
        self.fusion_variant = fusion_variant
        self.use_domain_mlp_projector = use_domain_mlp_projector
        if fusion_variant:
            self.fusion = build_fusion(fusion_variant, hidden_dim)
        # Pretrained domain embeddings
        if pretrained_domain_embeddings is not None:
            #Kích thước ma trận pretrained_domain_embeddings :(num_nodes, hidden_dim)
            domain_embeddings = torch.from_numpy(pretrained_domain_embeddings).float()
            #Cái norm_domain_embeddings đơn giản là đưa về khoảng [0,1]
            norm_domain_embeddings = (domain_embeddings - domain_embeddings.min()) / (
                    domain_embeddings.max() - domain_embeddings.min())

            domain_in = pretrained_domain_embeddings.shape[1]
            if use_domain_mlp_projector:
                self.domain_mlp_projector = MLPProjector(domain_in, hidden_dim, dropout)
            else:
                self.poincare_to_euclidean = nn.Linear(domain_in, hidden_dim)
            #Cái này để nói rằng chúng ta sẽ không coi embedding là tham số để học
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(norm_domain_embeddings, freeze=freeze)

            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
            print(
                "Domain map: MLPProjector."
                if use_domain_mlp_projector
                else "Domain map: Linear."
            )
        else:
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
            if use_domain_mlp_projector:
                self.domain_mlp_projector = MLPProjector(hidden_dim, hidden_dim, dropout)
            else:
                self.poincare_to_euclidean = nn.Linear(hidden_dim, hidden_dim)
            print("Initialized random domain embeddings.")

        # Pretrained text embeddings, which will be linearly transformed
        if pretrained_text_embeddings is not None:
            text_embeddings = torch.from_numpy(pretrained_text_embeddings).float()
            norm_text_embeddings = (text_embeddings - text_embeddings.min()) / (
                    text_embeddings.max() - text_embeddings.min())

            self.norm_text_embeddings = nn.Embedding.from_pretrained(norm_text_embeddings, freeze=freeze)

            self.autoencoder = TextEmbeddingAutoencoder(pretrained_text_embeddings.shape[1], hidden_dim)

            print(f"Loaded pretrained text embeddings, freeze is {freeze}.")
        else:
            self.norm_text_embeddings = nn.Embedding(num_nodes, hidden_dim)
            self.autoencoder = TextEmbeddingAutoencoder(hidden_dim, hidden_dim)
            print("Initialized random text embeddings.")

    def forward(self, graph, node_ids, rel_ids, norm):
        # Transform the text_embeddings to match the GCN embedding's dimensions
        transformed_text_embeddings, _ = self.autoencoder(self.norm_text_embeddings(node_ids.squeeze()))

        # Map Poincaré embeddings to Euclidean space
        raw_domain = self.norm_domain_embeddings(node_ids.squeeze())
        if self.use_domain_mlp_projector:
            transformed_domain_embeddings = self.domain_mlp_projector(raw_domain)
        else:
            transformed_domain_embeddings = self.poincare_to_euclidean(raw_domain)

        if self.fusion_variant:
            combined_embedding = self.fusion(
                transformed_text_embeddings, transformed_domain_embeddings
            )
        else:
            combined_embedding = (1 - self.w) * transformed_domain_embeddings + self.w * transformed_text_embeddings

        return combined_embedding

    def embed_nodes(self, node_ids):
        """Node features for the PPR branch. Skips the unused text decoder."""
        ids = node_ids.squeeze()
        text = self.norm_text_embeddings(ids)
        if self.training and text.requires_grad:
            transformed_text = checkpoint(
                self.autoencoder.encoder, text, use_reentrant=False,
            )
        else:
            transformed_text = self.autoencoder.encoder(text)
        raw_domain = self.norm_domain_embeddings(ids)
        if self.use_domain_mlp_projector:
            transformed_domain = self.domain_mlp_projector(raw_domain)
        else:
            transformed_domain = self.poincare_to_euclidean(raw_domain)
        if self.fusion_variant:
            return self.fusion(transformed_text, transformed_domain)
        return (1 - self.w) * transformed_domain + self.w * transformed_text


class ResidualRelGraphConvBlock(nn.Module):
    """Residual R-GCN. Replaces a plain RelGraphConv hidden layer."""

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


class RGCN(BaseRGCN):
    """
    Implementation of R-GCN with support for link prediction.
    """

    def build_input_layer(self):
        # Initialize node features with embedding layer
        return EmbeddingLayer(self.num_nodes, self.hidden_dim, self.pretrained_text_embeddings,
                            self.pretrained_domain_embeddings, self.freeze, self.w,
                            fusion_variant=self.fusion_variant,
                            use_domain_mlp_projector=self.use_domain_mlp_projector,
                            dropout=self.dropout)

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
        # Activation function for all but the last layer
        activation = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(in_feat=self.hidden_dim, #Số chiều đầu vào của node embedding
                            out_feat=self.hidden_dim, #Số chiều đầu ra của node embedding
                            num_rels=self.num_relations, #Số quan hệ trong đồ thị
                            regularizer='bdd', # Relation chỉ học tương tác cục bộ từng nhóm feature
                            num_bases=self.num_bases,
                            activation=activation,
                            self_loop=self.use_self_loop,
                            dropout=self.dropout)

    def encode_input(self, node_ids):
        return self.layers[0].embed_nodes(node_ids)


@torch.no_grad()
def _build_normalized_sparse_adjacency_cuda(
    num_nodes, train_triples, device, add_self_loop=True,
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
        indices, values, size=(num_nodes, num_nodes),
        device=device, dtype=torch.float32,
    ).coalesce()

    indices = adjacency.indices()
    values = torch.ones_like(adjacency.values())
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.index_add_(0, indices[0], values)
    deg_inv_sqrt = deg.clamp_min(1.0).rsqrt()
    norm_values = deg_inv_sqrt[indices[0]] * values * deg_inv_sqrt[indices[1]]
    return torch.sparse_coo_tensor(
        indices, norm_values, size=(num_nodes, num_nodes),
        device=device, dtype=torch.float32,
    ).coalesce()


@torch.no_grad()
def compute_ppr_sparse(num_nodes, train_triples, c=0.15, epsilon=1e-4,
                        add_self_loop=True, use_iterative=True, num_iter=50,
                        topk=50, device=None, batch_size=128):
    """CUDA blockwise Personalized PageRank. Each center keeps at most topk neighbours."""
    del use_iterative

    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PPR was requested but CUDA is not available.")
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
    adjacency = _build_normalized_sparse_adjacency_cuda(
        num_nodes, train_triples, device=device, add_self_loop=add_self_loop,
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
            next_state = float(c) * torch.sparse.mm(adjacency, state)
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
                0, keep_nodes.transpose(0, 1), keep_values.transpose(0, 1),
            )

        state_t = state.transpose(0, 1)
        state_t[col_ids, seeds] = -torch.inf
        values, neighbours = torch.topk(
            state_t, k=topk, dim=1, largest=True, sorted=True,
        )
        valid = torch.isfinite(values) & (values.abs() >= float(epsilon))
        centers = seeds[:, None].expand_as(neighbours)
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


class _WeightedSourceAggregate(torch.autograd.Function):
    """Sum_e w_e * h[src_e] into dst_e, in chunks so the message buffer stays small."""

    @staticmethod
    def forward(ctx, features, src, dst, weight):
        ctx.save_for_backward(features, src, dst, weight)
        out = torch.zeros_like(features)
        chunk = 16384
        for start in range(0, src.shape[0], chunk):
            end = min(start + chunk, src.shape[0])
            message = features[src[start:end]] * weight[start:end]
            out.index_add_(0, dst[start:end], message)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, src, dst, weight = ctx.saved_tensors
        grad_features = torch.zeros_like(features)
        chunk = 16384
        for start in range(0, src.shape[0], chunk):
            end = min(start + chunk, src.shape[0])
            message_grad = grad_out[dst[start:end]]
            grad_features.index_add_(
                0, src[start:end], message_grad * weight[start:end],
            )
        return grad_features, None, None, None


class PPRGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.ReLU() if activation else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x, edge_weight):
        features = self.linear(x)
        src, dst = g.edges()
        aggregated = _WeightedSourceAggregate.apply(
            features, src, dst, edge_weight.view(-1, 1),
        )
        aggregated = self.norm(aggregated)
        aggregated = self.act(aggregated)
        return self.dropout(aggregated)


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


class FusionGate(nn.Module):
    """Gate that mixes R-GCN states with the PPR branch. Zero weight starts at 0.75 / 0.25."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, torch.log(torch.tensor(0.75 / 0.25)).item())

    def forward(self, rgcn_x, ppr_x):
        gate = torch.sigmoid(self.gate(torch.cat([rgcn_x, ppr_x], dim=-1)))
        return gate * rgcn_x + (1.0 - gate) * ppr_x


class LinkPredict(nn.Module):
    """
    Link prediction model using R-GCN.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
                pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                pretrained_relation_embeddings=None, freeze=False, w=0.5,
                fusion_variant=None, use_domain_mlp_projector=False,
                use_residual_rgcn=False, use_ppr=False, ppr_num_layers=2):
        super(LinkPredict, self).__init__()
        #Tạo backbone RGCN để sinh embedding cho các node với 2 nguồn thông tin:text_embedding và domain_knowledge_embedding
        #Do dùng cả cạnh thuận và cạnh nghịch nên số quan hệ trong graph conv được x2 : num_relations *2
        self.rgcn = RGCN(input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
                        num_hidden_layers, dropout, use_cuda, pretrained_text_embeddings=pretrained_text_embeddings,
                        pretrained_domain_embeddings=pretrained_domain_embeddings, freeze=freeze,w=w,
                        fusion_variant=fusion_variant,
                        use_domain_mlp_projector=use_domain_mlp_projector,
                        use_residual_rgcn=use_residual_rgcn)
        self.use_ppr = use_ppr
        if use_ppr:
            self.ppr_branch = PPRBranch(hidden_dim, num_layers=ppr_num_layers, dropout=dropout)
            self.ppr_fusion = FusionGate(hidden_dim)
        #Hệ số để cộng vòa regularization loss
        self.regularization_param = regularization_param
        # nếu có pretrained_relation_embeddings thì chuyển thành tham số để train tiếp
        if pretrained_relation_embeddings is not None:
            # Chuyển self.relation_weights thành tham số để có thể train tiếp
            self.relation_weights = nn.Parameter(torch.Tensor(pretrained_relation_embeddings))
            # normalize relation_weights để chuyển về [0,1]
            normalized_relations = (self.relation_weights - self.relation_weights.min()) / (
                    self.relation_weights.max() - self.relation_weights.min())

            self.relation_weights.data.copy_(normalized_relations)

            print("Loaded pretrained relation embeddings.")
        else:
            #nếu ko có pretrained_relation_embeddings thì tạo random relation_weights
            self.relation_weights = nn.Parameter(torch.Tensor(num_relations, hidden_dim))
            nn.init.xavier_uniform_(self.relation_weights, gain=nn.init.calculate_gain('relu'))
            print("Initialized random relation embeddings.")

    def calculate_score(self, embeddings, triplets):
        """
        Calculate the score for triplets using DistMult.
        """
        subject_embeddings = embeddings[triplets[:, 0]]
        relation_embeddings = self.relation_weights[triplets[:, 1]]
        object_embeddings = embeddings[triplets[:, 2]]
        score = torch.sum(subject_embeddings * relation_embeddings * object_embeddings, dim=1)
        return score

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
        """
        Compute regularization loss for embeddings and relation weights.
        """
        return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))

    def get_loss(self, graph, embeddings, triplets, labels):
        """
        Compute loss for link prediction, including regularization loss.
        """
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

    def get_final_embeddings(self, graph, node_ids, rel_ids, norm):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(graph, node_ids, rel_ids, norm)
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
                                batch_size=4096, seed=None, return_pairs=False):
        embeddings = self.get_final_embeddings(graph, node_ids, rel_ids, norm)
        return self.evaluate_auroc(
            embeddings, test_triplets, total_data,
            batch_size=batch_size, seed=seed, return_pairs=return_pairs,
        )
