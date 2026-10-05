import sys
from pathlib import Path

import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class BaseRGCN(nn.Module):
    """
    Base class for Relational Graph Convolutional Network (R-GCN) model.
    This class initializes the model and defines the base layers.
    """

    def __init__(self, num_nodes, hidden_dim, output_dim, num_relations, num_bases=-1,
                num_hidden_layers=1, dropout=0.0, use_self_loop=False, use_cuda=False, pretrained_text_embeddings=None,
                pretrained_domain_embeddings=None, freeze=False, w=0.5, fusion_variant=None):
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

    def __init__(self, num_nodes, hidden_dim, pretrained_text_embeddings, pretrained_domain_embeddings, freeze=False,w=0.5, fusion_variant=None):
        super(EmbeddingLayer, self).__init__()
        self.w = w
        self.fusion_variant = fusion_variant
        if fusion_variant:
            self.fusion = build_fusion(fusion_variant, hidden_dim)
        # Pretrained domain embeddings
        if pretrained_domain_embeddings is not None:
            #Kích thước ma trận pretrained_domain_embeddings :(num_nodes, hidden_dim)
            domain_embeddings = torch.from_numpy(pretrained_domain_embeddings).float()
            #Cái norm_domain_embeddings đơn giản là đưa về khoảng [0,1]
            norm_domain_embeddings = (domain_embeddings - domain_embeddings.min()) / (
                    domain_embeddings.max() - domain_embeddings.min())

            self.poincare_to_euclidean = nn.Linear(pretrained_domain_embeddings.shape[1], hidden_dim)
            #Cái này để nói rằng chúng ta sẽ không coi embedding là tham số để học
            self.norm_domain_embeddings = nn.Embedding.from_pretrained(norm_domain_embeddings, freeze=freeze)

            print(f"Loaded pretrained domain embeddings, freeze is {freeze}.")
        else:
            self.norm_domain_embeddings = nn.Embedding(num_nodes, hidden_dim)
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
        transformed_domain_embeddings = self.poincare_to_euclidean(self.norm_domain_embeddings(node_ids.squeeze()))

        if self.fusion_variant:
            combined_embedding = self.fusion(
                transformed_text_embeddings, transformed_domain_embeddings
            )
        else:
            combined_embedding = (1 - self.w) * transformed_domain_embeddings + self.w * transformed_text_embeddings

        return combined_embedding


class RGCN(BaseRGCN):
    """
    Implementation of R-GCN with support for link prediction.
    """

    def build_input_layer(self):
        # Initialize node features with embedding layer
        return EmbeddingLayer(self.num_nodes, self.hidden_dim, self.pretrained_text_embeddings,
                            self.pretrained_domain_embeddings, self.freeze, self.w,
                            fusion_variant=self.fusion_variant)

    def build_hidden_layer(self, idx):
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


class LinkPredict(nn.Module):
    """
    Link prediction model using R-GCN.
    """

    def __init__(self, input_dim, hidden_dim, num_relations, num_bases=-1,
                num_hidden_layers=1, dropout=0.0, use_cuda=False, regularization_param=0.0,
                pretrained_text_embeddings=None, pretrained_domain_embeddings=None,
                pretrained_relation_embeddings=None, freeze=False, w=0.5,
                fusion_variant=None):
        super(LinkPredict, self).__init__()
        #Tạo backbone RGCN để sinh embedding cho các node với 2 nguồn thông tin:text_embedding và domain_knowledge_embedding
        #Do dùng cả cạnh thuận và cạnh nghịch nên số quan hệ trong graph conv được x2 : num_relations *2
        self.rgcn = RGCN(input_dim, hidden_dim, hidden_dim, num_relations * 2, num_bases,
                        num_hidden_layers, dropout, use_cuda, pretrained_text_embeddings=pretrained_text_embeddings,
                        pretrained_domain_embeddings=pretrained_domain_embeddings, freeze=freeze,w=w,
                        fusion_variant=fusion_variant)
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

    def forward(self, graph, node_ids, rel_ids, norm):
        return self.rgcn(graph, node_ids, rel_ids, norm)

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
