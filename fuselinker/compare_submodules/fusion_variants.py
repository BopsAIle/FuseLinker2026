"""Text/domain fusion variants for compare_submodules/model_base_test.py.

Each class mixes text_x and domain_x with forward(text_x, domain_x, w=None).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F


def _logit(probability):
    prior = torch.tensor(float(probability)).clamp(1e-4, 1.0 - 1e-4)
    return torch.log(prior / (1.0 - prior)).item()


class ChannelAlphaFusion(nn.Module):
    """One mix weight per hidden channel, shared by every node."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.alpha = nn.Parameter(torch.full((hidden_dim,), _logit(0.75)))

    def forward(self, text_x, domain_x, w=None):
        del w
        alpha = torch.sigmoid(self.alpha)
        return alpha * text_x + (1.0 - alpha) * domain_x


class NodeScalarFusion(nn.Module):
    """One scalar gate per node. Starts at 0.75 while the weight is zero."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, _logit(0.75))

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


class NodeChannelFusion(nn.Module):
    """Per-node, per-channel linear gate. Starts at 0.75 while the weight is zero."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, _logit(0.75))

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionGatePy(nn.Module):
    """Verbatim copy of fuselinker/Fusion Gate.py."""

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
        if w is not None:
            gate = 0.5 * gate + 0.5 * w
        fused = gate * text_x + (1.0 - gate) * domain_x
        fused = fused + self.residual(torch.cat([text_x, domain_x], dim=-1))
        return self.norm(fused)


class AdditiveResidualFusion(nn.Module):
    """Fixed w-mix plus a residual MLP whose last layer starts at zero."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, text_x, domain_x, w=None):
        mix = 0.75 if w is None else w
        base = (1.0 - mix) * domain_x + mix * text_x
        delta = self.net(torch.cat([text_x, domain_x], dim=-1))
        return base + delta


class FusionV1(nn.Module):
    """Concat gate, default Linear init. One gate per node and channel."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV2(nn.Module):
    """Concat gate with zero init, so every node starts at a 50/50 mix."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV3(nn.Module):
    """One scalar gate per node."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = torch.sigmoid(self.gate(torch.cat([text_x, domain_x], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV4(nn.Module):
    """Gate from the absolute difference of the two embeddings."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        diff = torch.abs(text_x - domain_x)
        gate = torch.sigmoid(self.gate(diff))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV5(nn.Module):
    """Gate from text, domain, and their absolute difference."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        diff = torch.abs(text_x - domain_x)
        features = torch.cat([text_x, domain_x, diff], dim=-1)
        gate = torch.sigmoid(self.gate(features))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV6(nn.Module):
    """Gate from text, domain, and their elementwise product."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        interaction = text_x * domain_x
        features = torch.cat([text_x, domain_x, interaction], dim=-1)
        gate = torch.sigmoid(self.gate(features))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV7(nn.Module):
    """Low-rank nonlinear gate: 2H -> H/4 -> H, no residual or LayerNorm."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        bottleneck = max(hidden_dim // 4, 16)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, text_x, domain_x, w=None):
        del w
        gate = self.gate(torch.cat([text_x, domain_x], dim=-1))
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV8(nn.Module):
    """Global channel prior plus a zero-init per-node adjustment."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.base_gate = nn.Parameter(torch.zeros(hidden_dim))
        self.node_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)

    def forward(self, text_x, domain_x, w=None):
        del w
        node_adjustment = self.node_gate(torch.cat([text_x, domain_x], dim=-1))
        gate = torch.sigmoid(self.base_gate + node_adjustment)
        return gate * text_x + (1.0 - gate) * domain_x


class FusionV9(nn.Module):
    """Per-channel softmax competition between a text score and a domain score."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.text_score = nn.Linear(hidden_dim, hidden_dim)
        self.domain_score = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        scores = torch.stack(
            [self.text_score(text_x), self.domain_score(domain_x)],
            dim=-1,
        )
        weights = torch.softmax(scores, dim=-1)
        return weights[..., 0] * text_x + weights[..., 1] * domain_x


class FusionV10(nn.Module):
    """Gate from L2-normalized inputs; the mix still uses the original vectors."""

    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        del dropout
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_x, domain_x, w=None):
        del w
        text_norm = F.normalize(text_x, p=2, dim=-1)
        domain_norm = F.normalize(domain_x, p=2, dim=-1)
        gate = torch.sigmoid(self.gate(torch.cat([text_norm, domain_norm], dim=-1)))
        return gate * text_x + (1.0 - gate) * domain_x


FUSION_BUILDERS = {
    "channel_alpha": ChannelAlphaFusion,
    "node_scalar": NodeScalarFusion,
    "node_channel": NodeChannelFusion,
    "fusion_gate_py": FusionGatePy,
    "additive_residual": AdditiveResidualFusion,
    "v1": FusionV1,
    "v2": FusionV2,
    "v3": FusionV3,
    "v4": FusionV4,
    "v5": FusionV5,
    "v6": FusionV6,
    "v7": FusionV7,
    "v8": FusionV8,
    "v9": FusionV9,
    "v10": FusionV10,
}


def build_fusion(name, hidden_dim, dropout=0.2):
    try:
        builder = FUSION_BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(FUSION_BUILDERS))
        raise ValueError(f"Unknown fusion variant {name!r}. Known: {known}") from exc
    return builder(hidden_dim, dropout)
