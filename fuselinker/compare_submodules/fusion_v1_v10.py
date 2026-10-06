"""Fusion V1-V10 from compare_submodules/10_fusion_variants.md."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV3(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV4(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_x, domain_x):
        diff = torch.abs(text_x - domain_x)

        gate = torch.sigmoid(
            self.gate(diff)
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV5(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x):
        diff = torch.abs(text_x - domain_x)

        features = torch.cat(
            [text_x, domain_x, diff],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(features)
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV6(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x):
        interaction = text_x * domain_x

        features = torch.cat(
            [text_x, domain_x, interaction],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(features)
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV7(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        bottleneck = max(hidden_dim // 4, 16)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, text_x, domain_x):
        features = torch.cat(
            [text_x, domain_x],
            dim=-1
        )

        gate = self.gate(features)

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV8(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.base_gate = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.node_gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)

    def forward(self, text_x, domain_x):
        node_adjustment = self.node_gate(
            torch.cat([text_x, domain_x], dim=-1)
        )

        gate = torch.sigmoid(
            self.base_gate + node_adjustment
        )

        return gate * text_x + (1.0 - gate) * domain_x


class FusionV9(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.text_score = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.domain_score = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(self, text_x, domain_x):
        text_logit = self.text_score(text_x)
        domain_logit = self.domain_score(domain_x)

        scores = torch.stack(
            [text_logit, domain_logit],
            dim=-1
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        text_weight = weights[..., 0]
        domain_weight = weights[..., 1]

        return (
            text_weight * text_x
            + domain_weight * domain_x
        )


class FusionV10(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

    def forward(self, text_x, domain_x):
        text_norm = F.normalize(
            text_x,
            p=2,
            dim=-1
        )

        domain_norm = F.normalize(
            domain_x,
            p=2,
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [text_norm, domain_norm],
                    dim=-1
                )
            )
        )

        return (
            gate * text_x
            + (1.0 - gate) * domain_x
        )


FUSION_BUILDERS = {
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


def build_fusion(name, hidden_dim):
    try:
        builder = FUSION_BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(FUSION_BUILDERS))
        raise ValueError(f"Unknown fusion variant {name!r}. Known: {known}") from exc
    return builder(hidden_dim)
