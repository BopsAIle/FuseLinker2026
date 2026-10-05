class FusionGate(nn.Module):
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