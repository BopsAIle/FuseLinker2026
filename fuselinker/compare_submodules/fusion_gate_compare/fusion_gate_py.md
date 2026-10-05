# fusion_gate_py

## Code

```python
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
```

## Runs

- seed 42: MR 3.700301 | MRR 0.659621 | Hits@1 0.531177 | Hits@3 0.738047 | Hits@10 0.918870 | AUROC 0.961323
- seed 43: MR 3.768914 | MRR 0.658381 | Hits@1 0.530459 | Hits@3 0.735531 | Hits@10 0.916337 | AUROC 0.959935
- seed 44: MR 3.428887 | MRR 0.672060 | Hits@1 0.542632 | Hits@3 0.753162 | Hits@10 0.931501 | AUROC 0.963385

## Mean

MR 3.632700 | MRR 0.663354 | Hits@1 0.534756 | Hits@3 0.742246 | Hits@10 0.922236 | AUROC 0.961548
