# additive_residual

## Code

```python
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
```

## Runs

- seed 42: MR 3.061963 | MRR 0.714862 | Hits@1 0.596392 | Hits@3 0.793637 | Hits@10 0.943054 | AUROC 0.963322
- seed 43: MR 2.908657 | MRR 0.725550 | Hits@1 0.610477 | Hits@3 0.803817 | Hits@10 0.948119 | AUROC 0.964331
- seed 44: MR 3.066456 | MRR 0.704896 | Hits@1 0.582470 | Hits@3 0.786186 | Hits@10 0.943691 | AUROC 0.967651

## Mean

MR 3.012359 | MRR 0.715103 | Hits@1 0.596446 | Hits@3 0.794547 | Hits@10 0.944955 | AUROC 0.965101
