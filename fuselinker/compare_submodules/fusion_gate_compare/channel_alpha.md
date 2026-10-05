# channel_alpha

## Code

```python
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
```

## Runs

- seed 42: MR 2.972417 | MRR 0.725081 | Hits@1 0.609660 | Hits@3 0.803098 | Hits@10 0.945162 | AUROC 0.968258
- seed 43: MR 2.897480 | MRR 0.732114 | Hits@1 0.617994 | Hits@3 0.811219 | Hits@10 0.948871 | AUROC 0.971426
- seed 44: MR 3.063025 | MRR 0.716263 | Hits@1 0.599284 | Hits@3 0.794650 | Hits@10 0.942237 | AUROC 0.968077

## Mean

MR 2.977641 | MRR 0.724486 | Hits@1 0.608980 | Hits@3 0.802989 | Hits@10 0.945423 | AUROC 0.969254
