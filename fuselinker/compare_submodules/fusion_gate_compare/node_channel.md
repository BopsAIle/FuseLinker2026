# node_channel

## Code

```python
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
```

## Runs

- seed 42: MR 3.002435 | MRR 0.709014 | Hits@1 0.586326 | Hits@3 0.792395 | Hits@10 0.944966 | AUROC 0.966969
- seed 43: MR 3.363966 | MRR 0.692842 | Hits@1 0.572862 | Hits@3 0.769159 | Hits@10 0.930602 | AUROC 0.967498
- seed 44: MR 4.003448 | MRR 0.669267 | Hits@1 0.552011 | Hits@3 0.737688 | Hits@10 0.905079 | AUROC 0.955394

## Mean

MR 3.456616 | MRR 0.690374 | Hits@1 0.570400 | Hits@3 0.766414 | Hits@10 0.926882 | AUROC 0.963287
