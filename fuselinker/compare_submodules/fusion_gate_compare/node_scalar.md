# node_scalar

## Code

```python
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
```

## Runs

- seed 42: MR 3.053417 | MRR 0.712563 | Hits@1 0.592715 | Hits@3 0.793997 | Hits@10 0.942743 | AUROC 0.969496
- seed 43: MR 3.185088 | MRR 0.698060 | Hits@1 0.574316 | Hits@3 0.779993 | Hits@10 0.939148 | AUROC 0.965273
- seed 44: MR 3.255155 | MRR 0.690151 | Hits@1 0.564479 | Hits@3 0.773293 | Hits@10 0.936893 | AUROC 0.969193

## Mean

MR 3.164553 | MRR 0.700258 | Hits@1 0.577170 | Hits@3 0.782428 | Hits@10 0.939595 | AUROC 0.967987
