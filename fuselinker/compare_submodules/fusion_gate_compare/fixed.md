# fixed

## Code

```python
fused = (1 - w) * domain_x + w * text_x
# w = 0.75
```

## Runs

- seed 42: MR 2.926256 | MRR 0.734288 | Hits@1 0.623468 | Hits@3 0.808997 | Hits@10 0.946681 | AUROC 0.967217
- seed 43: MR 2.830256 | MRR 0.742514 | Hits@1 0.633076 | Hits@3 0.818246 | Hits@10 0.949361 | AUROC 0.968721
- seed 44: MR 3.071947 | MRR 0.717490 | Hits@1 0.600657 | Hits@3 0.796170 | Hits@10 0.941714 | AUROC 0.971044

## Mean

MR 2.942820 | MRR 0.731430 | Hits@1 0.619067 | Hits@3 0.807804 | Hits@10 0.945919 | AUROC 0.968994
