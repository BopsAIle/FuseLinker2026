# model.py vs model.py + Residual R-GCN

Dataset: suppKG (`--data suppkg`).
Seeds: 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 seed.
suppKG (data=suppkg), 40000 iterations, evaluate_every=1000, num_hidden_layers=2, w=0.75, knowledge=poincare_embeddings.npy, fusion=fixed, domain map=Linear. Hidden layers are RelGraphConv in model.py, and ResidualRelGraphConvBlock (LayerNorm, RelGraphConv, residual, FFN, residual, ReLU on every layer except the last) when --use_residual_rgcn true.

Thư mục ghi chú: `compare_submodules/residual_rgcn_5seed_suppkg/`.
Checkpoint gốc: `compare_submodules/checkpoints/suppkg/`.

| Bản | Thư mục checkpoint |
| --- | --- |
| model.py | `checkpoints/suppkg/<embedding>/compare_model/seed_<seed>/` |
| model.py + Residual R-GCN | `checkpoints/suppkg/<embedding>/compare_model/residual_rgcn/seed_<seed>/` |

File từng bản:

- `bert_pretrained_embeddings_768__model.md`
- `bert_pretrained_embeddings_768__model_plus_residual_rgcn.md`
- `pubmedbert_pretrained_embeddings_768__model.md`
- `pubmedbert_pretrained_embeddings_768__model_plus_residual_rgcn.md`

## bert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.966329 | 0.724361 | 0.608363 | 0.803798 | 0.945939 | 0.968892 |
| model.py + Residual R-GCN | 3.094467 | 0.733537 | 0.623096 | 0.809736 | 0.940812 | 0.985219 |
| delta (residual − model.py) | 0.128138 | 0.009176 | 0.014733 | 0.005938 | -0.005128 | 0.016327 |

## pubmedbert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.919494 | 0.725326 | 0.609036 | 0.804967 | 0.947596 | 0.970405 |
| model.py + Residual R-GCN | 2.641197 | 0.764839 | 0.660695 | 0.840456 | 0.956384 | 0.988616 |
| delta (residual − model.py) | -0.278297 | 0.039513 | 0.051659 | 0.035488 | 0.008788 | 0.018211 |

MR thấp hơn là tốt hơn. MRR, Hits và AUROC cao hơn là tốt hơn.
Delta dương trên MRR, Hits, AUROC nghĩa là Residual R-GCN cao hơn model.py.
