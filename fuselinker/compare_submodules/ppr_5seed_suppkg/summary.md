# model.py vs model.py + PPR

Dataset: suppKG (`--data suppkg`).
Seeds: 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 seed.
suppKG (data=suppkg), 40000 iterations, evaluate_every=1000, num_hidden_layers=2, w=0.75, knowledge=poincare_embeddings.npy, fusion=fixed, domain map=Linear, hidden layers=RelGraphConv. PPR is a CUDA top-50 Personalized PageRank branch (c=0.15, epsilon=1e-4, 50 iterations, batch_size=128, 2 PPR layers) fused with the R-GCN output by a zero-init gate starting at 0.75/0.25 when --use_ppr true.

Thư mục ghi chú: `compare_submodules/ppr_5seed_suppkg/`.
Checkpoint gốc: `compare_submodules/checkpoints/suppkg/`.
Đồ thị PPR dùng chung: `checkpoints/suppkg/ppr_cache/`.

| Bản | Thư mục checkpoint |
| --- | --- |
| model.py | `checkpoints/suppkg/<embedding>/compare_model/seed_<seed>/` |
| model.py + PPR | `checkpoints/suppkg/<embedding>/compare_model/ppr/seed_<seed>/` |

File từng bản:

- `bert_pretrained_embeddings_768__model.md`
- `bert_pretrained_embeddings_768__model_plus_ppr.md`
- `pubmedbert_pretrained_embeddings_768__model.md`
- `pubmedbert_pretrained_embeddings_768__model_plus_ppr.md`

## bert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.966329 | 0.724361 | 0.608363 | 0.803798 | 0.945939 | 0.968892 |
| model.py + PPR | 3.223390 | 0.718338 | 0.603755 | 0.795836 | 0.937034 | 0.977488 |
| delta (PPR − model.py) | 0.257061 | -0.006023 | -0.004608 | -0.007961 | -0.008906 | 0.008596 |

## pubmedbert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.919494 | 0.725326 | 0.609036 | 0.804967 | 0.947596 | 0.970405 |
| model.py + PPR | 3.263355 | 0.721004 | 0.607834 | 0.798131 | 0.934933 | 0.975038 |
| delta (PPR − model.py) | 0.343861 | -0.004322 | -0.001203 | -0.006837 | -0.012664 | 0.004633 |

MR thấp hơn là tốt hơn. MRR, Hits và AUROC cao hơn là tốt hơn.
Delta dương trên MRR, Hits, AUROC nghĩa là PPR cao hơn model.py.
