# model.py vs model.py + Domain MLP Projector

Dataset: suppKG (`--data suppkg`).
Seeds: 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 seed.
suppKG (data=suppkg), 40000 iterations, evaluate_every=1000, num_hidden_layers=2, w=0.75, knowledge=poincare_embeddings.npy, fusion=fixed. Domain map is Linear in model.py, and MLPProjector (Linear -> ReLU -> Dropout -> Linear -> LayerNorm) when --use_domain_mlp_projector true.

Thư mục ghi chú: `compare_submodules/domain_mlp_projector_5seed_suppkg/`.
Checkpoint gốc: `compare_submodules/checkpoints/suppkg/`.

| Bản | Thư mục checkpoint |
| --- | --- |
| model.py | `checkpoints/suppkg/<embedding>/compare_model/seed_<seed>/` |
| model.py + Domain MLP Projector | `checkpoints/suppkg/<embedding>/compare_model/domain_mlp_projector/seed_<seed>/` |

File từng bản:

- `bert_pretrained_embeddings_768__model.md`
- `bert_pretrained_embeddings_768__model_plus_domain_mlp_projector.md`
- `pubmedbert_pretrained_embeddings_768__model.md`
- `pubmedbert_pretrained_embeddings_768__model_plus_domain_mlp_projector.md`

## bert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.966329 | 0.724361 | 0.608363 | 0.803798 | 0.945939 | 0.968892 |
| model.py + Domain MLP Projector | 2.826292 | 0.731329 | 0.614801 | 0.813206 | 0.951570 | 0.975888 |
| delta (projector − model.py) | -0.140037 | 0.006968 | 0.006438 | 0.009409 | 0.005631 | 0.006996 |

## pubmedbert_pretrained_embeddings_768

| Bản | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model.py | 2.919494 | 0.725326 | 0.609036 | 0.804967 | 0.947596 | 0.970405 |
| model.py + Domain MLP Projector | 2.783895 | 0.735230 | 0.619474 | 0.817092 | 0.953492 | 0.974551 |
| delta (projector − model.py) | -0.135599 | 0.009904 | 0.010438 | 0.012125 | 0.005896 | 0.004146 |

MR thấp hơn là tốt hơn. MRR, Hits và AUROC cao hơn là tốt hơn.
Delta dương trên MRR, Hits, AUROC nghĩa là Domain MLP Projector cao hơn model.py.
