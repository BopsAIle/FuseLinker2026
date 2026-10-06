# Baseline 5-seed means

Mỗi script train 5 lần với seed 42, 43, 44, 45, 46. Mean là trung bình cộng của 5 lần.
suppkg, 40000 iterations, pubmedbert_pretrained_embeddings_768, poincare, w=0.75, num_hidden_layers=2, no extra module flags.

| Model | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| model | 2.919494 | 0.725326 | 0.609036 | 0.804967 | 0.947596 | 0.970405 |
| model_base_test | 2.929099 | 0.723126 | 0.605850 | 0.803644 | 0.947789 | 0.969750 |

Từng seed nằm trong file markdown của từng model.
