# Fusion variant means

Mỗi bản train 3 lần với seed 42, 43, 44. Mean là trung bình cộng của 3 lần.
Protocol: suppkg, 40000 iterations, bert_pretrained_embeddings_768, poincare, w=0.75, num_hidden_layers=2.

| Variant | MR | MRR | Hits@1 | Hits@3 | Hits@10 | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed | 2.942820 | 0.731430 | 0.619067 | 0.807804 | 0.945919 | 0.968994 |
| channel_alpha | 2.977641 | 0.724486 | 0.608980 | 0.802989 | 0.945423 | 0.969254 |
| additive_residual | 3.012359 | 0.715103 | 0.596446 | 0.794547 | 0.944955 | 0.965101 |
| node_scalar | 3.164553 | 0.700258 | 0.577170 | 0.782428 | 0.939595 | 0.967987 |
| node_channel | 3.456616 | 0.690374 | 0.570400 | 0.766414 | 0.926882 | 0.963287 |
| fusion_gate_py | 3.632700 | 0.663354 | 0.534756 | 0.742246 | 0.922236 | 0.961548 |

Chi tiết từng seed và code nằm trong file markdown của từng bản.
