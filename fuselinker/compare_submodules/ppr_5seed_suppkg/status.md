# PPR 5-seed — dừng giữa chừng

Đã dừng lúc train `model.py + PPR`, embedding `bert_pretrained_embeddings_768`, seed 43, khoảng epoch 4000/40000. Seed 43 chưa có `metrics.json` và chưa có checkpoint, nên lần chạy lại sẽ train seed này từ đầu.

## Đã xong

- `model.py` đủ 5 seed trên cả hai embedding. Script sẽ bỏ qua các seed này.
- Đồ thị PPR đã cache: `compare_submodules/checkpoints/suppkg/ppr_cache/topk50_c0.15_iter50_eps0.0001_batch128_87dc21b72e00.pt`
- `model.py + PPR` / `bert_pretrained_embeddings_768` / seed 42:

MR 3.121262 | MRR 0.724909 | Hits@1 0.612161 | Hits@3 0.801693 | Hits@10 0.940701 | AUROC 0.975386

Checkpoint: `compare_submodules/checkpoints/suppkg/bert_pretrained_embeddings_768/compare_model/ppr/seed_42/`

## Còn thiếu

- BERT + PPR: seed 43, 44, 45, 46
- PubMedBERT + PPR: seed 42, 43, 44, 45, 46

## Chạy lại

Trong `fuselinker/`:

```bash
python -u compare_submodules/run_ppr_5seeds.py
```

Script bỏ qua seed đã có `metrics.json`, rồi ghi tiếp `ppr_5seed_suppkg/summary.md`.
