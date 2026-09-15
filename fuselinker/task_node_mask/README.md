# Nhiệm vụ 1 — dự đoán entity bị mask

Đây là **FuseLinker gốc**: link prediction trên knowledge graph. Với mỗi triple
`(head, relation, tail)`, mô hình che head hoặc tail rồi phải xếp entity đúng
cao hơn các entity âm. Nhiệm vụ này **không** phân loại loại cạnh hay bucket
bậc node; phần đó nằm ở [`../task_degree_edge/`](../task_degree_edge/).

Tổng quan hai nhiệm vụ: [`../README.md`](../README.md).

## Mô tả nhiệm vụ

Quy trình train mỗi iteration:

1. Lấy một batch cạnh từ `train.tsv`.
2. Giữ một phần cạnh (`--graph_split_size`) làm graph truyền thông điệp R-GCN;
   phần còn lại là các triple dương để học DistMult.
3. Sinh triple âm bằng cách thay head hoặc tail.
4. Kết hợp embedding văn bản và embedding tri thức, mã hóa node bằng R-GCN
   (bản upgrade thêm nhánh PPR).
5. Tính điểm DistMult:

   ```text
   score(h, r, t) = sum(h ⊙ r ⊙ t)
   ```

6. Tối ưu binary cross-entropy trên triple dương/âm, cộng regularization L2.

Sau train, mô hình được đánh giá trên `test.tsv` theo ranking (MR, MRR, Hits@K)
và AUROC.

## Cấu trúc thư mục

```text
task_node_mask/
  train_base.py              # DistMult base, encoder model.py
  train_upgrade.py           # DistMult upgrade, encoder model_base4.py
  train_upgrade_base2.py     # checkpoint cũ, encoder model_base2.py
  model.py / model_base4.py / model_base2.py
  eval_auroc.py              # đánh giá AUROC từ checkpoint
  plot_roc_comparision.py    # vẽ ROC so sánh nhiều dataset/biến thể
  task_setup.py              # sys.path: thư mục này + common/
  docs/                      # sơ đồ pipeline encoder
```

`--data` và mọi checkpoint tương đối luôn resolve về `fuselinker/`, không phụ
thuộc CWD. Nên chạy từ `fuselinker/`:

```bash
cd FuseLinker/fuselinker
python task_node_mask/train_base.py --help
```

## Hai biến thể

- **Base** — `train_base.py` + `model.py`: chiếu embedding domain và text về
  cùng số chiều, trộn `(1-w) * domain + w * text`, rồi đưa qua R-GCN.
- **Upgrade** — `train_upgrade.py` + `model_base4.py`: cùng backbone DistMult
  với base (min-max, `(1-w)*domain + w*text`, R-GCN); thêm nhánh PPR residual.
- `train_upgrade_base2.py` chỉ phục vụ kiến trúc/checkpoint cũ (`model_base2`).

## Train base

Mặc định lưu:
`checkpoints/hetionet/pubmedbert/model/model_state.pth`.

```bash
python task_node_mask/train_base.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 1000 \
  --negative_sample 20 \
  --neg_sample_size_eval 100 \
  --w 0.75 \
  --model_state_file checkpoints/hetionet/pubmedbert/model/model_state.pth
```

## Train upgrade

Mặc định lưu:
`checkpoints/hetionet/pubmedbert/modelbase/model_state.pth`.

```bash
python task_node_mask/train_upgrade.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 1000 \
  --negative_sample 20 \
  --neg_sample_size_eval 100 \
  --w 0.75 \
  --use_ppr true \
  --ppr_iterative true \
  --model_state_file checkpoints/hetionet/pubmedbert/modelbase/model_state.pth
```

Train legacy (`model_base2`) nếu cần nạp checkpoint cũ:

```bash
python task_node_mask/train_upgrade_base2.py \
  --data hetionet \
  --iterations 40000 \
  --model_state_file checkpoints/hetionet/pubmedbert/model_base2/model_state.pth
```

## Tham số quan trọng

- `--data`: tên thư mục dataset dưới `fuselinker/` hoặc đường dẫn tuyệt đối.
- `--text_embedding_file` / `--knowledge_embedding_file`: file `.npy` trong
  thư mục dataset. Hàng thứ `i` phải khớp entity ID `i`.
- `--graph_batch_size`: số cạnh được lấy mẫu mỗi iteration (mặc định 250).
- `--graph_split_size`: tỷ lệ cạnh mẫu dùng cho message passing; phần còn lại
  dùng làm triple dương DistMult (mặc định 0.5).
- `--negative_sample`: số mẫu âm khi train (mặc định 20).
- `--n_hidden`, `--num_hidden_layers`, `--dropout`: cấu hình encoder.
- `--w`: prior khi trộn embedding text/domain. Ở base đây là trọng số text;
  ở upgrade được kết hợp với gate học được.
- `--freeze`: không cập nhật hai embedding đầu vào.
- `--eval_protocol`: mặc định `filtered` khi tính ranking.
- `--use_ppr`, `--ppr_c`, `--ppr_eps`, `--ppr_num_layers`, `--ppr_iterative`:
  chỉ bản upgrade / `model_base2`.

## Đánh giá sau train

Script train lưu checkpoint rồi đánh giá trên `test.tsv`, in:

- **MR** (Mean Rank): thấp hơn tốt hơn;
- **MRR** và **Hits@1/3/10**: cao hơn tốt hơn;
- **AUROC**: cao hơn tốt hơn;
- ảnh ROC mặc định là `roc_curve.png` nằm cạnh checkpoint.

Đánh giá lại một checkpoint mà không train:

```bash
python task_node_mask/eval_auroc.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --model_module model \
  --model_state_file checkpoints/hetionet/pubmedbert/model/model_state.pth \
  --num_hidden_layers 2 \
  --w 0.75
```

Với checkpoint upgrade, đổi `--model_module model_base4` và giữ đúng cấu hình
PPR. `--model_module` nhận `model`, `model_base4` hoặc `model_base2`.

So sánh ROC nhiều dataset/biến thể:

```bash
python task_node_mask/plot_roc_comparision.py \
  --root . \
  --datasets hetionet suppkg kegg50k \
  --variants model modelbase \
  --output roc_comparison.png \
  --skip-missing
```

## Checkpoint

```text
checkpoints/<dataset>/<embedding>/
  model/            # DistMult base
  modelbase/        # DistMult upgrade (model_base4)
  model_base2/      # DistMult legacy
```

Sửa kiến trúc DistMult chỉ trong thư mục này; SSL bậc/cạnh
([`../task_degree_edge/`](../task_degree_edge/)) không bị ảnh hưởng.
