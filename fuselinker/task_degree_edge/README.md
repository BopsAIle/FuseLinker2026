# Nhiệm vụ 2 — dự đoán bậc node và cạnh bị mask

SSL kiểu LukePi: mask cạnh khỏi message-passing, phân loại loại cạnh (kèm
negative) và bucket bậc node. Nhiệm vụ này **không** dùng DistMult để khôi phục
entity bị che; phần đó nằm ở [`../task_node_mask/`](../task_node_mask/).

Tổng quan hai nhiệm vụ: [`../README.md`](../README.md). Chi tiết độ đo:
[`docs/ssl_metrics.txt`](docs/ssl_metrics.txt).

## Mô tả nhiệm vụ

Đây là bước pretrain tự giám sát gồm hai đầu phân loại dùng chung encoder:

1. **Masked-edge classification.** Bỏ ngẫu nhiên một tỷ lệ cạnh
   (`--mask_rate`) khỏi graph encode. Từ `concat(h_src, h_dst)`, head phân loại
   dự đoán một trong `R` loại quan hệ thật hoặc lớp thứ `R+1` dành cho cạnh
   âm/fake.
2. **Node-degree classification.** Tính bậc node trên graph train, chia thành
   `K` bucket (`--num_degree_bins`) rồi dự đoán bucket từ embedding node. Cách
   chia: `log` (theo log2, mặc định) hoặc `quantile`.

Loss tổng cân bằng hai task bằng **trọng số tự học** (uncertainty weighting,
Kendall et al. 2018 — mặc định `--learn_loss_weights true`):

```text
loss = exp(-s_edge) * CE_edge + exp(-s_node) * CE_node
     + 0.5 * (s_edge + s_node)
```

`s = log(sigma^2)` là tham số học được cho mỗi task; trọng số hiệu dụng
`lambda_task = exp(-s)`. Số hạng `0.5*s` là regularizer chống việc đẩy trọng số
về 0. Task nào nhiễu/khó hơn sẽ tự nhận trọng số nhỏ hơn, nên không cần dò tay.

Tắt bằng `--learn_loss_weights false` để dùng lambda tĩnh:

```text
loss = CE_edge + lambda_degree * CE_node   (--lambda_degree, mặc định 1.0)
```

Degree loss có class weight tỷ lệ nghịch căn bậc hai tần suất, để giảm ảnh
hưởng mất cân bằng giữa các bucket. Checkpoint tốt nhất được chọn theo edge CE
trên tập validation (không in loss valid mỗi epoch).

## Cấu trúc thư mục

```text
task_degree_edge/
  train_base.py              # SSL base, encoder model.py
  train_upgrade.py           # SSL upgrade, encoder model_base4.py
  model.py / model_base4.py
  lukepi_graph.py            # lấy mẫu graph đã mask + nhãn cạnh
  ssl_eval.py                # độ đo, checkpoint, encode graph đầy đủ
  task_setup.py              # sys.path: thư mục này + common/
  docs/ssl_metrics.txt
```

`--data` và `--ssl_model_state_file` tương đối luôn resolve về `fuselinker/`.
Nên chạy từ `fuselinker/`:

```bash
cd FuseLinker/fuselinker
python task_degree_edge/train_base.py --help
```

## Hai biến thể

- **Base** — `train_base.py` + `model.py`: R-GCN và phép trộn embedding tuyến
  tính `(1-w) * domain + w * text`. Task bậc mặc định học trên subgraph
  (LukePi, `--degree_on_full false`).
- **Upgrade** — `train_upgrade.py` + `model_base4.py`: FusionGate và nhánh PPR.
  Mặc định degree head học trên toàn graph train (`--degree_on_full true`) để
  mọi node nhận supervision, khớp lúc TEST.

Sửa kiến trúc SSL chỉ trong thư mục này; DistMult (nhiệm vụ 1) không bị ảnh
hưởng.

## Train base

Mặc định lưu:
`checkpoints/hetionet/pubmedbert/ssl_base/ssl_model_state.pth`.

```bash
python task_degree_edge/train_base.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 50 \
  --evaluate_every 1 \
  --graph_batch_size 1024 \
  --mask_rate 0.2 \
  --negative_sample 1 \
  --num_degree_bins 10 \
  --degree_bin_strategy quantile \
  --w 0.75 \
  --ssl_model_state_file checkpoints/hetionet/pubmedbert/ssl_base/ssl_model_state.pth
```

## Train upgrade

Mặc định lưu:
`checkpoints/hetionet/pubmedbert/ssl_upgrade/ssl_model_state.pth`.

```bash
python task_degree_edge/train_upgrade.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 50 \
  --evaluate_every 1 \
  --graph_batch_size 1024 \
  --mask_rate 0.2 \
  --negative_sample 1 \
  --num_degree_bins 10 \
  --degree_bin_strategy quantile \
  --degree_on_full true \
  --use_ppr true \
  --ppr_iterative true \
  --w 0.75 \
  --ssl_model_state_file checkpoints/hetionet/pubmedbert/ssl_upgrade/ssl_model_state.pth
```

## Tham số quan trọng

- `--data`: tên thư mục dataset dưới `fuselinker/` hoặc đường dẫn tuyệt đối.
- `--text_embedding_file` / `--knowledge_embedding_file`: file `.npy` trong
  thư mục dataset. Hàng thứ `i` phải khớp entity ID `i`.
- `--mask_rate`: tỷ lệ cạnh dương bị loại khỏi graph encode (mặc định 0.2).
- `--negative_sample`: số cạnh âm trên mỗi cạnh dương bị mask (mặc định 1,
  gần LukePi).
- `--num_degree_bins`: số lớp bậc node (mặc định 10).
- `--degree_bin_strategy {log,quantile}`: cách tạo bucket bậc.
- `--learn_loss_weights`: `true` (mặc định) tự học trọng số 2 task theo
  uncertainty; `false` để dùng `--lambda_degree` tĩnh.
- `--lambda_degree`: trọng số cho `CE_node` khi `--learn_loss_weights false`
  (mặc định 1.0).
- `--evaluate_every`: chu kỳ encode graph train đầy đủ và đánh giá validation
  (mặc định 100). TEST cuối cũng encode trên **graph train** (nhãn bậc lấy từ
  `train.tsv`; cạnh test chỉ dùng để chấm điểm, không đưa vào encoder).
- `--eval_neg_rate`: tỷ lệ âm khi đánh giá cạnh; mặc định bằng
  `--negative_sample`.
- `--eval_max_triples`: giới hạn số triple valid/test mỗi lần SSL edge eval
  (mặc định 5000).
- `--degree_on_full`: upgrade mặc định `true` (học bậc trên toàn graph train).
  Base mặc định `false` (LukePi: học bậc trên subgraph). Đừng bật cùng lúc cho
  cả hai nếu muốn so sánh encoder; đó là chỗ upgrade khác base.
- `--w`, `--freeze`, `--n_hidden`, `--num_hidden_layers`, `--dropout`: giống
  nhiệm vụ 1.
- `--use_ppr`, `--ppr_c`, `--ppr_eps`, `--ppr_num_layers`, `--ppr_iterative`,
  `--ppr_batch_size`: chỉ bản upgrade.

`--reg_param` được giữ trong checkpoint để tương thích DistMult, **không**
dùng trong SSL CE loss.

## Độ đo

Log mỗi chu kỳ tách riêng **2 dòng cho 2 task** (kèm dòng tổng loss), trung
bình trên batch train rồi in test:

```text
Training average | step N | loss_total ...
Training average | step N | [EDGE]   loss ... auc ... aupr ... f1 ... bacc ... w ...
Training average | step N | [DEGREE] loss ... auc ... aupr ... f1 ... bacc ... w ...
TEST             | step N | loss_total ...
TEST             | step N | [EDGE]   loss ... auc ... aupr ... f1 ... bacc ... w ...
TEST             | step N | [DEGREE] loss ... auc ... aupr ... f1 ... bacc ... w ...
```

| Token | Ý nghĩa | Hướng tốt |
|---|---|---|
| `loss_total` | Objective thực tối ưu (tổng có trọng số) | thấp hơn |
| `[EDGE] loss` | CE cạnh (thô, chưa nhân trọng số) | thấp hơn |
| `[EDGE] auc` | Edge macro one-vs-rest AUROC | cao hơn |
| `[EDGE] aupr` | Edge macro average precision | cao hơn |
| `[EDGE] f1` | Edge Macro-F1 | cao hơn |
| `[EDGE] bacc` | Edge balanced accuracy | cao hơn |
| `[EDGE] w` | Trọng số hiệu dụng `lambda_edge` | — |
| `[DEGREE] loss` | CE degree (thô, chưa nhân trọng số) | thấp hơn |
| `[DEGREE] auc` | Degree macro one-vs-rest AUROC | cao hơn |
| `[DEGREE] aupr` | Degree macro average precision | cao hơn |
| `[DEGREE] f1` | Degree Macro-F1 | cao hơn |
| `[DEGREE] bacc` | Degree balanced accuracy | cao hơn |
| `[DEGREE] w` | Trọng số hiệu dụng `lambda_node` | — |

Softmax trên logits cho xác suất theo lớp. AUC là macro OVR trên các lớp có cả
mẫu dương và âm. AUPR là macro average precision (không nội suy). F1 là
Macro-F1 trên các lớp xuất hiện trong nhãn. BACC là trung bình recall theo
lớp có trong nhãn.

## Checkpoint

Script lưu checkpoint tốt nhất tại `--ssl_model_state_file`. File chứa encoder,
`edge_head`, `degree_head`, `loss_weighter` (nếu học trọng số), cấu hình bucket
và metadata SSL (gồm `lambda_edge`/`lambda_node` đã học). Trạng thái ở iteration
cuối được lưu thêm với hậu tố `_final`, ví dụ `ssl_model_state_final.pth`.

```text
checkpoints/<dataset>/<embedding>/
  ssl_base/         # SSL degree+edge base
  ssl_upgrade/      # SSL degree+edge upgrade
```
