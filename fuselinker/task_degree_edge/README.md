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
  Mặc định degree head học trên subgraph (`--degree_on_full false`), giống base.
  PPR chạy trên graph đã mask bằng phép lan truyền đặc trưng, không dựng PPR toàn graph train.

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
  --degree_on_full false \
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
- `--degree_on_full`: cả hai script mặc định `false` để so sánh cùng supervision.
  Bật `true` cho cả hai nếu cần giám sát toàn graph; chế độ này có gradient encoder
  và tốn VRAM hơn.
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

## Benchmark kiến trúc có kiểm soát

`benchmark.py` chạy một model mỗi lần, mặc định SUPPKG + PubMedBERT, 50 bước,
seed 42, batch 1024, mask 0.2, một negative, hidden 200, hai lớp R-GCN,
10 bucket quantile và w=0.75. Mỗi bước dùng cùng batch cho mọi kiến trúc.
Checkpoint chọn theo CE cạnh validation trên cùng 5.000 cạnh và mẫu âm cố định.
Không chấm test nếu chưa bật `--test`. Ví dụ chạy từ `fuselinker/`:

```powershell
python task_degree_edge/benchmark.py --model task_degree_edge/model.py --output experiments/degree_edge/base42.json
python task_degree_edge/benchmark.py --model task_degree_edge/model_base4.py --ppr --output experiments/degree_edge/upgrade42.json
```

Chỉ sau khi chốt kiến trúc, dùng `--checkpoint ...pth --test --output ...json`
để chấm checkpoint đã chọn. Lặp lại `--seed 7` và `--seed 2026` để kiểm tra
độ nhạy với khởi tạo; không chọn seed tốt nhất để báo cáo.

### Thay đổi tương thích cần biết

- FusionGate khởi tạo đúng `w`, sau đó học offset trong không gian logit.
- PPR tính `S X` bằng lặp `h = c M h + (1-c) X` trên graph hiện tại, gồm
  self-loop. Train chỉ nhìn cạnh còn lại sau mask; validation/test chỉ nhìn
  graph train. Cách này không cần ma trận N×N hoặc lấy mẫu fanout ở nhánh PPR.
  Các cạnh song song được tính theo multiplicity, không sparsify theo top-k.
- `--ppr_c`, `--ppr_num_layers`, `--ppr_iter_num` còn hiệu lực; mặc định 8 vòng.
  `--ppr_eps`, `--ppr_batch_size`, `--ppr_iterative` chỉ còn nhận để tương thích
  lệnh cũ. Các helper dựng PPR tường minh vẫn được giữ để dùng độc lập.
- Edge head giữ đường concat có thứ tự và bổ sung đường tương tác tích/độ lệch
  giữa hai endpoint. Cần train lại head; checkpoint SSL cũ không tương đương
  kiến trúc mới dù một số tên trọng số encoder còn giống nhau.
- `--seed` mặc định 42 cho cả hai script. Validation và test dùng RNG riêng cố định.
- `encode_full_train_graph` chỉ ngắt gradient khi eval; degree loss khi train
  full graph giờ truyền được về encoder.
- `train_base.py` / `train_upgrade.py` vẫn in TEST của iteration cuối theo
  hành vi cũ. Benchmark chấm checkpoint tốt nhất trên validation.
- Degree metric hiện đánh giá bucket của các node trên graph train; không phải
  phép đánh giá tổng quát hóa sang node/graph chưa thấy.

Kiểm tra hồi quy trên CPU:

```powershell
python -m unittest discover -s task_degree_edge -p test_model_base4.py -v
```
