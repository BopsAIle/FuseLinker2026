# FuseLinker: hướng dẫn hai nhiệm vụ

FuseLinker học biểu diễn node trên knowledge graph bằng R-GCN, kết hợp embedding
văn bản và embedding tri thức. Mã nguồn được tách thành hai nhiệm vụ độc lập:

| Thư mục | Mục tiêu | Hàm mất mát và độ đo |
|---|---|---|
| [`task_node_mask/`](task_node_mask/) | Dự đoán entity (head hoặc tail) bị che trong một triple | DistMult + BCE; MR, MRR, Hits@K, AUROC |
| [`task_degree_edge/`](task_degree_edge/) | Tự giám sát: dự đoán loại cạnh bị mask và bucket bậc của node | CE cạnh + CE bậc; macro AUROC, Macro-F1 |
| [`common/`](common/) | Mã dùng chung: nạp dữ liệu, lấy mẫu graph, xử lý đường dẫn và tính độ đo | Không chạy trực tiếp |

Hai thư mục nhiệm vụ có `model.py` và script train riêng. Vì vậy có thể thay đổi
kiến trúc của một nhiệm vụ mà không làm thay đổi luồng train của nhiệm vụ còn lại.

## 1. Chuẩn bị môi trường và dữ liệu

Yêu cầu chính: Python 3.10+, PyTorch, DGL, NumPy, pandas, scikit-learn và
matplotlib. Nên chạy các lệnh từ thư mục này:

```bash
cd FuseLinker/fuselinker
```

Mỗi dataset nằm trong một thư mục con, ví dụ `hetionet/`, và cần tối thiểu:

```text
hetionet/
  train.tsv
  valid.tsv
  test.tsv
  pubmedbert_pretrained_embeddings_768.npy
  poincare_embeddings.npy
```

- Mỗi dòng TSV là một triple `head<TAB>relation<TAB>tail`, không có header.
- Mỗi file `.npy` là ma trận hai chiều. Hàng thứ `i` phải thuộc entity có ID
  `i` trong ánh xạ `entity2index`.
- Có thể đổi tên file embedding qua `--text_embedding_file` và
  `--knowledge_embedding_file`.
- Nếu không nạp được embedding, chương trình dùng embedding khởi tạo ngẫu nhiên.
  Vì vậy cần kiểm tra log để tránh vô tình train sai đầu vào.
- Khi đọc dữ liệu, chương trình sinh các file ánh xạ như `entity2index.pkl` và
  `relation2index.pkl` ngay trong thư mục dataset.

Mọi đường dẫn tương đối được resolve theo thư mục `fuselinker/`, không phụ thuộc
thư mục hiện tại của terminal:

- `--data hetionet` tương ứng với `fuselinker/hetionet/`;
- `--model_state_file checkpoints/...` tương ứng với
  `fuselinker/checkpoints/...`;
- thư mục cha của checkpoint được tự động tạo trước khi lưu.

Để xem toàn bộ tham số và giá trị mặc định của một script:

```bash
python task_node_mask/train_base.py --help
python task_degree_edge/train_base.py --help
```

## 2. Nhiệm vụ 1: dự đoán entity bị mask

### Mô tả nhiệm vụ

Với triple đúng `(head, relation, tail)`, mô hình che head hoặc tail và phải xếp
entity đúng cao hơn các entity âm. Quy trình train gồm:

1. Lấy một batch cạnh từ `train.tsv`.
2. Giữ một phần cạnh làm graph truyền thông điệp R-GCN; phần còn lại là các
   triple dương để học DistMult.
3. Sinh triple âm bằng cách thay head hoặc tail.
4. Kết hợp embedding text và domain, mã hóa node bằng R-GCN.
5. Tính điểm DistMult:
   `score(h, r, t) = sum(h ⊙ r ⊙ t)`.
6. Tối ưu binary cross-entropy trên triple dương/âm, cộng regularization L2.

Đây là link prediction/masked-entity prediction, không phải bài toán phân loại
loại quan hệ của nhiệm vụ 2.

### Hai biến thể

- **Base** — `task_node_mask/train_base.py`, encoder trong `model.py`:
  embedding domain và text được chiếu về cùng số chiều, sau đó trộn theo
  `(1-w) * domain + w * text` và đưa qua R-GCN.
- **Upgrade** — `task_node_mask/train_upgrade.py`, encoder trong
  `model_base4.py`: cùng backbone DistMult với base (min-max, trộn
  `(1-w) * domain + w * text`, R-GCN), thêm nhánh PPR residual.
- `train_upgrade_base2.py` chỉ phục vụ kiến trúc/checkpoint cũ.

### Lệnh train base

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

### Lệnh train upgrade

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

Các tham số quan trọng:

- `--graph_batch_size`: số cạnh được lấy mẫu mỗi iteration.
- `--graph_split_size`: tỷ lệ cạnh mẫu dùng cho message passing; phần còn lại
  dùng làm triple dương DistMult.
- `--negative_sample`: số mẫu âm khi train.
- `--n_hidden`, `--num_hidden_layers`, `--dropout`: cấu hình encoder.
- `--w`: prior khi trộn embedding text/domain; ở base là trọng số text, ở
  upgrade được kết hợp với gate học được.
- `--freeze`: không cập nhật hai embedding đầu vào.
- `--use_ppr`, `--ppr_c`, `--ppr_eps`, `--ppr_num_layers`: cấu hình nhánh PPR
  của bản upgrade.

Sau train, script lưu checkpoint, đánh giá trên `test.tsv`, in:

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

Dùng `--model_module model_base4` và đúng cấu hình PPR để đánh giá checkpoint
upgrade. Có thể vẽ ROC của nhiều dataset/biến thể bằng:

```bash
python task_node_mask/plot_roc_comparision.py \
  --root . \
  --datasets hetionet suppkg kegg50k \
  --variants model modelbase \
  --output roc_comparison.png \
  --skip-missing
```

## 3. Nhiệm vụ 2: SSL dự đoán cạnh và bậc node

### Mô tả nhiệm vụ

Đây là bước pretrain tự giám sát gồm hai đầu phân loại dùng chung encoder:

1. **Masked-edge classification**: bỏ ngẫu nhiên một tỷ lệ cạnh khỏi graph
   message passing. Từ `concat(h_src, h_dst)`, head phân loại dự đoán một trong
   `R` loại quan hệ thật hoặc lớp thứ `R+1` dành cho cạnh âm/fake.
2. **Node-degree classification**: tính bậc node trên graph train, chia thành
   `K` bucket và dự đoán bucket từ embedding node. Có hai cách chia:
   `log` (theo log2) và `quantile`.

Loss tổng là:

```text
loss = cross_entropy(edge_logits, edge_labels)
     + cross_entropy(degree_logits, degree_labels)
```

Degree loss có class weight tỷ lệ nghịch căn bậc hai tần suất để giảm ảnh hưởng
mất cân bằng giữa các bucket. Checkpoint tốt nhất được chọn theo edge CE trên
tập validation.

### Hai biến thể

- **Base** — `task_degree_edge/train_base.py` + `model.py`: R-GCN và phép trộn
  embedding tuyến tính. Task bậc mặc định học trên subgraph (LukePi).
- **Upgrade** — `task_degree_edge/train_upgrade.py` + `model_base4.py`:
  FusionGate và nhánh PPR. Mặc định degree head học trên toàn graph train
  (`--degree_on_full true`) để mọi node nhận supervision. TEST encode trên
  graph train (nhãn bậc từ `train.tsv`).

### Lệnh train base

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

### Lệnh train upgrade

```bash
python task_degree_edge/train_upgrade.py \
  --data hetionet \
  --text_embedding_file pubmedbert_pretrained_embeddings_768.npy \
  --knowledge_embedding_file poincare_embeddings.npy \
  --num_hidden_layers 2 \
  --iterations 40000 \
  --evaluate_every 100 \
  --graph_batch_size 250 \
  --mask_rate 0.2 \
  --negative_sample 1 \
  --num_degree_bins 10 \
  --degree_bin_strategy log \
  --degree_on_full true \
  --use_ppr true \
  --ppr_iterative true \
  --w 0.75 \
  --ssl_model_state_file checkpoints/hetionet/pubmedbert/ssl_upgrade/ssl_model_state.pth
```

Các tham số riêng quan trọng:

- `--mask_rate`: tỷ lệ cạnh dương bị loại khỏi graph encode.
- `--negative_sample`: số cạnh âm trên mỗi cạnh dương bị mask.
- `--num_degree_bins`: số lớp bậc node.
- `--degree_bin_strategy {log,quantile}`: cách tạo bucket bậc.
- `--evaluate_every`: chu kỳ encode graph train đầy đủ và đánh giá validation.
  TEST cuối cũng encode trên graph train.
- `--eval_neg_rate`: tỷ lệ âm khi đánh giá cạnh; mặc định bằng
  `--negative_sample`.
- `--eval_max_triples`: giới hạn triple validation/test mỗi lần đánh giá để
  kiểm soát thời gian và bộ nhớ.
- `--degree_on_full`: upgrade mặc định `true`; base mặc định `false` (LukePi).
- các tham số `--use_ppr`, `--ppr_*` có ý nghĩa như trong nhiệm vụ 1 upgrade.

Log train/test tách 2 task (kèm `aupr` / `bacc`):

- `loss`: CE cạnh hoặc CE degree (thô), thấp hơn tốt hơn;
- `auc`: macro one-vs-rest AUROC;
- `aupr`: macro average precision;
- `f1`: Macro-F1;
- `bacc`: balanced accuracy.

Script lưu checkpoint tốt nhất tại `--ssl_model_state_file`. Checkpoint chứa
encoder, `edge_head`, `degree_head`, cấu hình bucket và metadata SSL. Trạng thái
ở iteration cuối được lưu thêm với hậu tố `_final`, ví dụ
`ssl_model_state_final.pth`.

## 4. Layout checkpoint khuyến nghị

```text
checkpoints/<dataset>/<embedding>/
  model/            # nhiệm vụ 1 base
  modelbase/        # nhiệm vụ 1 upgrade (model_base4)
  model_base2/      # nhiệm vụ 1 legacy
  ssl_base/         # nhiệm vụ 2 base
  ssl_upgrade/      # nhiệm vụ 2 upgrade
```

Đọc thêm các ghi chú riêng tại
[`task_node_mask/README.md`](task_node_mask/README.md) và
[`task_degree_edge/README.md`](task_degree_edge/README.md).
