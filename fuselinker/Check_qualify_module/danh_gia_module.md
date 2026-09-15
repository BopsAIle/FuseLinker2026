# Đánh giá module — task_node_mask (DistMult / mask entity)

Rà soát `model.py`, `model_base2.py`, `model_base4.py` (bản cũ, commit `84e94ad`)
và `task_degree_edge/model_base4.py`, nhằm trả lời: **module nào thực sự đóng góp,
module nào là gánh nặng, module nào đang chủ động làm hại?**

Bối cảnh: bản upgrade (`train_upgrade.py` + `model_base4.py`) cho kết quả kém hơn
mong đợi trên nhiệm vụ dự đoán entity bị mask, dù nó chứa toàn bộ backbone của
bản base cộng thêm nhánh PPR.

Kết luận ngắn: **nhánh PPR không tự nó vô dụng — nó bị 4 cơ chế trong chính
code này bóp nghẹt.** Ba trong bốn cơ chế đó khiến gradient chảy về nhánh phụ
gần như bằng 0, nên thứ duy nhất nó kịp đóng góp là nhiễu.

---

## Bảng tổng hợp

| # | Module | Vị trí (bản cũ) | Phán quyết | Vấn đề |
|---|--------|-----------------|------------|--------|
| 1 | `regularization_loss` trên embedding đã fuse | dòng 962 | **Sửa** | Phạt chính phần PPR cộng thêm → tối ưu hoá đẩy gate về 0 |
| 2 | Decoder của `TextEmbeddingAutoencoder` | dòng 519–527 | **Bỏ** | Không có reconstruction loss → gradient = 0, tham số chết |
| 3 | `BatchNorm1d` trong autoencoder text | dòng 516, 519 | **Bỏ** | Running-stats từ subgraph ~250 cạnh, áp lên toàn bộ 43k node lúc eval |
| 4 | `PPRGraphConv` aggregate bằng SUM | dòng 476–483 | **Sửa** | Độ lớn phụ thuộc số hàng xóm lấy được → lệch scale train/eval |
| 5 | `MLPProjector`, `FusionGate` | dòng 537, 554 | **Bỏ** | Code chết: định nghĩa nhưng không bao giờ được khởi tạo |
| 6 | Min-max cho `relation_weights` pretrained | thiếu | **Khôi phục** | `model.py` có, `model_base4.py` làm mất |
| 7 | `ResidualPPRFusion` | dòng 572 | **Giữ** | Nguyên tắc đúng, nhưng che mất tín hiệu chẩn đoán |
| 8 | Dựng Gppr (PPR sparse + cache) | dòng 313–458 | **Giữ** | Đúng, đắt, đã có cache — không đụng vào |
| 9 | Fusion `(1-w)*domain + w*text` | dòng 713 | **Giữ** | Base đang thắng bằng đúng cái này |

---

## 1. Regularization phạt đúng phần mình muốn thêm vào

**Đây là vấn đề nghiêm trọng nhất.**

```python
# model_base4.py (cũ), dòng 962
def regularization_loss(self, embeddings):
    return torch.mean(embeddings.pow(2)) + torch.mean(self.relation_weights.pow(2))
```

`embeddings` truyền vào là embedding **sau fusion**, tức `h_main + g * h_aux`.
Nhánh PPR chỉ có thể đóng góp bằng cách làm `h` lệch khỏi `h_main`, mà mọi độ
lệch đều làm tăng `mean(h²)`. Với `reg_param = 0.01`, hàm mục tiêu chứa một số
hạng phạt trực tiếp lên "mức độ model dám dùng PPR".

Cộng thêm hai yếu tố nữa:

- `ResidualPPRFusion` khởi tạo `fc.weight = 0`, `fc.bias = -3` → `g ≈ 0.047`.
  Nhánh phụ bắt đầu ở trạng thái gần như tắt.
- Đường duy nhất để mở gate là cải thiện BCE loss đủ nhiều để bù lại phần phạt.

Nói cách khác, cấu hình này đặt nhánh PPR vào thế: *xuất phát từ 0, và mỗi bước
tiến đều bị tính phí.* Điểm cân bằng tự nhiên là `g → 0`, tức đúng bằng base —
nhưng model vẫn phải mang thêm tham số của `ppr_branch`, và trong lúc gate chưa
kịp đóng hẳn thì nhánh phụ bơm nhiễu vào embedding mà DistMult phải chấm điểm.
Đó là con đường hợp lý nhất dẫn tới "upgrade ≤ base".

**Cách sửa** — `reg_on="rgcn"` (mặc định mới): phạt `h_rgcn` như FuseLinker gốc,
không phạt phần bổ sung. `reg_on="fused"` giữ lại để đối chứng.

---

## 2. Decoder của autoencoder không bao giờ chạy

```python
# model_base4.py (cũ), dòng 531
def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return encoded, decoded
```

Nơi gọi:

```python
# dòng 713
transformed_text_embeddings, _ = self.autoencoder(...)   # decoded bị vứt
```

`decoded` bị bỏ ở **mọi** nơi gọi, và không có reconstruction loss ở bất kỳ đâu
trong `get_loss` hay trong train script. Nghĩa là toàn bộ decoder — hai lớp
Linear `200 → 400 → 768` cộng BatchNorm, khoảng **389 nghìn tham số** — có
gradient bằng 0 vĩnh viễn. Chúng được Adam cấp state, được lưu vào checkpoint,
được `state_dict` mang theo, và không làm gì cả.

Module này tên là "autoencoder" nhưng thứ đang chạy chỉ là một MLP encoder.

**Cách sửa** — `TextEncoder(kind="mlp")` mặc định: giữ đúng phần thực sự hoạt
động. Muốn autoencoder thật thì `--text_encoder ae --text_recon_weight 0.1`,
lúc đó reconstruction loss mới thực sự được cộng vào `get_loss`.

---

## 3. BatchNorm giữa hai phân phối batch khác nhau

```python
# dòng 516
nn.BatchNorm1d(encoding_dim * 2),
```

`EmbeddingLayer` được gọi ở hai ngữ cảnh rất khác nhau:

| | Lúc train | Lúc eval |
|---|---|---|
| Nguồn node | subgraph sample từ `graph_batch_size=250` cạnh | toàn bộ 43.474 node |
| Số node mỗi lần gọi | vài trăm | hàng chục nghìn |

BatchNorm ước lượng `running_mean` / `running_var` từ cột trái rồi áp lên cột
phải. Hai phân phối này không có lý do gì để giống nhau: node trong subgraph
được chọn theo cạnh nên thiên lệch mạnh về node bậc cao.

Đáng chú ý: **bản `task_degree_edge/model_base4.py` đã sửa lỗi này** và ghi rõ
lý do trong comment (dòng 521–524) — LayerNorm chuẩn hoá theo từng sample nên
không phụ thuộc thành phần batch. Bản `task_node_mask` thì chưa được sửa theo.

**Cách sửa** — `text_norm="layer"` mặc định, `"batch"` vẫn chọn lại được.

---

## 4. SUM aggregation làm lệch scale train/eval

```python
# dòng 476–483
g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
```

Đường train và đường eval của nhánh PPR không giống nhau:

- Train (`_ppr_sampled`, dòng 883): neighbor sampling với `fanout=50`.
- Eval (`_ppr_full_cpu`, dòng 912): full graph, lấy hết hàng xóm.

Với SUM, độ lớn đầu ra tỉ lệ với số hàng xóm gom được. DistMult là tích ba
chiều `Σ h_s ⊙ r ⊙ h_o`, bậc hai theo `‖h‖`, nên rất nhạy với thay đổi scale
giữa train và test.

**Đo thực tế** (đồ thị 60 node, bậc PPR = 10, ép `fanout=3`):

| agg | `‖h_train‖` | `‖h_eval‖` | Lệch |
|-----|-------------|------------|------|
| sum | 4.3126 | 4.4606 | **3.3 %** |
| mean | 4.4720 | 4.4720 | **0.0 %** |

Cần nói cho đúng mức độ: **đây là lỗi nhỏ nhất trong 4 lỗi.** LayerNorm ở lớp
PPR cuối đã ghim `‖h‖` về `√d`, nên phần lệch còn lại chỉ đến từ hướng của
vector, không phải độ lớn. 3,3% không đủ để một mình giải thích việc upgrade
thua base — nhưng nó là lệch hệ thống, miễn phí để sửa, và không có lý do gì
để giữ.

**Cách sửa** — `ppr_agg="mean"` (chuẩn hoá theo tổng trọng số PPR) và
`ppr_fanout=-1`. Vì Gppr đã bị cắt top-k (mặc định 50) từ lúc dựng, lấy hết
hàng xóm không hề đắt hơn, mà lại khớp chính xác đường eval.

---

## 5. Code chết: `MLPProjector` và `FusionGate`

Hai lớp này được định nghĩa đầy đủ ở `task_node_mask/model_base4.py` dòng 537
và 554, kèm docstring giải thích ý tưởng — nhưng **không nơi nào trong file
khởi tạo chúng**. Kiểm chứng:

```bash
grep -n "MLPProjector\|FusionGate" task_node_mask/model_base4.py
# chỉ ra 3 dòng: 2 định nghĩa class + 1 lần nhắc trong comment
```

Trong khi đó `task_degree_edge/model_base4.py` **có** dùng chúng (dòng 694,
730). Hai file trùng tên, khác nội dung, một bên dùng một bên không — đây là
nguồn nhầm lẫn khi đọc code, nhất là khi cả hai cùng được gọi là "model_base4".

**Cách xử lý** — `FusionGate` được giữ lại nhưng **chỉ dựng khi
`--input_fusion gate`**, để nó là một lựa chọn ablation có thể đo được thay vì
code treo. `MLPProjector` bỏ hẳn: vai trò của nó (chiếu domain về `hidden_dim`)
đã được `poincare_to_euclidean` đảm nhiệm.

---

## 6. Mất bước chuẩn hoá relation embeddings

`model.py` (dòng ~190) có:

```python
normalized_relations = (self.relation_weights - min) / (max - min)
self.relation_weights.data.copy_(normalized_relations)
```

`model_base4.py` bỏ mất bước này. Hiện tại **chưa gây hậu quả** vì
`train_upgrade.py` không truyền `pretrained_relation_embeddings` (luôn là
`None` → đi nhánh xavier init). Nhưng đây là sai khác âm thầm giữa base và
upgrade, sẽ cắn khi nào có ai đó nạp relation embedding pretrained và không
hiểu vì sao hai bản cho scale khác nhau.

**Cách sửa** — khôi phục, đặt sau cờ `relation_minmax=True`.

---

## 7. `ResidualPPRFusion` — giữ, nhưng nó che mất tín hiệu chẩn đoán

Thiết kế `h_main + g*h_aux` với `g ≈ 0` lúc khởi tạo là **đúng về nguyên tắc**:
upgrade xuất phát từ đúng vị trí của base, nên về lý thuyết không thể tệ hơn.

Vấn đề là cái giá phải trả về mặt chẩn đoán: `g` là một vector `hidden_dim`
chiều, tính từ `fc([h_main, h_aux])` — nhìn vào không biết được model đang dùng
PPR nhiều hay ít. Khi upgrade hoà với base, ta không phân biệt được hai khả
năng hoàn toàn khác nhau:

- gate đã đóng → PPR vô dụng, nên bỏ nhánh phụ;
- gate mở nhưng đóng góp bị triệt tiêu → PPR có ích, lỗi nằm ở chỗ khác.

**Bổ sung** — thêm `ppr_fusion="scalar"`: `h_main + α*h_aux` với `α` là một
scalar học được, init 0. Sau train, đọc `α` là biết ngay. Thêm
`model.ppr_gate_value()` để in giá trị này vào log mỗi lần in loss.

Nếu sau 40k iteration `ppr_gate` vẫn nằm ở giá trị khởi tạo (≈ 0.047), kết
luận là nhánh PPR không học được gì và nên bỏ hẳn.

---

## 8–9. Những gì giữ nguyên, không đụng tới

**Dựng Gppr** (`compute_ppr_sparse`, `load_or_compute_ppr_sparse`, dòng
313–458): cài đặt đúng theo công thức HGDC `S = (1-c)(I - cM)^{-1}`, có ba
đường tính (dense GPU, batched DGL, streaming sparse) chọn theo bộ nhớ khả
dụng, có cache ra file. Đây là phần đắt nhất và cũng là phần viết cẩn thận
nhất. Giữ nguyên từng dòng.

**Fusion `(1-w)*domain + w*text`**: không có bằng chứng nào cho thấy nó là vấn
đề — ngược lại, bản base đang thắng bằng đúng công thức này. Giữ làm mặc định;
`FusionGate` chỉ là lựa chọn thay thế có thể bật lên để đo.

**Min-max toàn ma trận cho hai embedding đầu vào**: đặc trưng của FuseLinker
gốc. Về mặt lý thuyết min-max theo toàn ma trận (thay vì theo từng chiều) là
lựa chọn lạ, nhưng nó thuộc về phần đang hoạt động tốt, nên không đổi.

**DistMult + `relation_weights`**: giữ để checkpoint cũ và `calc_mrr` /
`calc_auroc` dùng lại được.

---

## Kiểm chứng đã làm

- 19 cấu hình ablation trên đồ thị sinh ngẫu nhiên (60 node, 300 cạnh): train 2
  iteration + eval full graph, tất cả đều chạy, không nhánh nào chết.
- Đo lệch scale train/eval cho `sum` vs `mean` (bảng ở mục 4).
- Chạy end-to-end thật trên `suppkg` (43.474 node, 305.986 cạnh, GPU) với
  `--use_ppr false`: train + ranking + AUROC + lưu ROC, exit code 0.
- `--help` hiển thị đủ ba nhóm cờ ablation; `EncoderConfig.from_args` map đúng.

## Chưa kiểm chứng — đọc phần này trước khi tin bản mới

1. **Chưa có bằng chứng bản mới thắng base.** Tài liệu này chỉ ra bốn cơ chế
   đang cản nhánh PPR và gỡ chúng. Việc PPR có thực sự bổ sung thông tin cho
   DistMult hay không thì chỉ số liệu mới trả lời được.
2. **Chưa chạy đủ seed.** Mọi so sánh cần ≥ 3 seed mỗi cấu hình. Trước khi có
   khoảng dao động giữa các seed, không thể phân biệt "module có ích" với
   "nhiễu may mắn". `--seed` đã được thêm vào `train_upgrade.py`.
3. **Checkpoint cũ không nạp được.** Tên tham số đã đổi (`autoencoder.*` →
   `text_encoder.*`), nên các checkpoint trong `checkpoints/*/modelbase/` phải
   train lại. `load_checkpoint` phát hiện và báo lỗi rõ ràng thay vì fail khó
   hiểu.

## Lưới thí nghiệm đề xuất

Mỗi dòng chạy với `--seed 1 2 3`, so trung bình ± độ lệch:

```bash
# 1. Bản mới vs bản cũ vs base
train_upgrade.py --seed S
train_upgrade.py --seed S --text_encoder ae --text_norm batch \
                 --ppr_agg sum --ppr_fanout 50 --reg_on fused    # tái hiện bản cũ
train_base.py    --seed S

# 2. Tách riêng từng lỗi — bật lại đúng một cái mỗi lần
train_upgrade.py --seed S --reg_on fused                 # lỗi 1
train_upgrade.py --seed S --text_encoder ae              # lỗi 2
train_upgrade.py --seed S --text_norm batch              # lỗi 3
train_upgrade.py --seed S --ppr_agg sum --ppr_fanout 50  # lỗi 4

# 3. Nhánh PPR có đáng giữ không
train_upgrade.py --seed S --use_ppr false
train_upgrade.py --seed S --ppr_fusion scalar   # đọc alpha sau train
train_upgrade.py --seed S --ppr_fusion replace  # PPR một mình mạnh cỡ nào

# 4. Backbone
train_upgrade.py --seed S --use_text false
train_upgrade.py --seed S --use_domain false
train_upgrade.py --seed S --text_encoder linear
train_upgrade.py --seed S --input_fusion gate
```

Thứ tự ưu tiên nếu không đủ thời gian chạy hết: nhóm 1 → nhóm 3 → nhóm 2.
Nhóm 2 chỉ có giá trị giải thích; nhóm 1 và 3 mới trả lời câu hỏi "có nên giữ
nhánh PPR không".
