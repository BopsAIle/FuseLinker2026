# ĐẶC TẢ ABLATION STUDY CHO MODEL V4

## 1. Mục tiêu

Mục tiêu của tài liệu này là hướng dẫn thực hiện ablation study giữa:

- `model.py`: phiên bản baseline ban đầu.
- `model_base4_best_v4.py`: phiên bản V4 hoàn chỉnh.

Mục tiêu chính không phải chỉ xem V4 có tốt hơn `model.py` hay không, mà là xác định **từng module được bổ sung trong V4 có thực sự cải thiện chất lượng node embedding hay không**.

Nguyên tắc quan trọng:

> Mỗi thí nghiệm chỉ thay đổi MỘT module tại một thời điểm.  
> Tất cả các thành phần còn lại, dữ liệu, seed, optimizer, learning rate, số epoch, negative sampling, batch size và cách đánh giá phải được giữ nguyên.

---

# 2. Hai mô hình gốc cần giữ

## 2.1. Baseline: `model.py`

Pipeline chính:

```text
Text embedding
    ↓
TextEmbeddingAutoencoder
    ┐
    ├── fixed weighted average theo w
    │
Domain / medical embedding
    ↓
Linear projection
    ┘
    ↓
RGCN
    ↓
DistMult
    ↓
BCE loss + regularization
```

Các đặc điểm chính:

- Text embedding:
  - Min-max normalize về `[0,1]`.
  - Qua `TextEmbeddingAutoencoder`.
- Domain/medical embedding:
  - Min-max normalize về `[0,1]`.
  - Qua một `Linear`.
- Fusion:
  - Fixed weighted average:

```python
combined = (1 - w) * domain + w * text
```

- RGCN:
  - DGL `RelGraphConv`.
  - `regularizer="bdd"`.
  - ReLU ở tất cả layer trừ layer cuối.
- Không có residual alpha.
- Không có PPR.
- Không có semantic alignment loss.
- Scorer là DistMult.

---

## 2.2. Full model: `model_base4_best_v4.py`

Pipeline chính:

```text
Text / node-name embedding
        ↓
TextEmbeddingAutoencoder
        │
        ├───────────────────────────┐
                                    ↓
                              Adaptive Fusion
                                    │
Medical / domain embedding          │
        ↓                           │
Optional Poincare log-map           │
        ↓                           │
MLP Projector ──────────────────────┘
                 ↓
         Semantic embedding
                 ↓
       RGCN + residual alpha
                 ↓
              h_RGCN
                 │
                 ├─────────────────────┐
                 │                     │
                 │               CUDA PPR top-50
                 │                     │
                 └──── Adaptive Fusion ┘
                           ↓
                  Final node embedding
                           ↓
                        DistMult
                           ↓
       BCE + regularization + alignment loss
```

---

# 3. Các module chính cần ablation

Có 5 module chính bắt buộc nên kiểm tra.

---

# MODULE 1 — Medical / Domain Semantic Projector

## V4 Full

Medical/domain embedding được xử lý:

```text
Domain embedding
    ↓
Optional PoincareLogMap0
    ↓
LayerNorm
    ↓
Linear(input_dim → 2H)
    ↓
GELU
    ↓
Linear(2H → H)
    ↓
LayerNorm
```

Module tương ứng:

```python
MLPProjector
```

## Ablation

Tên thí nghiệm:

```text
V4 - Semantic Projector
```

Thay `MLPProjector` bằng một phép biến đổi đơn giản giống `model.py`:

```python
nn.Linear(domain_input_dim, hidden_dim)
```

Không thay đổi các phần khác.

## Câu hỏi nghiên cứu

> MLP projector có giúp chuyển medical/domain embedding sang latent space chung tốt hơn so với một phép chiếu tuyến tính đơn giản hay không?

---

# MODULE 2 — Adaptive Text/Medical Fusion

## V4 Full

V4 dùng learned feature-wise gate.

Input của gate:

```python
[
    text,
    medical,
    abs(text - medical),
    text * medical
]
```

Sau đó:

```python
gate = sigmoid(MLP(features))

fused = gate * text + (1 - gate) * medical
```

Gate khác nhau theo:

- từng node;
- từng dimension.

## Ablation

Tên thí nghiệm:

```text
V4 - Adaptive Fusion
```

Bỏ `FusionGate` ở bước text/medical fusion.

Thay bằng weighted average đơn giản.

Khuyến nghị:

```python
fused = 0.5 * text + 0.5 * medical
```

hoặc sử dụng đúng `w` của baseline:

```python
fused = w * text + (1 - w) * medical
```

Nếu dùng `w`, phải cố định cùng một giá trị trong tất cả các run.

Không được thay đổi RGCN, PPR hoặc loss.

## Câu hỏi nghiên cứu

> Learned adaptive fusion có giúp kết hợp node-name/text embedding và medical knowledge embedding tốt hơn fixed weighted average hay không?

---

# MODULE 3 — Semantic Alignment Loss

## V4 Full

V4 thêm alignment giữa hai semantic views của cùng node.

```python
text_n = normalize(text_x)
domain_n = normalize(domain_x)

alignment_loss =
    mean(1 - cosine_similarity(text_n, domain_n))
```

Loss đầy đủ:

```text
L =
    BCE
    + lambda_reg * regularization
    + lambda_align * alignment_loss
```

Mặc định V4:

```python
semantic_alignment_weight = 0.01
```

## Ablation

Tên thí nghiệm:

```text
V4 - Alignment Loss
```

Đặt:

```python
semantic_alignment_weight = 0.0
```

hoặc loại bỏ hoàn toàn term alignment khỏi `get_loss()`.

Không thay đổi kiến trúc.

## Câu hỏi nghiên cứu

> Việc ép text representation và medical representation của cùng node trở nên tương thích về semantic space có giúp final embedding tốt hơn hay không?

---

# MODULE 4 — LukePi-inspired Residual Alpha trong RGCN

## V4 Full

Mỗi RGCN layer có output:

```python
h_new = RelGraphConv(...)(x)

alpha = sigmoid(alpha_logit)

h_out =
    alpha * h_new
    + (1 - alpha) * x
```

Mỗi layer có một scalar `alpha` riêng.

Đây là phần cải tiến nhỏ lấy cảm hứng từ adaptive residual connection trong LuKePi.

## Ablation

Tên thí nghiệm:

```text
V4 - Residual Alpha
```

Bỏ residual alpha:

```python
h_out = h_new
```

Tức RGCN quay về gần `model.py`.

Phải giữ:

```python
regularizer="bdd"
```

và activation giống V4/model.py:

```text
ReLU ở layer trung gian
No activation ở layer cuối
```

Không được thêm LayerNorm, FFN hoặc Jumping Knowledge.

## Câu hỏi nghiên cứu

> Learnable residual alpha có giúp giữ lại semantic information trước graph propagation và giảm nguy cơ over-smoothing hay không?

---

# MODULE 5 — CUDA PPR Top-50 Structural Branch

## V4 Full

V4 có thêm một graph structural branch dựa trên Personalized PageRank.

Các đặc điểm:

- Dùng topology không phân biệt relation type.
- Tạo adjacency dạng undirected.
- PPR chạy bằng PyTorch CUDA.
- Không tạo dense matrix `N x N`.
- Chạy theo block seed.
- Mặc định:

```python
c = 0.85
num_iter = 30
topk = 50
batch_size = 128
```

- Mỗi center node giữ tối đa 50 PPR neighbors mạnh nhất.
- Bỏ self node khỏi final top-50.
- Edge direction:

```text
PPR_neighbor → center
```

để center nhận message từ PPR neighbors.

## Ablation

Tên thí nghiệm:

```text
V4 - PPR
```

Bỏ hoàn toàn:

- `compute_ppr_sparse`
- `PPRGraphConv`
- `PPRBranch`
- PPR/RGCN fusion

Final embedding lúc đó:

```python
final_embedding = h_rgcn
```

Không thay đổi semantic fusion hoặc RGCN.

## Câu hỏi nghiên cứu

> Global structural information từ top-50 PPR neighbors có bổ sung thông tin hữu ích ngoài relation-aware local information của RGCN hay không?

---

# 4. Module phụ nên kiểm tra sau khi 5 ablation chính hoàn thành

## MODULE 6 — Learned PPR/RGCN Fusion

Module này chỉ cần chạy nếu thí nghiệm `V4 - PPR` chứng minh rằng PPR có ích.

### V4 Full

```python
final_embedding =
    FusionGate(h_ppr, h_rgcn)
```

### Ablation

Tên:

```text
V4 - Adaptive PPR Fusion
```

Giữ PPR branch nhưng thay learned fusion bằng fixed fusion:

```python
final_embedding =
    0.5 * h_rgcn
    + 0.5 * h_ppr
```

hoặc:

```python
final_embedding =
    h_rgcn + lambda_ppr * h_ppr
```

với `lambda_ppr` cố định.

### Câu hỏi

> PPR có ích là do bản thân thông tin PPR, hay adaptive fusion giữa PPR và RGCN cũng đóng góp thêm?

---

# 5. Bảng thí nghiệm chính cần chạy

## Primary Leave-One-Out Ablation

| Experiment | Projector | Adaptive Text/Medical Fusion | Alignment | Residual Alpha | PPR |
|---|:---:|:---:|:---:|:---:|:---:|
| V4 Full | ✓ | ✓ | ✓ | ✓ | ✓ |
| V4 - Projector | ✗ | ✓ | ✓ | ✓ | ✓ |
| V4 - Adaptive Fusion | ✓ | ✗ | ✓ | ✓ | ✓ |
| V4 - Alignment | ✓ | ✓ | ✗ | ✓ | ✓ |
| V4 - Residual Alpha | ✓ | ✓ | ✓ | ✗ | ✓ |
| V4 - PPR | ✓ | ✓ | ✓ | ✓ | ✗ |

Đây là bảng ablation chính.

---

# 6. Incremental Study bổ sung

Ngoài leave-one-out ablation, nên có một chuỗi thêm module từ baseline.

Khuyến nghị:

```text
M0 = Clean model.py baseline

M1 = M0
     + Semantic Projector

M2 = M1
     + Adaptive Text/Medical Fusion

M3 = M2
     + Residual Alpha RGCN

M4 = M3
     + PPR top-50

M5 = M4
     + Semantic Alignment Loss

M5 ≈ V4 Full
```

Bảng:

| Model | Projector | Adaptive Fusion | Residual Alpha | PPR | Alignment |
|---|:---:|:---:|:---:|:---:|:---:|
| M0 | ✗ | ✗ | ✗ | ✗ | ✗ |
| M1 | ✓ | ✗ | ✗ | ✗ | ✗ |
| M2 | ✓ | ✓ | ✗ | ✗ | ✗ |
| M3 | ✓ | ✓ | ✓ | ✗ | ✗ |
| M4 | ✓ | ✓ | ✓ | ✓ | ✗ |
| M5 / V4 | ✓ | ✓ | ✓ | ✓ | ✓ |

Mục đích:

> Cho thấy quá trình thêm dần từng module có làm performance tăng theo xu hướng hay không.

---

# 7. Clean Baseline rất quan trọng

Không nên chỉ so:

```text
original model.py
vs
V4
```

vì V4 còn có một số implementation fix.

Nên tạo:

```text
model_clean.py
```

Giữ nguyên kiến trúc `model.py`, nhưng chỉ sửa các lỗi kỹ thuật.

Ví dụ:

- Sử dụng named arguments đúng cho:
  - `use_self_loop`
  - `use_cuda`
- Giữ cùng random seed.
- Thống nhất initialization khi cần.
- Không thêm:
  - Projector mới
  - Fusion gate
  - Alignment loss
  - Residual alpha
  - PPR

Clean baseline vẫn phải là:

```text
Text Autoencoder
+
Domain Linear
+
Fixed weighted average
+
Original RGCN
+
DistMult
```

---

# 8. Không được thay đổi đồng thời các preprocessing detail khi ablation module

V4 và `model.py` còn khác nhau ở các chi tiết sau:

- V4 bỏ min-max normalization của text embedding.
- V4 bỏ min-max normalization của domain embedding.
- V4 bỏ min-max normalization pretrained relation embedding.
- Random relation initialization có khác.
- `reshape(-1)` thay `squeeze()`.
- Self-loop argument được truyền rõ ràng hơn.

Các chi tiết này KHÔNG nên lẫn vào ablation module chính.

Khuyến nghị:

> Khi so sánh module, dùng cùng một clean preprocessing/configuration cho tất cả variant.

Nếu muốn kiểm tra preprocessing, tạo ablation riêng.

---

# 9. Optional Preprocessing Ablations

Chỉ chạy sau khi architecture ablation hoàn thành.

## 9.1. Raw embedding vs Min-Max normalization

```text
V4 raw pretrained embeddings
vs
V4 + min-max [0,1]
```

## 9.2. Relation embedding normalization

```text
raw pretrained relation embedding
vs
min-max pretrained relation embedding
```

## 9.3. LayerNorm trong semantic projector/fusion

Có thể kiểm tra:

```text
V4 Full
vs
V4 - semantic LayerNorm
```

Nhưng đây không phải ablation ưu tiên ban đầu.

---

# 10. Những thứ KHÔNG nên coi là module chính

Không cần chạy riêng ngay từ đầu cho:

- `reshape(-1)` vs `squeeze()`
- shape validation
- checkpoint `strict=True/False`
- tên class
- comments
- API compatibility parameter `w`
- API compatibility parameter `ppr_prior`
- decoder method helper
- device validation

Đây chủ yếu là implementation details.

---

# 11. Metrics phải đánh giá

Không chỉ đánh giá pretraining loss.

## 11.1. Pretraining / KG metrics

Nếu pipeline hỗ trợ:

```text
MR
MRR
Hits@1
Hits@3
Hits@10
AUROC
AUPRC
```

## 11.2. Downstream metrics

Vì mục tiêu thật sự là tạo universal node embeddings cho downstream tasks, cần ưu tiên:

```text
AUROC
AUPRC
F1
Accuracy
Precision
Recall
```

tùy downstream task.

---

# 12. Ba evaluation regimes bắt buộc nên tách

Đánh giá riêng:

## Vanilla

Cả hai node có thể đã xuất hiện trong downstream training.

## Weak Cold Start

Một node trong cặp test chưa xuất hiện trong downstream training.

## Cold Start

Cả hai node trong cặp test chưa xuất hiện trong downstream training.

Điều này đặc biệt quan trọng để kiểm tra:

> Module có thực sự tăng khả năng generalization hay chỉ giúp memorize các node đã thấy.

---

# 13. Random Seed

Không nên kết luận từ một run.

Khuyến nghị tối thiểu:

```text
5 seeds
```

Ví dụ:

```text
42
123
3407
2025
2026
```

Báo kết quả:

```text
mean ± std
```

Ví dụ:

```text
V4 Full:
AUROC = 0.891 ± 0.003

V4 - PPR:
AUROC = 0.879 ± 0.004
```

---

# 14. Fair Comparison Checklist

Mỗi variant phải dùng cùng:

```text
Dataset
Train/validation/test split
Negative samples
Random seeds
Hidden dimension
Number of RGCN layers
Batch size
Optimizer
Learning rate
Weight decay
Epochs
Early stopping rule
Relation count
Evaluation code
Downstream classifier
Threshold selection
```

Ngoại trừ parameter/module đang được ablate.

---

# 15. Output mà AI thực hiện ablation phải tạo

AI nên tạo các file model riêng, ví dụ:

```text
model_v4_full.py

model_v4_no_projector.py

model_v4_no_semantic_fusion.py

model_v4_no_alignment.py

model_v4_no_residual_alpha.py

model_v4_no_ppr.py

model_v4_fixed_ppr_fusion.py
```

Không chỉnh nhiều module trong cùng một file.

---

# 16. Quy tắc khi tạo từng ablation model

## `model_v4_no_projector.py`

Chỉ:

```text
MLPProjector
→ Linear
```

Mọi phần khác y hệt V4.

---

## `model_v4_no_semantic_fusion.py`

Chỉ:

```text
FusionGate(text, domain)
→ fixed weighted average
```

Mọi phần khác y hệt V4.

---

## `model_v4_no_alignment.py`

Chỉ:

```python
semantic_alignment_weight = 0
```

Mọi phần khác y hệt V4.

---

## `model_v4_no_residual_alpha.py`

Chỉ:

```python
alpha * h_new + (1-alpha) * x
```

thành:

```python
h_new
```

Mọi phần khác y hệt V4.

---

## `model_v4_no_ppr.py`

Chỉ bỏ toàn bộ PPR branch.

Final embedding:

```python
return h_rgcn
```

Mọi semantic/RGCN component khác giữ nguyên.

---

# 17. Thứ tự ưu tiên chạy experiment

Nếu tài nguyên hạn chế, chạy theo thứ tự:

```text
1. Clean model.py baseline

2. V4 Full

3. V4 - PPR

4. V4 - Adaptive Fusion

5. V4 - Residual Alpha

6. V4 - Projector

7. V4 - Alignment

8. V4 - Adaptive PPR Fusion
```

Lý do:

- PPR, semantic fusion và residual RGCN là thay đổi cấu trúc lớn.
- Alignment loss là objective phụ nhỏ.
- PPR fusion chỉ cần kiểm tra sau khi biết PPR hữu ích.

---

# 18. Cách diễn giải kết quả

Ví dụ:

```text
V4 Full              = 0.900
V4 - PPR             = 0.882
V4 - Adaptive Fusion = 0.870
V4 - Residual Alpha  = 0.891
V4 - Projector       = 0.886
V4 - Alignment       = 0.896
```

Có thể diễn giải:

```text
Adaptive Fusion:
largest degradation
→ đóng góp lớn nhất trong cấu hình V4.

PPR:
second largest degradation
→ global structural information có tác dụng rõ.

Residual Alpha:
moderate improvement
→ giúp graph refinement ổn định hơn.

Projector:
moderate improvement
→ nonlinear medical projection có lợi.

Alignment:
small but consistent improvement
→ objective phụ có lợi nhưng không phải yếu tố chính.
```

Không nên kết luận chỉ dựa vào một seed.

---

# 19. Không được kết luận module có lợi chỉ từ pretraining metric

Ví dụ có thể xảy ra:

```text
PPR:
Pretraining MRR ↑
Downstream Cold-start AUROC ↓
```

Trong trường hợp đó không thể kết luận:

> PPR tạo embedding tốt hơn.

Mục tiêu cuối cùng là downstream node embedding quality.

Do đó kết luận module cần dựa chủ yếu vào:

```text
Vanilla
+
Weak cold-start
+
Cold-start
```

trên downstream tasks.

---

# 20. Thiết kế ablation cuối cùng được khuyến nghị

```text
                    ┌──────────────────────┐
                    │  Original model.py   │
                    └──────────┬───────────┘
                               ↓
                    ┌──────────────────────┐
                    │ Clean model baseline │
                    └──────────┬───────────┘
                               │
                               │
         ┌─────────────────────┴────────────────────┐
         │                                          │
         ↓                                          ↓
Incremental Study                           Leave-One-Out Ablation

Baseline                                  V4 Full
 + Projector                              V4 - Projector
 + Adaptive Fusion                        V4 - Adaptive Fusion
 + Residual Alpha                         V4 - Alignment
 + PPR                                    V4 - Residual Alpha
 + Alignment                              V4 - PPR
         │
         ↓
       V4 Full
```

Hai hướng thí nghiệm bổ sung cho nhau:

- Leave-one-out trả lời:
  > Thành phần nào là cần thiết trong V4?

- Incremental study trả lời:
  > Thêm từng thành phần vào baseline mang lại cải thiện như thế nào?

---

# 21. Các module chính cuối cùng

AI khác chỉ cần nhớ 5 module chính sau:

```text
1. Medical / Domain Semantic Projector

2. Adaptive Text-Medical Fusion

3. Semantic Alignment Loss

4. LukePi-inspired Learnable Residual Alpha RGCN

5. CUDA PPR Top-50 Structural Branch
```

Optional module:

```text
6. Adaptive PPR-RGCN Fusion
```

Đây là các module cần được ablate để chứng minh đóng góp của V4.
