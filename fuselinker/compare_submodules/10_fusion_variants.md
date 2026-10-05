# 10 Phiên Bản Fusion cho Text Embedding và Domain Embedding

Mục tiêu: kết hợp `text_x` và `domain_x` theo hướng đơn giản, adaptive theo từng node, hạn chế làm biến dạng embedding gốc và tránh kiến trúc fusion quá phức tạp.

---

## Fusion V1 — Simple Feature Gate

Mỗi node và mỗi dimension có một gate riêng.

```python
class FusionV1(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

**Ý tưởng:**

```text
text_x ----\
            -> concat -> Linear -> sigmoid -> gate
domain_x --/

fused = gate * text_x + (1 - gate) * domain_x
```

- Adaptive theo từng node.
- Adaptive theo từng chiều embedding.
- Kiến trúc đơn giản, dễ giải thích.

---

## Fusion V2 — Zero-init Feature Gate

Giống V1 nhưng khởi tạo gate tại trạng thái 50/50.

```python
class FusionV2(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Ban đầu:

```text
Linear output = 0
sigmoid(0) = 0.5

fused = 0.5 * text_x + 0.5 * domain_x
```

Sau đó model tự học cách điều chỉnh gate.

- Khởi đầu ổn định.
- Không thiên lệch ngẫu nhiên về một embedding ngay từ đầu.
- Rất phù hợp làm baseline adaptive fusion.

---

## Fusion V3 — Node-wise Scalar Gate

Mỗi node chỉ có một scalar gate thay vì một gate cho từng dimension.

```python
class FusionV3(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text_x, domain_x):
        gate = torch.sigmoid(
            self.gate(torch.cat([text_x, domain_x], dim=-1))
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Ví dụ:

```text
Node A: 0.8 * text + 0.2 * domain
Node B: 0.3 * text + 0.7 * domain
Node C: 0.55 * text + 0.45 * domain
```

- Adaptive theo node.
- Không adaptive theo từng dimension.
- Rất nhẹ và dễ giải thích.

---

## Fusion V4 — Difference-aware Gate

Gate được tính dựa trên độ khác biệt giữa hai embedding.

```python
class FusionV4(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_x, domain_x):
        diff = torch.abs(text_x - domain_x)

        gate = torch.sigmoid(
            self.gate(diff)
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Ý tưởng:

```text
diff = |text_x - domain_x|
```

- Gate tập trung vào mức độ disagreement giữa hai nguồn.
- Input gate chỉ có kích thước `H`.
- Rất nhẹ.

---

## Fusion V5 — Concat + Difference Gate

Sử dụng cả embedding gốc và độ khác biệt giữa chúng.

```python
class FusionV5(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x):
        diff = torch.abs(text_x - domain_x)

        features = torch.cat(
            [text_x, domain_x, diff],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(features)
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Input:

```text
[text_x, domain_x, |text_x - domain_x|]
```

- Giàu thông tin hơn V1.
- Vẫn chỉ sử dụng một lớp Linear.
- Không có MLP residual, dropout hoặc LayerNorm.

---

## Fusion V6 — Similarity-aware Gate

Dùng tương tác từng chiều giữa hai embedding.

```python
class FusionV6(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(hidden_dim * 3, hidden_dim)

    def forward(self, text_x, domain_x):
        interaction = text_x * domain_x

        features = torch.cat(
            [text_x, domain_x, interaction],
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(features)
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Input:

```text
[text_x, domain_x, text_x * domain_x]
```

- `text_x * domain_x` biểu diễn mức độ interaction/agreement theo từng dimension.
- Khác V5 ở chỗ tập trung vào interaction thay vì difference.

---

## Fusion V7 — Low-rank Adaptive Gate

Thêm phi tuyến nhưng sử dụng bottleneck nhỏ để hạn chế số lượng tham số.

```python
class FusionV7(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        bottleneck = max(hidden_dim // 4, 16)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, text_x, domain_x):
        features = torch.cat(
            [text_x, domain_x],
            dim=-1
        )

        gate = self.gate(features)

        return gate * text_x + (1.0 - gate) * domain_x
```

Kiến trúc:

```text
2H -> H/4 -> H
```

- Có nonlinear modeling.
- Nhẹ hơn MLP fusion lớn.
- Không residual.
- Không LayerNorm.
- Không dropout.

---

## Fusion V8 — Global Prior + Node-specific Gate

Kết hợp một prior toàn cục với adjustment riêng cho từng node.

```python
class FusionV8(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.base_gate = nn.Parameter(
            torch.zeros(hidden_dim)
        )

        self.node_gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        nn.init.zeros_(self.node_gate.weight)
        nn.init.zeros_(self.node_gate.bias)

    def forward(self, text_x, domain_x):
        node_adjustment = self.node_gate(
            torch.cat([text_x, domain_x], dim=-1)
        )

        gate = torch.sigmoid(
            self.base_gate + node_adjustment
        )

        return gate * text_x + (1.0 - gate) * domain_x
```

Ý tưởng:

```text
global prior
    +
node-specific adjustment
    =
adaptive gate
```

- `base_gate`: xu hướng fusion chung của toàn dataset.
- `node_adjustment`: điều chỉnh cho từng node.
- Cân bằng giữa global knowledge và node-specific information.

---

## Fusion V9 — Two-score Softmax Fusion

Hai embedding cạnh tranh trực tiếp thông qua hai score riêng biệt.

```python
class FusionV9(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.text_score = nn.Linear(
            hidden_dim,
            hidden_dim
        )

        self.domain_score = nn.Linear(
            hidden_dim,
            hidden_dim
        )

    def forward(self, text_x, domain_x):
        text_logit = self.text_score(text_x)
        domain_logit = self.domain_score(domain_x)

        scores = torch.stack(
            [text_logit, domain_logit],
            dim=-1
        )

        weights = torch.softmax(
            scores,
            dim=-1
        )

        text_weight = weights[..., 0]
        domain_weight = weights[..., 1]

        return (
            text_weight * text_x
            + domain_weight * domain_x
        )
```

Tại mỗi dimension:

```text
[text_score, domain_score]
          |
       softmax
          |
[w_text, w_domain]
```

Luôn đảm bảo:

```text
w_text + w_domain = 1
```

- Cơ chế cạnh tranh rõ ràng giữa hai nguồn.
- Dễ diễn giải.

---

## Fusion V10 — Normalized Decision Gate

Normalize embedding chỉ để tính gate, nhưng vẫn fusion embedding gốc.

```python
class FusionV10(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.gate = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

    def forward(self, text_x, domain_x):
        text_norm = F.normalize(
            text_x,
            p=2,
            dim=-1
        )

        domain_norm = F.normalize(
            domain_x,
            p=2,
            dim=-1
        )

        gate = torch.sigmoid(
            self.gate(
                torch.cat(
                    [text_norm, domain_norm],
                    dim=-1
                )
            )
        )

        return (
            gate * text_x
            + (1.0 - gate) * domain_x
        )
```

Ý tưởng:

```text
normalized embeddings
        |
        v
    tính gate

original embeddings
        |
        v
      fusion
```

- Giảm ảnh hưởng do scale/norm giữa hai embedding khác nhau.
- Vẫn bảo toàn magnitude của embedding gốc khi fusion.

---

# Bảng so sánh nhanh

| Version | Cơ chế | Adaptive theo node | Adaptive theo dimension | Complexity |
|---|---|---:|---:|---:|
| V1 | Concatenate gate | Có | Có | Thấp |
| V2 | Zero-init concat gate | Có | Có | Thấp |
| V3 | Scalar node gate | Có | Không | Rất thấp |
| V4 | Difference gate | Có | Có | Rất thấp |
| V5 | Concat + difference | Có | Có | Thấp |
| V6 | Concat + interaction | Có | Có | Thấp |
| V7 | Low-rank nonlinear gate | Có | Có | Thấp-vừa |
| V8 | Global prior + node gate | Có | Có | Thấp |
| V9 | Two-score Softmax | Có | Có | Thấp |
| V10 | Normalized decision gate | Có | Có | Thấp |

---

# Các phiên bản nên ưu tiên chạy trước

Nếu cần rút gọn số lượng thí nghiệm ban đầu, nên ưu tiên:

```text
V2
V5
V8
V9
V10
```

Trong đó:

- **V2**: baseline adaptive fusion đơn giản và ổn định.
- **V5**: bổ sung thông tin disagreement nhưng vẫn nhẹ.
- **V8**: kết hợp global prior và node-specific gate.
- **V9**: hai embedding cạnh tranh trực tiếp bằng softmax.
- **V10**: hữu ích khi scale/norm của hai embedding khác nhau.

Nếu cần một kiến trúc vừa mạnh vừa dễ giải thích trong paper, nhóm đáng tập trung nhất là:

```text
V2
V5
V8
V9
```
