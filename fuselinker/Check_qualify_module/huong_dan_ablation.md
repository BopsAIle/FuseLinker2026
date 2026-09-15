# Hướng dẫn chạy ablation — bật/tắt từng module

Áp dụng cho `task_node_mask/model_base4.py` (bản viết lại), chạy qua
`task_node_mask/train_upgrade.py`.

> Bản `Check_qualify_module/model_base4.py` là ảnh chụp đông cứng của code gốc,
> **không có công tắc** và không script nào gọi nó. Đừng nhầm hai file.

Mọi lệnh chạy từ `D:\FuseLinker\FuseLinker\fuselinker`.

---

## 1. Chạy full (mọi module bật)

```powershell
cd D:\FuseLinker\FuseLinker\fuselinker

..\.venv\Scripts\python.exe task_node_mask\train_upgrade.py `
  --data suppkg `
  --text_embedding_file bert_pretrained_embeddings_768.npy `
  --knowledge_embedding_file poincare_embeddings.npy `
  --num_hidden_layers 2 --iterations 40000 --evaluate_every 1000 `
  --negative_sample 20 --neg_sample_size_eval 100 --w 0.75 `
  --seed 1 `
  --model_state_file checkpoints/suppkg/bert/abl_full_s1/model_state.pth
```

Dòng đầu của log in ra toàn bộ cấu hình đang chạy — đối chiếu dòng đó để chắc
chắn cờ đã vào đúng:

```
Encoder config: use_text=True | use_domain=True | text_encoder=mlp | ...
```

---

## 2. Bảy module tắt được

Mỗi dòng = thêm đúng một cờ vào lệnh full ở trên, rồi đổi tên thư mục checkpoint.

| # | Module | Cờ để TẮT | Mặc định | Tắt đi thì mất gì |
|---|--------|-----------|----------|-------------------|
| 1 | Nhánh phụ PPR (toàn bộ) | `--use_ppr false` | `true` | Quay về đúng backbone của base |
| 2 | Text embedding | `--use_text false` | `true` | Chỉ còn Poincaré làm feature node |
| 3 | Domain embedding (Poincaré) | `--use_domain false` | `true` | Chỉ còn text làm feature node |
| 4 | MLP encoder cho text | `--text_encoder linear` | `mlp` | Còn 1 Linear 768→200 thuần |
| 5 | LayerNorm trong text encoder | `--text_norm none` | `layer` | Không chuẩn hoá trước khi fuse |
| 6 | LayerNorm trong PPR conv | `--ppr_layer_norm false` | `true` | Nhánh PPR hết bị ghim scale |
| 7 | Regularization L2 | `--reg_on none` | `rgcn` | Chỉ còn BCE loss thuần |

Không tắt được đồng thời #2 và #3 — model sẽ báo lỗi ngay, vì không còn feature
đầu vào nào.

### Ba module có biến thể thay vì tắt hẳn

| Module | Cờ | Các lựa chọn |
|--------|-----|--------------|
| Cách gộp nhánh PPR | `--ppr_fusion` | `residual` (mặc định) · `scalar` · `gate` · `replace` |
| Cách aggregate PPR | `--ppr_agg` | `mean` (mặc định) · `sum` |
| Regularizer của R-GCN | `--rgcn_regularizer` | `bdd` (mặc định) · `basis` · `none` |

`--ppr_fusion scalar` đáng chạy nhất trong nhóm này: nó thay gate 200 chiều
bằng **một** scalar `α` init 0, nên sau khi train chỉ cần đọc `α` là biết model
có dùng PPR hay không, thay vì phải suy đoán qua metric.

`--ppr_fusion replace` bỏ hẳn backbone, chỉ dùng nhánh PPR — trả lời câu hỏi
"PPR một mình mạnh cỡ nào".

### Số lớp

`--ppr_num_layers 1` (mặc định 2) · `--num_hidden_layers 1` (R-GCN)

---

## 3. Sáu module mặc định TẮT, bật lên để thử

| Module | Cờ để BẬT | Ý tưởng lấy từ |
|--------|-----------|----------------|
| Gated fusion text/domain | `--input_fusion gate` | `model_base2.py` |
| LayerNorm sau khi fuse | `--input_norm true` | `model_base2.py` |
| Residual node-id embedding | `--node_id_embedding true` | `model_base2.py` |
| Residual block quanh R-GCN | `--rgcn_block residual` | `model_base2.py` |
| Autoencoder text (có recon loss) | `--text_encoder ae --text_recon_weight 0.1` | ý đồ gốc của `model.py` |
| LayerNorm sau PPR fusion | `--ppr_fusion_norm true` | mới |

---

## 4. Tái hiện đúng bản cũ (để đối chứng)

Bốn cờ này bật lại đồng thời bốn lỗi đã nêu trong `danh_gia_module.md`:

```powershell
--text_encoder ae --text_norm batch --ppr_agg sum --ppr_fanout 50 --reg_on fused
```

Chạy riêng từng cờ để tách xem lỗi nào đóng góp bao nhiêu vào chênh lệch:

| Cờ | Ứng với lỗi |
|----|-------------|
| `--reg_on fused` | 1 — regularization phạt phần PPR cộng thêm |
| `--text_encoder ae` | 2 — decoder chết |
| `--text_norm batch` | 3 — BatchNorm lệch train/eval |
| `--ppr_agg sum --ppr_fanout 50` | 4 — lệch scale train/eval |

---

## 5. Script chạy cả lưới

Lưu thành `run_ablation.ps1` trong `fuselinker/`:

```powershell
$base = @(
  "--data","suppkg",
  "--text_embedding_file","bert_pretrained_embeddings_768.npy",
  "--knowledge_embedding_file","poincare_embeddings.npy",
  "--num_hidden_layers","2","--iterations","40000","--evaluate_every","1000",
  "--negative_sample","20","--neg_sample_size_eval","100","--w","0.75"
)

$runs = @{
  "full"          = @()
  "no_ppr"        = @("--use_ppr","false")
  "no_text"       = @("--use_text","false")
  "no_domain"     = @("--use_domain","false")
  "text_linear"   = @("--text_encoder","linear")
  "no_text_norm"  = @("--text_norm","none")
  "no_ppr_norm"   = @("--ppr_layer_norm","false")
  "no_reg"        = @("--reg_on","none")
  "fusion_scalar" = @("--ppr_fusion","scalar")
  "legacy"        = @("--text_encoder","ae","--text_norm","batch",
                      "--ppr_agg","sum","--ppr_fanout","50","--reg_on","fused")
}

foreach ($seed in 1,2,3) {
  foreach ($name in $runs.Keys) {
    $ckpt = "checkpoints/suppkg/bert/abl_${name}_s${seed}/model_state.pth"
    $log  = "logs/abl_${name}_s${seed}.log"
    New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null
    Write-Host "=== $name seed=$seed ==="
    & ..\.venv\Scripts\python.exe task_node_mask\train_upgrade.py `
        @base @($runs[$name]) --seed $seed --model_state_file $ckpt *>&1 |
      Tee-Object -FilePath $log
  }
}
```

Gom kết quả:

```powershell
Select-String -Path logs\*.log -Pattern "^(MRR|AUROC|MR):" |
  ForEach-Object { "{0}`t{1}" -f $_.Filename, $_.Line }
```

---

## 6. Ba điều dễ làm hỏng thí nghiệm

**Một seed không đủ.** Mỗi cấu hình phải chạy ≥ 3 seed rồi so trung bình ±
độ lệch. Chênh lệch AUROC giữa hai run cùng cấu hình khác seed có thể lớn hơn
chênh lệch giữa hai cấu hình khác nhau — khi đó mọi kết luận "module này có
ích" đều là đọc nhiễu.

**Đổi checkpoint mỗi lần chạy.** Nếu để nguyên `--model_state_file`, run sau đè
run trước và bạn mất kết quả cũ. Các lệnh trên đã đặt tên theo
`abl_<tên>_s<seed>`.

**Theo dõi `ppr_gate` trong log.** Nó được in cùng loss mỗi
`--evaluate_every` iteration. Nếu sau 40k iteration vẫn đứng ở 0.0474 (đúng giá
trị khởi tạo `sigmoid(-3)`) thì nhánh PPR không học được gì — lúc đó kết luận
là bỏ nhánh phụ, bất kể metric có nhỉnh hơn base một chút hay không.

---

## 7. Danh sách đầy đủ 22 trường cấu hình

```
use_text=True              use_domain=True           text_encoder=mlp
text_norm=layer            text_recon_weight=0.0     input_fusion=fixed
input_norm=False           node_id_embedding=False   rgcn_block=plain
rgcn_regularizer=bdd       use_ppr=True              ppr_num_layers=2
ppr_fanout=-1              ppr_agg=mean              ppr_layer_norm=True
ppr_dropout=None           ppr_fusion=residual       ppr_fusion_init_bias=-3.0
ppr_fusion_norm=False      ppr_eval_device=cpu       reg_on=rgcn
relation_minmax=True
```

Xem `--help` để có danh sách sinh trực tiếp từ code:

```powershell
..\.venv\Scripts\python.exe task_node_mask\train_upgrade.py --help
```
