"""
Tính chỉ số AUROC (Area Under ROC curve) cho bài toán dự đoán liên kết.

================================================================================
CÁC BƯỚC TÍNH AUROC (theo phương pháp phân loại nhị phân 1:1)
================================================================================

Bước 1 – Chuẩn bị tập kiểm tra tích cực (positive test triples)
    - Lấy tập T gồm các bộ ba thực sự (s, r, o) trong tập test.
    - Mỗi bộ ba tích cực được gán nhãn y = 1.

Bước 2 – Sinh bộ ba tiêu cực bằng random corrupting (tỷ lệ 1:1)
    - Với mỗi bộ ba tích cực (s, r, o), tạo đúng 1 bộ ba tiêu cực:
        + Ngẫu nhiên chọn thay head (s) HOẶC tail (o).
        + Thay thế bằng một entity ngẫu nhiên khác trong biểu đồ.
        + Lặp lại cho đến khi bộ ba mới KHÔNG tồn tại trong toàn bộ KG
          (train + valid + test) — tránh corrupt ra triple thật.
    - Mỗi bộ ba tiêu cực được gán nhãn y = 0.
    - Kết quả: |T_pos| = |T_neg|, tỷ lệ 1:1.

Bước 3 – Tính điểm (score) cho từng bộ ba bằng mô hình
    - Dùng hàm chấm DistMult (giống pipeline MR/MRR/Hits@k):
        score_raw(h, r, t) = sum( emb[h] * w[r] * emb[t] )
        score      = sigmoid(score_raw)   # xác suất dự đoán ∈ (0, 1)

Bước 4 – Xây dựng đường cong ROC
    - Sắp xếp tất cả mẫu theo score giảm dần (ưu tiên điểm cao).
    - Duyệt từng ngưỡng:
        TPR (True Positive Rate)  = TP / (TP + FN) = TP / n_pos
        FPR (False Positive Rate) = FP / (FP + TN) = FP / n_neg
    - Vẽ đường cong (FPR, TPR) khi thay đổi ngưỡng.

Bước 5 – Tính AUROC
    - AUROC = diện tích dưới đường cong ROC (tích phân theo FPR).
    - Giá trị ∈ [0, 1]; càng gần 1 nghĩa là mô hình phân biệt tốt hơn
      giữa bộ ba thực sự và bộ ba tiêu cực.

Công thức tóm tắt:
    AUROC = ∫₀¹ TPR(FPR) d(FPR)

Sử dụng:
    from calc_auroc import calc_auroc

    auroc, fpr, tpr, scores, labels = calc_auroc(
        emb, relation_weights, test_triplets, total_data
    )
    print(f"AUROC: {auroc:.4f}")
"""

import random
from typing import Optional, Set, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Bước 2: Random corrupting – sinh 1 negative cho mỗi positive (tỷ lệ 1:1)
# ---------------------------------------------------------------------------

def _random_corrupt_one(
    s: int,
    r: int,
    o: int,
    num_nodes: int,
    triplets_to_filter: Set[Tuple[int, int, int]],
    rng: random.Random,
    max_tries: int = 100,
) -> Tuple[int, int, int]:
    """
    Thay ngẫu nhiên head hoặc tail của (s, r, o) để tạo 1 bộ ba tiêu cực.
    Đảm bảo: entity thay thế khác entity gốc và bộ ba mới không có trong KG.
    """
    for _ in range(max_tries):
        corrupt_head = rng.random() > 0.5
        e = rng.randrange(num_nodes)
        if corrupt_head:
            if e == s:
                continue
            neg = (e, r, o)
        else:
            if e == o:
                continue
            neg = (s, r, e)
        if neg not in triplets_to_filter:
            return neg

    # Fallback an toàn: duyệt toàn bộ ứng viên hợp lệ (tránh corrupt ra triple thật)
    candidates = []
    for e in range(num_nodes):
        if e != s and (e, r, o) not in triplets_to_filter:
            candidates.append((e, r, o))
        if e != o and (s, r, e) not in triplets_to_filter:
            candidates.append((s, r, e))
    if not candidates:
        raise ValueError(
            f"Không thể corrupt triple ({s}, {r}, {o}): không còn entity hợp lệ."
        )
    return rng.choice(candidates)


def generate_auroc_test_pairs(
    test_triplets: np.ndarray,
    num_nodes: int,
    total_data: np.ndarray,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bước 1 + 2: Tạo cặp (positive, negative) 1:1 cho đánh giá AUROC.

    Returns
    -------
    triplets : (2N, 3) – N positive rồi N negative xen kẽ [pos, neg, pos, neg, ...]
    labels   : (2N,)   – nhãn 1 cho positive, 0 cho negative
    """
    rng = random.Random(seed)
    triplets_to_filter = {tuple(t) for t in total_data.tolist()}

    pos_list = []
    neg_list = []
    for triplet in test_triplets:
        s, r, o = int(triplet[0]), int(triplet[1]), int(triplet[2])
        pos_list.append([s, r, o])
        neg = _random_corrupt_one(s, r, o, num_nodes, triplets_to_filter, rng)
        neg_list.append(list(neg))

    pos_arr = np.asarray(pos_list, dtype=np.int64)
    neg_arr = np.asarray(neg_list, dtype=np.int64)

    # Xen kẽ pos/neg để dễ theo dõi; thứ tự không ảnh hưởng AUROC
    n = len(pos_arr)
    triplets = np.empty((2 * n, 3), dtype=np.int64)
    labels = np.zeros(2 * n, dtype=np.float32)
    triplets[0::2] = pos_arr
    triplets[1::2] = neg_arr
    labels[0::2] = 1.0

    return triplets, labels


# ---------------------------------------------------------------------------
# Bước 3: Tính điểm DistMult + sigmoid (tương thích myutils / LinkPredict)
# ---------------------------------------------------------------------------

def distmult_scores(
    emb: torch.Tensor,
    w: torch.Tensor,
    triplets: torch.Tensor,
    batch_size: int = 4096,
    apply_sigmoid: bool = True,
) -> torch.Tensor:
    """
    Tính điểm DistMult cho từng bộ ba.
    Mặc định dùng sigmoid(raw) làm xác suất dự đoán ∈ (0, 1) cho ROC.
    (Thứ hạng không đổi nếu bỏ sigmoid vì hàm đơn điệu.)
    """
    scores = []
    n = triplets.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = triplets[start : start + batch_size]
            h = emb[batch[:, 0]]
            r = w[batch[:, 1]]
            t = emb[batch[:, 2]]
            raw = torch.sum(h * r * t, dim=1)
            scores.append(torch.sigmoid(raw) if apply_sigmoid else raw)
    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Bước 4 + 5: Đường cong ROC và AUROC
# ---------------------------------------------------------------------------

def _auroc_mann_whitney(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    AUROC chuẩn theo Mann-Whitney U (Wilcoxon rank-sum).
    Xử lý đúng trường hợp điểm bằng nhau (ties).
    """
    from scipy.stats import rankdata

    labels = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cần cả mẫu positive và negative để tính AUROC.")

    ranks = rankdata(scores, method="average")
    pos_rank_sum = ranks[labels == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_roc_curve(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Bước 4–5: Tính (FPR, TPR) cho vẽ đồ thị và AUROC theo Mann-Whitney U.

    Parameters
    ----------
    labels : (N,) nhãn 0/1
    scores : (N,) điểm dự đoán, càng cao càng nghiêng về positive

    Returns
    -------
    fpr, tpr, auroc
    """
    labels = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()

    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cần cả mẫu positive và negative để tính AUROC.")

    auroc = _auroc_mann_whitney(labels, scores)

    # Đường cong ROC (bước nhảy) để vẽ biểu đồ
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = fp = 0

    for label in sorted_labels:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    fpr = np.asarray(fpr_list)
    tpr = np.asarray(tpr_list)
    return fpr, tpr, auroc


def calc_auroc(
    emb: torch.Tensor,
    w: torch.Tensor,
    test_triplets: torch.Tensor,
    total_data: torch.Tensor,
    batch_size: int = 4096,
    seed: Optional[int] = None,
    return_pairs: bool = False,
):
    """
    Hàm chính: tính AUROC trên tập test (giao diện tương tự calc_mrr).

    Parameters
    ----------
    emb            : embedding node từ mô hình, shape (num_nodes, dim)
    w              : embedding quan hệ, shape (num_rels, dim)
    test_triplets  : (N, 3) bộ ba test
    total_data     : toàn bộ triple trong KG (để lọc corrupt hợp lệ)
    batch_size     : batch khi tính score
    seed           : seed cho random corrupting (tái lập kết quả)
    return_pairs   : nếu True, trả thêm triplets và labels đã dùng

    Returns
    -------
    auroc          : float
    fpr, tpr       : mảng cho vẽ đồ thị ROC
    scores         : điểm dự đoán của từng mẫu
    labels         : nhãn 0/1 tương ứng
    (triplets)     : chỉ khi return_pairs=True
    """
    num_nodes = emb.shape[0]

    if isinstance(test_triplets, torch.Tensor):
        test_np = test_triplets.detach().cpu().numpy()
    else:
        test_np = np.asarray(test_triplets)

    if isinstance(total_data, torch.Tensor):
        total_np = total_data.detach().cpu().numpy()
    else:
        total_np = np.asarray(total_data)

    # Bước 1–2
    triplets_np, labels_np = generate_auroc_test_pairs(
        test_np, num_nodes, total_np, seed=seed
    )

    # Kiểm tra: negative không được trùng triple thật trong KG
    triplets_to_filter = {tuple(t) for t in total_np.tolist()}
    for i in range(1, len(triplets_np), 2):
        if tuple(triplets_np[i]) in triplets_to_filter:
            raise ValueError(f"Negative sample không hợp lệ (tồn tại trong KG): {triplets_np[i]}")

    triplets_t = torch.as_tensor(triplets_np, device=emb.device, dtype=torch.long)
    labels_t = torch.as_tensor(labels_np, device=emb.device, dtype=torch.float32)

    # Bước 3
    scores_t = distmult_scores(emb, w, triplets_t, batch_size=batch_size)
    scores_np = scores_t.detach().cpu().numpy()

    # Bước 4–5
    fpr, tpr, auroc = compute_roc_curve(labels_np, scores_np)

    if return_pairs:
        return auroc, fpr, tpr, scores_np, labels_np, triplets_np
    return auroc, fpr, tpr, scores_np, labels_np


def roc_curve_path_from_checkpoint(model_state_file: str, filename: str = "roc_curve.png") -> str:
    """Đường dẫn ROC cùng thư mục với checkpoint, ví dụ .../pubmedbert/roc_curve.png."""
    import os
    return os.path.join(os.path.dirname(model_state_file) or ".", filename)


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auroc: float,
    save_path: Optional[str] = None,
    title: str = "ROC Curve (Link Prediction)",
    show: Optional[bool] = None,
):
    """
    Vẽ đồ thị ROC (cần matplotlib). Tùy chọn lưu ra file.
    Mặc định: có save_path thì chỉ lưu file, không mở cửa sổ.
    """
    import os
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("Cần cài matplotlib để vẽ ROC: pip install matplotlib") from e

    if show is None:
        show = save_path is None

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Ngẫu nhiên (0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Ví dụ chạy độc lập (sau khi đã có embedding từ train script DistMult)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("calc_auroc.py - xem docstring trong file de biet cac buoc tinh AUROC.")
    print("Demo toi thieu (du lieu gia):\n")

    num_nodes, num_rels, dim = 20, 5, 8
    emb = torch.randn(num_nodes, dim)
    w = torch.randn(num_rels, dim)

    test_triplets = torch.tensor(
        [[0, 0, 1], [2, 1, 3], [4, 2, 5], [1, 0, 6], [3, 3, 7]],
        dtype=torch.long,
    )
    total_data = torch.tensor(
        [
            [0, 0, 1], [2, 1, 3], [4, 2, 5], [1, 0, 6], [3, 3, 7],
            [0, 1, 2], [5, 2, 8], [7, 0, 9],
        ],
        dtype=torch.long,
    )

    auroc, fpr, tpr, scores, labels = calc_auroc(
        emb, w, test_triplets, total_data, seed=42
    )
    print(f"AUROC (demo): {auroc:.4f}")
    print(f"Samples: {len(labels)} (positive={int(labels.sum())}, negative={int((1-labels).sum())})")
