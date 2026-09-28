"""
ssl_eval.py - Độ đo SSL chung cho so sánh encoder (model.py vs model_base4.py).

Hai nhiệm vụ SSL (LukePi-style):
  1) Edge-type trên cạnh mask + negative  → multi-class (num_rels + 1)
  2) Degree-bin của node                 → multi-class (K bins)

Log mỗi epoch tách 2 task: loss, auc, aupr, f1, bacc (và bản node).
"""
from __future__ import annotations

import os
import random

import dgl
import numpy as np
import torch
import torch.nn.functional as F

import myutils
from checkpoint_utils import build_train_checkpoint


def seed_everything(seed):
    """Seed model initialization, dropout and graph samplers."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)


def _to_numpy_int(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.int64).ravel()
    return np.asarray(x, dtype=np.int64).ravel()


def _average_ranks(x):
    """Rank 1-based, average ties (scipy.stats.rankdata method='average')."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * ((i + 1) + j)
        i = j
    return ranks


def _binary_roc_auc(y_true, y_score):
    """Wilcoxon-Mann-Whitney AUC; None nếu thiếu pos hoặc neg."""
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    n_neg = y_true.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = _average_ranks(y_score)
    sum_pos = ranks[y_true].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _binary_average_precision(y_true, y_score):
    """sklearn-style average precision (AUPR không nội suy)."""
    y_true = np.asarray(y_true).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(y_score, kind="mergesort")[::-1]
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    prec = tp / np.arange(1, y_sorted.size + 1)
    rec = tp / float(n_pos)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def _macro_ovr_auc_aupr(y_true, scores, num_classes, need_aupr=True):
    """Macro one-vs-rest AUC và (tuỳ chọn) AUPR; bỏ lớp không có cả pos lẫn neg."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    scores = np.asarray(scores, dtype=np.float64)
    present = np.unique(y_true)
    aucs, auprs = [], []
    for c in present:
        if c < 0 or c >= num_classes:
            continue
        y_bin = (y_true == c).astype(np.int64)
        auc_c = _binary_roc_auc(y_bin, scores[:, c])
        if auc_c is not None:
            aucs.append(auc_c)
        if need_aupr:
            aupr_c = _binary_average_precision(y_bin, scores[:, c])
            if aupr_c is not None:
                auprs.append(aupr_c)
    return (
        float(np.mean(aucs)) if aucs else 0.0,
        float(np.mean(auprs)) if auprs else 0.0,
    )


def _macro_f1(preds, labels, num_classes):
    """Macro-F1; bỏ qua lớp không xuất hiện trong labels."""
    preds = np.asarray(preds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    f1s = []
    for c in range(num_classes):
        true_c = labels == c
        pred_c = preds == c
        if not true_c.any():
            continue
        tp = int((pred_c & true_c).sum())
        fp = int((pred_c & ~true_c).sum())
        fn = int((~pred_c & true_c).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def _balanced_accuracy(preds, labels):
    """BACC = trung bình recall theo lớp có trong labels (sklearn)."""
    preds = np.asarray(preds, dtype=np.int64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    recs = []
    for c in np.unique(labels):
        mask = labels == c
        recs.append(float((preds[mask] == c).mean()) if mask.any() else 0.0)
    return float(np.mean(recs)) if recs else 0.0


def compute_clf_metrics(labels, preds, logits, num_classes, extra=True):
    """
    AUC (+ AUPR/BACC nếu extra=True) và macro-F1 cho phân loại đa lớp.

    extra=False: chỉ auc + f1 (log train kiểu LukePi, rẻ hơn).
    """
    labels_np = _to_numpy_int(labels)
    preds_np = _to_numpy_int(preds)
    if torch.is_tensor(logits):
        scores = torch.softmax(logits.detach(), dim=1).cpu().numpy()
    else:
        logits_np = np.asarray(logits, dtype=np.float64)
        exp = np.exp(logits_np - logits_np.max(axis=1, keepdims=True))
        scores = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)

    auc, aupr = _macro_ovr_auc_aupr(
        labels_np, scores, num_classes, need_aupr=extra,
    )
    out = {
        "auc": auc,
        "f1": _macro_f1(preds_np, labels_np, num_classes),
    }
    if extra:
        out["aupr"] = aupr
        out["bacc"] = _balanced_accuracy(preds_np, labels_np)
    return out


def macro_f1(preds, labels, num_classes):
    """Giữ tên cũ cho chỗ gọi ngoài; ủy quyền sang _macro_f1."""
    return _macro_f1(_to_numpy_int(preds), _to_numpy_int(labels), num_classes)


def sample_eval_negatives(num_nodes, known_edges, num_neg, rng):
    """Sample cặp (src, dst) không nằm trong known_edges (giống LukePi neg)."""
    src_neg, dst_neg = [], []
    max_trials = max(num_neg * 20, num_neg + 100)
    trials = 0
    seen = set()
    while len(src_neg) < num_neg and trials < max_trials:
        trials += 1
        s = int(rng.randint(0, num_nodes))
        d = int(rng.randint(0, num_nodes))
        if s == d:
            continue
        key = (s, d)
        if key in known_edges or key in seen:
            continue
        seen.add(key)
        src_neg.append(s)
        dst_neg.append(d)
    return (
        np.asarray(src_neg, dtype=np.int64),
        np.asarray(dst_neg, dtype=np.int64),
    )


@torch.no_grad()
def evaluate_ssl_edge(
    embed,
    edge_head,
    triplets,
    num_rels,
    neg_rate,
    total_data,
    device,
    max_triples=5000,
    rng=None,
):
    """
    SSL edge eval trên held-out triples (valid/test).

    Positive: (s, r, o) với nhãn = r.
    Negative: cặp ngẫu nhiên không có trong total_data, nhãn = num_rels (fake).

    Returns:
        dict: loss, auc, aupr, f1, bacc, num_pos, num_neg
    """
    if rng is None:
        rng = np.random.RandomState(42)

    empty = {
        "loss": 0.0, "auc": 0.0, "aupr": 0.0, "f1": 0.0, "bacc": 0.0,
        "num_pos": 0, "num_neg": 0,
    }
    triplets = np.asarray(triplets)
    if triplets.size == 0:
        return empty

    if max_triples is not None and len(triplets) > max_triples:
        idx = rng.choice(len(triplets), size=int(max_triples), replace=False)
        triplets = triplets[idx]

    total_np = (
        total_data.detach().cpu().numpy()
        if torch.is_tensor(total_data)
        else np.asarray(total_data)
    )
    known = set(zip(total_np[:, 0].tolist(), total_np[:, 2].tolist()))
    known |= {(d, s) for s, d in list(known)}

    n_pos = len(triplets)
    n_neg = int(n_pos * neg_rate)
    num_nodes = embed.shape[0]
    neg_src, neg_dst = sample_eval_negatives(num_nodes, known, n_neg, rng)
    n_neg = len(neg_src)

    pos_src = torch.from_numpy(triplets[:, 0].astype(np.int64)).to(device)
    pos_rel = torch.from_numpy(triplets[:, 1].astype(np.int64)).to(device)
    pos_dst = torch.from_numpy(triplets[:, 2].astype(np.int64)).to(device)
    neg_src_t = torch.from_numpy(neg_src).to(device)
    neg_dst_t = torch.from_numpy(neg_dst).to(device)

    src = torch.cat([pos_src, neg_src_t], dim=0)
    dst = torch.cat([pos_dst, neg_dst_t], dim=0)
    labels = torch.cat(
        [
            pos_rel,
            torch.full((n_neg,), num_rels, dtype=torch.long, device=device),
        ],
        dim=0,
    )

    logits = edge_head(embed[src], embed[dst])
    loss = F.cross_entropy(logits, labels).item()
    preds = torch.argmax(logits, dim=1)
    clf = compute_clf_metrics(labels, preds, logits, num_rels + 1)

    return {
        "loss": loss,
        "auc": clf["auc"],
        "aupr": clf["aupr"],
        "f1": clf["f1"],
        "bacc": clf["bacc"],
        "num_pos": n_pos,
        "num_neg": n_neg,
    }


@torch.no_grad()
def evaluate_ssl_degree(embed, degree_head, degree_label_full):
    """Degree probe: dự đoán bucket bậc (nhãn từ train_deg) trên mọi node đã encode."""
    labels = degree_label_full
    if labels.dim() > 1:
        labels = labels.view(-1)
    logits = degree_head(embed)
    loss = F.cross_entropy(logits, labels).item()
    preds = torch.argmax(logits, dim=1)
    num_bins = logits.shape[1]
    clf = compute_clf_metrics(labels, preds, logits, num_bins)
    return {
        "loss": loss,
        "auc": clf["auc"],
        "aupr": clf["aupr"],
        "f1": clf["f1"],
        "bacc": clf["bacc"],
    }


def encode_full_train_graph(
    model,
    train_graph,
    train_rel,
    train_norm,
    num_nodes,
    device,
    **encode_kwargs,
):
    """
    Encode mọi node trên đồ thị train (valid và TEST SSL dùng cùng graph).
    encode_kwargs: tùy chọn cho model_base4 (ppr_graph, ppr_edge_weight, all_node_ids).
    """
    eval_graph = train_graph.to(device)
    eval_node_id = torch.arange(0, num_nodes, dtype=torch.long, device=device).view(-1, 1)
    eval_rel = torch.from_numpy(train_rel).to(device)
    eval_norm = myutils.node_norm_2_edge_norm(
        train_graph, torch.from_numpy(train_norm).view(-1, 1)
    ).to(device)
    # Degree supervision needs encoder gradients. Evaluation remains detached.
    with torch.set_grad_enabled(model.training and torch.is_grad_enabled()):
        return model(
            eval_graph,
            eval_node_id,
            eval_rel,
            eval_norm,
            **encode_kwargs,
        )


def build_degree_labels(train_deg, num_bins, strategy="log"):
    """
    Sinh nhãn bucket bậc từ bậc trên đồ thị train.

    - strategy="log": label = min(floor(log2(deg + 1)), num_bins - 1)
    - strategy="quantile": chia theo phân vị; fallback "log" nếu quá ít giá trị bậc.
    """
    deg = train_deg.detach().cpu().float().view(-1).numpy()

    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, num_bins + 1)[1:-1]
        edges = np.unique(np.quantile(deg, quantiles))
        if edges.shape[0] < 1:
            strategy = "log"
        else:
            labels = np.digitize(deg, edges, right=False)
            labels = np.clip(labels, 0, num_bins - 1)
            return torch.from_numpy(labels).long(), edges.tolist()

    labels = np.floor(np.log2(deg + 1.0)).astype(np.int64)
    labels = np.clip(labels, 0, num_bins - 1)
    bin_edges = [float(2 ** k - 1) for k in range(1, num_bins)]
    return torch.from_numpy(labels).long(), bin_edges


def build_ssl_checkpoint(
    model,
    edge_head,
    degree_head,
    iteration,
    args,
    text_embeddings,
    ontology_embeddings,
    num_rels,
    degree_bin_edges,
    model_module="model",
    best_valid_loss_edge=None,
    best_iter=None,
    loss_weighter=None,
):
    checkpoint = build_train_checkpoint(
        model, iteration, args, text_embeddings, ontology_embeddings,
        model_module=model_module,
    )
    checkpoint["ssl_heads"] = {
        "edge_head": edge_head.state_dict(),
        "degree_head": degree_head.state_dict(),
    }
    checkpoint["meta"]["ssl"] = {
        "task": "degree+edge",
        "method": "lukepi_edge_mask",
        "mask_rate": args.mask_rate,
        "num_edge_classes": num_rels + 1,
        "num_degree_bins": args.num_degree_bins,
        "degree_bin_strategy": args.degree_bin_strategy,
        "degree_bin_edges": degree_bin_edges,
        "seed": getattr(args, "seed", None),
        "degree_on_full": getattr(args, "degree_on_full", False),
    }
    if model_module == 'model_base4':
        checkpoint['meta']['ssl']['architecture_version'] = 'masked_global_ppr_mass_v2'
        checkpoint['meta']['ssl']['ppr_c'] = getattr(args, 'ppr_c', 0.85)
        checkpoint['meta']['ssl']['ppr_iterations'] = getattr(args, 'ppr_iter_num', 8)
        checkpoint['meta']['ssl']['ppr_num_layers'] = getattr(args, 'ppr_num_layers', 2)
        checkpoint['meta']['ssl']['degree_prior_correction'] = bool(
            getattr(args, 'degree_prior_correction', False)
            and (getattr(args, 'degree_on_full', False)
                 or getattr(args, 'edge_sampler', 'uniform') == 'uniform'))
    if loss_weighter is not None:
        checkpoint["ssl_heads"]["loss_weighter"] = loss_weighter.state_dict()
        w_edge, w_node = loss_weighter.weights()
        checkpoint["meta"]["ssl"]["loss_weighting"] = "learned_uncertainty"
        checkpoint["meta"]["ssl"]["lambda_edge"] = w_edge
        checkpoint["meta"]["ssl"]["lambda_node"] = w_node
    else:
        checkpoint["meta"]["ssl"]["loss_weighting"] = "static"
        checkpoint["meta"]["ssl"]["lambda_degree"] = float(
            getattr(args, "lambda_degree", 1.0)
        )
    if best_valid_loss_edge is not None:
        checkpoint["meta"]["best_valid_loss_edge"] = float(best_valid_loss_edge)
    if best_iter is not None:
        checkpoint["meta"]["best_iter"] = int(best_iter)
    return checkpoint


def final_ckpt_path(path):
    stem, ext = os.path.splitext(path)
    return f"{stem}_final{ext or '.pth'}"


class UncertaintyWeighter(torch.nn.Module):
    """Tự học trọng số 2 task theo homoscedastic uncertainty (Kendall et al. 2018).

    Mỗi task có 1 tham số log-variance s = log(sigma^2) học được. Tổng loss:

        L = exp(-s_edge) * L_edge + exp(-s_node) * L_node
            + 0.5 * (s_edge + s_node)

    Số hạng 0.5*s là regularizer: nếu không có nó, gradient sẽ đẩy mọi trọng số
    về 0 để làm loss nhỏ giả tạo. Trọng số hiệu dụng (lambda) của mỗi task
    = exp(-s) = 1 / sigma^2, nên task nào "nhiễu/khó" hơn tự nhận trọng số nhỏ
    hơn. lambda_degree = exp(-s_node).
    """

    def __init__(self, init_log_var_edge=0.0, init_log_var_node=0.0):
        super().__init__()
        self.log_var_edge = torch.nn.Parameter(
            torch.tensor(float(init_log_var_edge))
        )
        self.log_var_node = torch.nn.Parameter(
            torch.tensor(float(init_log_var_node))
        )

    def forward(self, loss_edge, loss_node):
        w_edge = torch.exp(-self.log_var_edge)
        w_node = torch.exp(-self.log_var_node)
        return (
            w_edge * loss_edge + 0.5 * self.log_var_edge
            + w_node * loss_node + 0.5 * self.log_var_node
        )

    def weights(self):
        """Trọng số hiệu dụng (lambda_edge, lambda_node) để log/lưu checkpoint."""
        with torch.no_grad():
            return (
                float(torch.exp(-self.log_var_edge)),
                float(torch.exp(-self.log_var_node)),
            )


SSL_LOG_KEYS = (
    "loss", "loss_edge", "loss_node",
    "auc", "aupr", "f1", "bacc",
    "auc_node", "aupr_node", "f1_node", "bacc_node",
    "w_edge", "w_node",
)


def empty_ssl_log():
    return {k: 0.0 for k in SSL_LOG_KEYS}


def add_ssl_log(acc, log):
    for k in SSL_LOG_KEYS:
        acc[k] += log[k]
    return acc


def average_ssl_log(acc, n):
    n = max(int(n), 1)
    return {k: acc[k] / n for k in SSL_LOG_KEYS}


def _loss_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


def batch_ssl_log(
    loss_total,
    loss_edge,
    loss_node,
    edge_logits,
    edge_labels,
    deg_logits,
    deg_labels,
    w_edge=1.0,
    w_node=1.0,
):
    """
    Số log cho 1 batch train, TÁCH RIÊNG 2 task.

    loss_total: objective thực được tối ưu (tổng có trọng số, có thể học được).
    loss_edge / loss_node: CE thô của từng task (chưa nhân trọng số).
    w_edge / w_node: trọng số hiệu dụng đang áp cho mỗi task.
    """
    edge_preds = torch.argmax(edge_logits.detach(), dim=1)
    deg_preds = torch.argmax(deg_logits.detach(), dim=1)
    edge_m = compute_clf_metrics(
        edge_labels, edge_preds, edge_logits, edge_logits.size(1), extra=True,
    )
    deg_m = compute_clf_metrics(
        deg_labels, deg_preds, deg_logits, deg_logits.size(1), extra=True,
    )
    return {
        "loss": _loss_to_float(loss_total),
        "loss_edge": _loss_to_float(loss_edge),
        "loss_node": _loss_to_float(loss_node),
        "auc": edge_m["auc"],
        "aupr": edge_m["aupr"],
        "f1": edge_m["f1"],
        "bacc": edge_m["bacc"],
        "auc_node": deg_m["auc"],
        "aupr_node": deg_m["aupr"],
        "f1_node": deg_m["f1"],
        "bacc_node": deg_m["bacc"],
        "w_edge": float(w_edge),
        "w_node": float(w_node),
    }


def eval_ssl_log(edge_metrics, deg_metrics, w_edge=1.0, w_node=1.0):
    """Gộp evaluate_ssl_edge / evaluate_ssl_degree, giữ tách 2 task."""
    le = float(edge_metrics["loss"])
    ln = float(deg_metrics["loss"])
    return {
        "loss": le + ln,
        "loss_edge": le,
        "loss_node": ln,
        "auc": float(edge_metrics["auc"]),
        "aupr": float(edge_metrics["aupr"]),
        "f1": float(edge_metrics["f1"]),
        "bacc": float(edge_metrics["bacc"]),
        "auc_node": float(deg_metrics["auc"]),
        "aupr_node": float(deg_metrics["aupr"]),
        "f1_node": float(deg_metrics["f1"]),
        "bacc_node": float(deg_metrics["bacc"]),
        "w_edge": float(w_edge),
        "w_node": float(w_node),
    }


def format_ssl_log_edge(mode, step, metrics):
    """Dòng log riêng cho task EDGE (masked-edge classification)."""
    return (
        "{} | step {} | [EDGE]   loss {:.5f} auc {:.4f} aupr {:.4f} "
        "f1 {:.4f} bacc {:.4f} w {:.3f}".format(
            mode, step,
            metrics["loss_edge"], metrics["auc"], metrics["aupr"],
            metrics["f1"], metrics["bacc"],
            metrics.get("w_edge", 1.0),
        )
    )


def format_ssl_log_node(mode, step, metrics):
    """Dòng log riêng cho task DEGREE (node-degree classification)."""
    return (
        "{} | step {} | [DEGREE] loss {:.5f} auc {:.4f} aupr {:.4f} "
        "f1 {:.4f} bacc {:.4f} w {:.3f}".format(
            mode, step,
            metrics["loss_node"], metrics["auc_node"], metrics["aupr_node"],
            metrics["f1_node"], metrics["bacc_node"],
            metrics.get("w_node", 1.0),
        )
    )


def format_ssl_log(mode, step, metrics):
    """Ba dòng: tổng loss + 2 dòng tách riêng cho EDGE và DEGREE."""
    total = "{} | step {} | loss_total {:.5f}".format(
        mode, step, metrics["loss"]
    )
    return "\n".join([
        total,
        format_ssl_log_edge(mode, step, metrics),
        format_ssl_log_node(mode, step, metrics),
    ])
