import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from itertools import combinations
from few_shot_clustering.eval_utils import cluster_acc


def calculate_clustering_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, n_clusters: int
) -> dict:
    """Compute clustering metrics using Hungarian-matched accuracy, NMI, ARI, and pairwise F1."""
    if len(y_pred) != len(y_true):
        print("Error: Predicted assignments and true labels must have the same length.")
        return _empty_metrics()

    valid = y_pred != -1
    y_true_v, y_pred_v = y_true[valid], y_pred[valid]

    if len(y_pred_v) == 0:
        print("Warning: No valid predicted assignments found.")
        return _empty_metrics()

    true_uniq = np.unique(y_true_v)
    pred_uniq = np.unique(y_pred_v)
    if len(true_uniq) == 0 or len(pred_uniq) == 0:
        return _empty_metrics()

    true_to_idx = {lbl: i for i, lbl in enumerate(true_uniq)}
    pred_to_idx = {a: i for i, a in enumerate(pred_uniq)}

    w = np.zeros((len(pred_uniq), len(true_uniq)), dtype=np.int64)
    for t, p in zip(y_true_v, y_pred_v):
        w[pred_to_idx[p], true_to_idx[t]] += 1

    row_ind, col_ind = hungarian(w.max() - w)
    idx_to_pred = {i: a for a, i in pred_to_idx.items()}
    idx_to_true = {i: lbl for lbl, i in true_to_idx.items()}
    pred_assign_map = {idx_to_pred[r]: idx_to_true[c] for r, c in zip(row_ind, col_ind)}

    mapped = np.full(y_pred.shape, -2, dtype=np.int64)
    for i in range(len(y_pred)):
        if y_pred[i] != -1:
            mapped[i] = pred_assign_map.get(y_pred[i], -2)

    accuracy = None
    if cluster_acc is not None:
        try:
            accuracy = cluster_acc(y_true, y_pred)
        except Exception as e:
            print(f"Warning: Error calculating accuracy: {e}")

    mapped_v = mapped[valid]
    metrics = _empty_metrics()

    try:
        metrics["NMI"] = normalized_mutual_info_score(y_true_v, y_pred_v)
        metrics["ARI"] = adjusted_rand_score(y_true_v, y_pred_v)
    except Exception as e:
        print(f"Error calculating NMI/ARI: {e}")

    try:
        (
            metrics["Pairwise_Precision"],
            metrics["Pairwise_Recall"],
            metrics["Pairwise_F1"],
        ) = _pairwise_metrics(y_true_v, y_pred_v)
    except Exception as e:
        print(f"Error calculating pairwise metrics: {e}")

    if len(mapped_v) > 0:
        try:
            metrics.update(
                {
                    "Accuracy": accuracy,
                    "Precision": precision_score(
                        y_true_v, mapped_v, average="macro", zero_division=0
                    ),
                    "Recall": recall_score(
                        y_true_v, mapped_v, average="macro", zero_division=0
                    ),
                    "Macro_F1": f1_score(
                        y_true_v, mapped_v, average="macro", zero_division=0
                    ),
                    "Micro_F1": f1_score(
                        y_true_v, mapped_v, average="micro", zero_division=0
                    ),
                }
            )
        except Exception as e:
            print(f"Error calculating metrics: {e}")

    return metrics


def _empty_metrics() -> dict:
    return {
        k: None
        for k in [
            "Accuracy",
            "Precision",
            "Recall",
            "Macro_F1",
            "Micro_F1",
            "NMI",
            "ARI",
            "Pairwise_Precision",
            "Pairwise_Recall",
            "Pairwise_F1",
        ]
    }


def _pairwise_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    n = len(y_true)
    true_pairs, pred_pairs = set(), set()
    for i, j in combinations(range(n), 2):
        if y_true[i] == y_true[j]:
            true_pairs.add((i, j))
        if y_pred[i] == y_pred[j]:
            pred_pairs.add((i, j))

    tp = len(true_pairs & pred_pairs)
    precision = tp / len(pred_pairs) if pred_pairs else 0.0
    recall = tp / len(true_pairs) if true_pairs else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1
