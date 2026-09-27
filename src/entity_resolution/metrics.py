"""Leaderboard metric: macro F0.5 over Source 1 entities (including singletons)."""
import polars as pl
import numpy as np


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, all_s1_ids: pl.Series):
    """
    pred: (s1_id, m_id) predicted matches
    truth: (s1_id, m_id) ground truth matches
    all_s1_ids: all S1 entity_ids in the evaluation set (including those with zero matches)
    
    Formula:
    - If truth == 0 and pred == 0: F0.5 = 1.0 (correct singleton)
    - If truth == 0 and pred > 0: F0.5 = 0.0 (false merge on singleton)
    - If truth > 0 and pred == 0: F0.5 = 0.0 (missed all matches)
    - If truth > 0 and pred > 0:
        prec = tp / pred
        rec = tp / truth
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    """
    tp_df = pred.join(truth, on=["s1_id", "m_id"])
    tp = tp_df.group_by("s1_id").len().rename({"len": "tp"})
    pred_cnt = pred.group_by("s1_id").len().rename({"len": "pred"})
    truth_cnt = truth.group_by("s1_id").len().rename({"len": "truth"})

    stats = (pl.DataFrame({"s1_id": all_s1_ids})
             .join(tp, on="s1_id", how="left")
             .join(pred_cnt, on="s1_id", how="left")
             .join(truth_cnt, on="s1_id", how="left")
             .fill_null(0))

    # Singletons where truth == 0
    is_singleton = (stats["truth"] == 0)
    singleton_correct = is_singleton & (stats["pred"] == 0)
    
    # Non-singletons
    has_truth = ~is_singleton
    has_pred = stats["pred"] > 0
    eval_mask = has_truth & has_pred
    
    tp_arr = stats["tp"].to_numpy().astype(np.float64)
    pred_arr = stats["pred"].to_numpy().astype(np.float64)
    truth_arr = stats["truth"].to_numpy().astype(np.float64)
    
    prec_arr = np.zeros(len(stats), dtype=np.float64)
    rec_arr = np.zeros(len(stats), dtype=np.float64)
    f05_arr = np.zeros(len(stats), dtype=np.float64)
    
    # Set 1.0 for correct singletons
    prec_arr[singleton_correct.to_numpy()] = 1.0
    rec_arr[singleton_correct.to_numpy()] = 1.0
    f05_arr[singleton_correct.to_numpy()] = 1.0
    
    # Evaluate entities with both truth > 0 and pred > 0
    idx = eval_mask.to_numpy()
    p = tp_arr[idx] / pred_arr[idx]
    r = tp_arr[idx] / truth_arr[idx]
    denom = 0.25 * p + r
    valid_denom = denom > 0
    f = np.zeros_like(p)
    f[valid_denom] = (1.25 * p[valid_denom] * r[valid_denom]) / denom[valid_denom]
    
    prec_arr[idx] = p
    rec_arr[idx] = r
    f05_arr[idx] = f

    return {
        "macro_f05": float(np.mean(f05_arr)),
        "macro_precision": float(np.mean(prec_arr)),
        "macro_recall": float(np.mean(rec_arr)),
        "singleton_acc": float(singleton_correct.sum() / is_singleton.sum()) if is_singleton.sum() > 0 else 1.0,
    }