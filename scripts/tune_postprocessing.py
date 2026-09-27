import time
import sys
import os
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import cache_path
from src.entity_resolution.metrics import macro_f05
from src.entity_resolution.train import label_table, val_fold

sys.stdout.reconfigure(encoding='utf-8')

def evaluate_predictions(pred_ids: pl.DataFrame, val_s1: pl.DataFrame, tp: pl.DataFrame):
    """pred_ids: (s1_id, m_id)"""
    m = macro_f05(pred_ids, tp, val_s1["entity_id"])
    m_in = macro_f05(pred_ids, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])
    m_us = macro_f05(pred_ids, tp, val_s1.filter(pl.col("country") == "US")["entity_id"])
    return {
        "macro_f05": m["macro_f05"],
        "precision": m["macro_precision"],
        "recall": m["macro_recall"],
        "singleton_acc": m["singleton_acc"],
        "f05_India": m_in["macro_f05"],
        "f05_US": m_us["macro_f05"],
    }

def main():
    print("=" * 60)
    print("FAST POST-PROCESSING & THRESHOLD OPTIMIZATION")
    print("=" * 60)
    
    t0 = time.time()
    s1, q, lab, tp = label_table()
    val_s1 = s1.filter(pl.col("is_val"))
    print(f"Validation fold: {val_s1.height:,} S1 entities ({time.time()-t0:.1f}s)")
    
    # Load cached validation predictions
    best = pl.read_parquet(cache_path("train_best.parquet"))
    print(f"Loaded train_best: {best.height:,} queries ({time.time()-t0:.1f}s)")
    
    # Join with metadata for fast filtering
    ids = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
              .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
    print(f"Joined prediction table ready: {ids.height:,} rows ({time.time()-t0:.1f}s)")
    
    # Baseline threshold 0.65
    print("\n--- 1. Baseline (Threshold = 0.65) ---")
    pred_base = ids.filter(pl.col("p1") >= 0.65).select("s1_id", "m_id")
    res_base = evaluate_predictions(pred_base, val_s1, tp)
    print(f"Baseline F0.5: {res_base['macro_f05']:.4f} | Prec: {res_base['precision']:.4f} | Rec: {res_base['recall']:.4f} | Sing: {res_base['singleton_acc']:.4f}")
    print(f"  India F0.5: {res_base['f05_India']:.4f} | US F0.5: {res_base['f05_US']:.4f}")
    
    # 2. Sweep Score Gaps (p1 - p2 >= gap)
    print("\n--- 2. Margin Gap Sweep (at base threshold 0.60 and 0.65) ---")
    for t in [0.55, 0.60, 0.65]:
        for gap in [0.0, 0.05, 0.10, 0.15, 0.20]:
            pred = ids.filter((pl.col("p1") >= t) & ((pl.col("p1") - pl.col("p2")) >= gap)).select("s1_id", "m_id")
            res = evaluate_predictions(pred, val_s1, tp)
            print(f"Threshold: {t:.2f} | Gap: {gap:.2f} -> F0.5: {res['macro_f05']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | Sing: {res['singleton_acc']:.4f}")

    # 3. Per-Country Adaptive Thresholds
    print("\n--- 3. Per-Country Adaptive Threshold Sweep ---")
    best_comb = None
    best_f05 = 0.0
    
    for t_in in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for t_us in [0.55, 0.60, 0.65, 0.70, 0.75]:
            pred = ids.filter(
                ((pl.col("country") == "India") & (pl.col("p1") >= t_in)) |
                ((pl.col("country") == "US") & (pl.col("p1") >= t_us))
            ).select("s1_id", "m_id")
            res = evaluate_predictions(pred, val_s1, tp)
            if res["macro_f05"] > best_f05:
                best_f05 = res["macro_f05"]
                best_comb = (t_in, t_us, res)
                print(f"  [NEW BEST] India: {t_in:.2f}, US: {t_us:.2f} -> F0.5: {res['macro_f05']:.4f} (Prec: {res['precision']:.4f}, Rec: {res['recall']:.4f})")

    print(f"\nBest Adaptive: India={best_comb[0]}, US={best_comb[1]} -> Macro F0.5: {best_comb[2]['macro_f05']:.4f}")

if __name__ == "__main__":
    main()
