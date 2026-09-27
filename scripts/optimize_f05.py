import time
import sys
import os
import polars as pl
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import cache_path
from src.entity_resolution.metrics import macro_f05
from src.entity_resolution.train import label_table

sys.stdout.reconfigure(encoding='utf-8')

def evaluate(pred_pairs, val_s1, tp):
    """pred_pairs: DataFrame with (s1_id, m_id)"""
    m_all = macro_f05(pred_pairs, tp, val_s1["entity_id"])
    m_in = macro_f05(pred_pairs, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])
    m_us = macro_f05(pred_pairs, tp, val_s1.filter(pl.col("country") == "US")["entity_id"])
    return {
        "macro_f05": m_all["macro_f05"],
        "precision": m_all["macro_precision"],
        "recall": m_all["macro_recall"],
        "singleton_acc": m_all["singleton_acc"],
        "in_f05": m_in["macro_f05"],
        "in_prec": m_in["macro_precision"],
        "in_rec": m_in["macro_recall"],
        "us_f05": m_us["macro_f05"],
        "us_prec": m_us["macro_precision"],
        "us_rec": m_us["macro_recall"],
    }

def main():
    t0 = time.time()
    s1, q, lab, tp = label_table()
    val_s1 = s1.filter(pl.col("is_val"))
    print(f"Validation entities: {val_s1.height:,} ({time.time()-t0:.1f}s)", flush=True)

    # Load train_best
    best = pl.read_parquet(cache_path("train_best.parquet"))
    print(f"Loaded train_best: {best.height:,} ({time.time()-t0:.1f}s)", flush=True)

    # Join metadata
    df = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
             .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
    
    # Add diff column
    df = df.with_columns((pl.col("p1") - pl.col("p2")).alias("gap"))
    print(f"Prediction table ready: {df.height:,} rows ({time.time()-t0:.1f}s)", flush=True)

    # Grid search over India threshold, US threshold, and margin gap
    print("\n--- GRID SEARCH: Country Thresholds & Ambiguity Margin ---", flush=True)
    results = []
    
    for t_in in [0.60, 0.65, 0.70, 0.75, 0.80]:
        for t_us in [0.60, 0.65, 0.70, 0.75]:
            for gap in [0.0, 0.05, 0.10, 0.15]:
                pred = df.filter(
                    (pl.col("gap") >= gap) &
                    (
                        ((pl.col("country") == "India") & (pl.col("p1") >= t_in)) |
                        ((pl.col("country") == "US") & (pl.col("p1") >= t_us))
                    )
                ).select("s1_id", "m_id")
                
                res = evaluate(pred, val_s1, tp)
                results.append((t_in, t_us, gap, res))
                print(f"t_IN={t_in:.2f} t_US={t_us:.2f} gap={gap:.2f} | F0.5={res['macro_f05']:.4f} (Prec={res['precision']:.4f}, Rec={res['recall']:.4f}) | IN={res['in_f05']:.4f} US={res['us_f05']:.4f}", flush=True)

    # Sort by Macro F0.5
    results.sort(key=lambda x: x[3]["macro_f05"], reverse=True)
    best_t_in, best_t_us, best_gap, best_res = results[0]
    print("\n" + "=" * 60, flush=True)
    print(f"TOP CONFIG: t_IN={best_t_in} t_US={best_t_us} gap={best_gap}", flush=True)
    print(f"Overall Macro F0.5: {best_res['macro_f05']:.4f} | Prec: {best_res['precision']:.4f} | Rec: {best_res['recall']:.4f} | Sing: {best_res['singleton_acc']:.4f}", flush=True)
    print(f"India   Macro F0.5: {best_res['in_f05']:.4f} | Prec: {best_res['in_prec']:.4f} | Rec: {best_res['in_rec']:.4f}", flush=True)
    print(f"US      Macro F0.5: {best_res['us_f05']:.4f} | Prec: {best_res['us_prec']:.4f} | Rec: {best_res['us_rec']:.4f}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
