"""
Phase 1 Forensic Diagnostic Script
Evaluates the baseline model (threshold = 0.65) on the 20% validation split.
Computes fine-grained breakdowns across:
- Overall (Macro F0.5, Precision, Recall, Singleton Acc)
- By Country (US vs India)
- By Source (S2 vs S3)
- By Singleton vs Non-Singleton
- By S1 Ground Truth Match Count (0, 1, 2, 3-5, >5)
- Candidate Ranking & Probability distribution
- Validation A (Train distribution: 60% US / 40% India) vs
  Validation B (Test distribution: 38.3% US / 46.8% India / 15.0% Unseen)
"""
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

def evaluate_subset(pred: pl.DataFrame, truth: pl.DataFrame, all_s1_ids: pl.Series):
    return macro_f05(pred, truth, all_s1_ids)

def main():
    t0 = time.time()
    print("=" * 70)
    print("PHASE 1 FORENSIC ANALYSIS: BASELINE PERFORMANCE BREAKDOWN")
    print("=" * 70)

    # 1. Load S1 metadata and ground truth
    s1, q, lab, tp = label_table()
    val_s1 = s1.filter(pl.col("is_val"))
    print(f"Loaded validation set: {val_s1.height:,} S1 entities ({time.time()-t0:.1f}s)")

    # 2. Load train_best predictions (threshold = 0.65)
    best = pl.read_parquet(cache_path("train_best.parquet"))
    print(f"Loaded train_best: {best.height:,} queries ({time.time()-t0:.1f}s)")

    # Join metadata
    df = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
             .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
    
    # Baseline matches at threshold = 0.65
    matches_65 = df.filter(pl.col("p1") >= 0.65).select("s1_id", "m_id")
    print(f"Total predicted matches (p1 >= 0.65): {matches_65.height:,} ({time.time()-t0:.1f}s)")

    # 3. Overall Performance
    print("\n" + "-" * 50)
    print("1. OVERALL VALIDATION METRICS (Threshold = 0.65)")
    print("-" * 50)
    overall = evaluate_subset(matches_65, tp, val_s1["entity_id"])
    print(f"  Macro F0.5:        {overall['macro_f05']:.4f}")
    print(f"  Macro Precision:   {overall['macro_precision']:.4f}")
    print(f"  Macro Recall:      {overall['macro_recall']:.4f}")
    print(f"  Singleton Acc:     {overall['singleton_acc']:.4f}")

    # 4. By Country
    print("\n" + "-" * 50)
    print("2. BY COUNTRY BREAKDOWN")
    print("-" * 50)
    for c in ["US", "India"]:
        sub_s1 = val_s1.filter(pl.col("country") == c)["entity_id"]
        res_c = evaluate_subset(matches_65, tp, sub_s1)
        print(f"  Country: {c:<8} (Entities: {len(sub_s1):,})")
        print(f"    Macro F0.5:      {res_c['macro_f05']:.4f}")
        print(f"    Macro Precision: {res_c['macro_precision']:.4f}")
        print(f"    Macro Recall:    {res_c['macro_recall']:.4f}")
        print(f"    Singleton Acc:   {res_c['singleton_acc']:.4f}")

    # 5. By Query Source (S2 vs S3)
    print("\n" + "-" * 50)
    print("3. BY QUERY SOURCE (S2 vs S3)")
    print("-" * 50)
    for src in ["S2", "S3"]:
        m_src = matches_65.filter(pl.col("m_id").str.starts_with(f"{src}-"))
        tp_src = tp.filter(pl.col("m_id").str.starts_with(f"{src}-"))
        res_src = evaluate_subset(m_src, tp_src, val_s1["entity_id"])
        print(f"  Source {src}:")
        print(f"    Macro F0.5:      {res_src['macro_f05']:.4f}")
        print(f"    Macro Precision: {res_src['macro_precision']:.4f}")
        print(f"    Macro Recall:    {res_src['macro_recall']:.4f}")

    # 6. Singleton vs Non-Singleton
    print("\n" + "-" * 50)
    print("4. SINGLETON VS NON-SINGLETON ANALYSIS")
    print("-" * 50)
    truth_counts = tp.filter(pl.col("s1_id").is_in(val_s1["entity_id"])) \
                     .group_by("s1_id").len().rename({"len": "truth_cnt"})
    
    val_with_counts = val_s1.select(pl.col("entity_id").alias("s1_id"), "country") \
                            .join(truth_counts, on="s1_id", how="left") \
                            .with_columns(pl.col("truth_cnt").fill_null(0))

    pred_counts = matches_65.filter(pl.col("s1_id").is_in(val_s1["entity_id"])) \
                            .group_by("s1_id").len().rename({"len": "pred_cnt"})
    
    val_eval = val_with_counts.join(pred_counts, on="s1_id", how="left") \
                              .with_columns(pl.col("pred_cnt").fill_null(0))

    singletons = val_eval.filter(pl.col("truth_cnt") == 0)
    non_singletons = val_eval.filter(pl.col("truth_cnt") > 0)
    sing_fp = singletons.filter(pl.col("pred_cnt") > 0)

    print(f"  Total Singletons:           {singletons.height:,} ({singletons.height/val_s1.height*100:.2f}% of val)")
    print(f"  Singleton False Positives:  {sing_fp.height:,} ({sing_fp.height/singletons.height*100:.2f}% error rate)")
    print(f"  Singleton Accuracy:         {1 - sing_fp.height/singletons.height:.4f}")
    
    # Non-singleton performance
    non_sing_res = evaluate_subset(matches_65, tp, non_singletons["s1_id"])
    print(f"\n  Total Non-Singletons:       {non_singletons.height:,} ({non_singletons.height/val_s1.height*100:.2f}% of val)")
    print(f"    Macro F0.5:               {non_sing_res['macro_f05']:.4f}")
    print(f"    Macro Precision:          {non_sing_res['macro_precision']:.4f}")
    print(f"    Macro Recall:             {non_sing_res['macro_recall']:.4f}")

    # 7. By S1 Ground Truth Match Count Bins
    print("\n" + "-" * 50)
    print("5. PERFORMANCE BY GROUND TRUTH MATCH COUNT")
    print("-" * 50)
    bins = [
        ("0 matches (singletons)", val_eval.filter(pl.col("truth_cnt") == 0)["s1_id"]),
        ("1 match", val_eval.filter(pl.col("truth_cnt") == 1)["s1_id"]),
        ("2 matches", val_eval.filter(pl.col("truth_cnt") == 2)["s1_id"]),
        ("3-5 matches", val_eval.filter((pl.col("truth_cnt") >= 3) & (pl.col("truth_cnt") <= 5))["s1_id"]),
        (">5 matches", val_eval.filter(pl.col("truth_cnt") > 5)["s1_id"]),
    ]
    for name, s1_subset in bins:
        res_b = evaluate_subset(matches_65, tp, s1_subset)
        print(f"  Bucket: {name:<25} | Entities: {len(s1_subset):<8} | F0.5: {res_b['macro_f05']:.4f} | Prec: {res_b['macro_precision']:.4f} | Rec: {res_b['macro_recall']:.4f}")

    # 8. Validation A vs Validation B (Distribution Shift)
    print("\n" + "-" * 50)
    print("6. VALIDATION A (Train Dist) VS VALIDATION B (Test Dist)")
    print("-" * 50)
    # Val A: Standard (40% IN / 60% US)
    val_a = overall['macro_f05']
    # Val B: Test weighted (46.8% IN / 38.3% US / 15.0% Unseen/France)
    us_f05 = evaluate_subset(matches_65, tp, val_s1.filter(pl.col("country") == "US")["entity_id"])['macro_f05']
    in_f05 = evaluate_subset(matches_65, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])['macro_f05']
    # If unseen behaves at average of IN and US:
    val_b_mid = 0.468 * in_f05 + 0.383 * us_f05 + 0.150 * ((in_f05 + us_f05) / 2)
    # If unseen behaves like IN:
    val_b_pessimistic = 0.468 * in_f05 + 0.383 * us_f05 + 0.150 * in_f05

    print(f"  Validation A (Train weights: 60.0% US, 40.0% IN):         {val_a:.4f}")
    print(f"  Validation B (Test weights:  38.3% US, 46.8% IN, 15% FR): {val_b_mid:.4f} (pessimistic: {val_b_pessimistic:.4f})")
    print(f"  Real Official Leaderboard Score:                          0.9418")
    print(f"  --> Discrepancy between Val B and Leaderboard:            {abs(val_b_mid - 0.9418):.4f}")

    print("\n" + "=" * 70)
    print(f"Phase 1 diagnosis complete in {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    main()
