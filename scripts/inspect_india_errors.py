import time
import sys
import os
import polars as pl
from rapidfuzz import fuzz

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.entity_resolution.config import cache_path
from src.entity_resolution.train import label_table

sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 60)
    print("INSPECTING INDIA ERRORS")
    print("=" * 60)
    
    s1, q, lab, tp = label_table()
    val_s1 = s1.filter(pl.col("is_val"))
    best = pl.read_parquet(cache_path("train_best.parquet"))
    
    # Load raw text for inspection
    s1_norm = pl.read_parquet(cache_path("train_source1_norm.parquet"))
    s2_norm = pl.read_parquet(cache_path("train_source2_norm.parquet")).with_columns(is_s3=pl.lit(False))
    s3_norm = pl.read_parquet("cache/train_source3_norm.parquet").with_columns(is_s3=pl.lit(True))
    q_norm = pl.concat([s2_norm, s3_norm])
    del s2_norm, s3_norm
    
    ids = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
              .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")
              
    pred = ids.filter(pl.col("p1") >= 0.65).select("s1_id", "m_id", "q_idx", "s1_idx", "p1")
    
    # Ground truth set
    true_pairs = set(zip(tp["s1_id"].to_list(), tp["m_id"].to_list()))
    
    # Find False Positives for India
    in_val_s1_ids = set(val_s1.filter(pl.col("country") == "India")["entity_id"].to_list())
    in_pred = pred.filter(pl.col("s1_id").is_in(in_val_s1_ids))
    
    fps = []
    for row in in_pred.iter_rows(named=True):
        if (row["s1_id"], row["m_id"]) not in true_pairs:
            fps.append(row)
            if len(fps) >= 20:
                break
                
    print(f"Sample False Positives in India (at threshold 0.65):")
    for row in fps:
        s1_row = s1_norm.filter(pl.col("entity_id") == row["s1_id"]).row(0, named=True)
        q_row = q_norm.filter(pl.col("entity_id") == row["m_id"]).row(0, named=True)
        
        print(f"\n[FP - Prob: {row['p1']:.3f}]")
        print(f"  S1 ({row['s1_id']}): Name: '{s1_row['name_core']}' | Addr: '{s1_row['addr_norm']}' | Nums: '{s1_row['addr_nums']}'")
        print(f"  Q  ({row['m_id']}): Name: '{q_row['name_core']}' | Addr: '{q_row['addr_norm']}' | Nums: '{q_row['addr_nums']}'")
        print(f"  Lev Ratio: {fuzz.ratio(s1_row['name_core'], q_row['name_core'])} | Addr Ratio: {fuzz.token_set_ratio(s1_row['addr_norm'], q_row['addr_norm'])}")

if __name__ == "__main__":
    main()
