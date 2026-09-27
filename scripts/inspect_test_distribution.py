import sys
import os
import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.entity_resolution.config import cache_path

def main():
    best = pl.read_parquet(cache_path("test_best.parquet"))
    s1 = pl.read_parquet(cache_path("test_source1.parquet"), columns=["country"]).with_row_index("s1_idx")
    joined = best.join(s1, on="s1_idx")
    
    print("=" * 70)
    print("TEST MATCH COUNTS BY COUNTRY ACROSS THRESHOLDS")
    print("=" * 70)
    
    for t in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        sub = joined.filter(pl.col("p1") >= t)
        fr_cnt = sub.filter(pl.col("country") == "France").height
        in_cnt = sub.filter(pl.col("country") == "India").height
        us_cnt = sub.filter(pl.col("country") == "US").height
        s1_nonempty = sub.select("s1_idx").unique().height
        print(f"Threshold {t:.2f} | Total: {sub.height:<9,} | NonEmpty S1: {s1_nonempty:<9,} | France: {fr_cnt:<8,} | India: {in_cnt:<8,} | US: {us_cnt:<8,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
