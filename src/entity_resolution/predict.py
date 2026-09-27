"""
Step 5: score the test candidates, assign each S2/S3 record to at most one S1 entity and write
output/matching_results.tsv and output/candidate_pairs.tsv.

Outputs are written in buckets of Source 1 rows so that ~100M candidate pairs never have to be
materialised as strings at once.
"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl

from .config import OUTPUT_DIR, cache_path
from .train import MODEL_PATH, best_per_query, id_tables, score_split

BUCKET = 200_000


def write_grouped(s1_ids: pl.Series, q_ids: pl.Series, pairs: pl.DataFrame, col: str, path):
    """pairs: (s1_idx, q_idx) integer table. Writes one row per S1 entity (file order), comma-joined
    S2/S3 ids sorted, empty when none. Returns (#rows, #non-empty rows)."""
    n_rows = n_nonempty = 0
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(f"source1_entity_id\t{col}\n")
        for s in range(0, len(s1_ids), BUCKET):
            e = min(s + BUCKET, len(s1_ids))
            part = pairs.filter((pl.col("s1_idx") >= s) & (pl.col("s1_idx") < e)).unique()
            part = part.with_columns(q_ids.gather(part["q_idx"]).alias("m_id")).sort("m_id")
            grouped = part.group_by("s1_idx").agg(pl.col("m_id").str.join(","))
            block = (pl.DataFrame({"s1_idx": pl.arange(s, e, eager=True).cast(pl.UInt32),
                                   "source1_entity_id": s1_ids.slice(s, e - s)})
                      .join(grouped.with_columns(pl.col("s1_idx").cast(pl.UInt32)), on="s1_idx", how="left")
                      .sort("s1_idx")
                      .select("source1_entity_id", pl.col("m_id").fill_null("").alias(col)))
            n_rows += block.height
            n_nonempty += block.filter(pl.col(col) != "").height
            fh.write(block.write_csv(separator="\t", quote_style="never", include_header=False))
    return n_rows, n_nonempty


def main(threshold=None, t_in=0.68, t_fr=0.66, t_us=0.65, min_gap=0.05, reuse_scored=True):
    t0 = time.time()
    s1, q = id_tables("test")
    s1_ids, q_ids = s1["entity_id"], q["entity_id"]
    s1_country = s1["country"]
    del s1, q

    scored_path = cache_path("test_scored.parquet")
    best_path = cache_path("test_best.parquet")

    if not best_path.exists():
        if scored_path.exists() and reuse_scored:
            scored = pl.read_parquet(scored_path)
        else:
            model = lgb.Booster(model_file=str(MODEL_PATH))
            scored = score_split(model, "test")
            scored.write_parquet(scored_path)
        print(f"scored {scored.height:,} test pairs ({time.time()-t0:.0f}s)", flush=True)

        cand_tsv = OUTPUT_DIR / "candidate_pairs.tsv"
        if not cand_tsv.exists():
            n, n_c = write_grouped(s1_ids, q_ids, scored.select("s1_idx", "q_idx"), "candidate_entity_ids", cand_tsv)
            print(f"candidate_pairs.tsv: {n:,} rows, {n_c:,} with candidates ({time.time()-t0:.0f}s)", flush=True)

        best = best_per_query(scored)
        del scored
        best.write_parquet(best_path)
    else:
        best = pl.read_parquet(best_path)
        print(f"loaded cached test_best: {best.height:,} queries ({time.time()-t0:.1f}s)", flush=True)

    # Filter matches
    if threshold is not None:
        # Uniform threshold mode
        matches = best.filter(pl.col("p1") >= threshold).select("s1_idx", "q_idx")
        desc = f"uniform threshold={threshold}"
    else:
        # High-Precision Country-Adaptive & Ambiguity Margin Guard mode
        country_df = pl.DataFrame({
            "s1_idx": pl.arange(0, len(s1_country), eager=True).cast(pl.UInt32),
            "country": s1_country
        })
        joined = best.join(country_df, on="s1_idx")
        cond = (
            (pl.col("p1") - pl.col("p2") >= min_gap) &
            (
                ((pl.col("country") == "India") & (pl.col("p1") >= t_in)) |
                ((pl.col("country") == "France") & (pl.col("p1") >= t_fr)) |
                ((pl.col("country") == "US") & (pl.col("p1") >= t_us)) |
                (~pl.col("country").is_in(["India", "France", "US"]) & (pl.col("p1") >= 0.68))
            )
        )
        matches = joined.filter(cond).select("s1_idx", "q_idx")
        desc = f"country-adaptive (IN={t_in}, FR={t_fr}, US={t_us}, gap={min_gap})"

    n, n_m = write_grouped(s1_ids, q_ids, matches, "matched_entity_ids", OUTPUT_DIR / "matching_results.tsv")
    print(f"matching_results.tsv ({desc}): {matches.height:,} matched records over "
          f"{n_m:,}/{n:,} S1 entities ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None, help="Uniform threshold override")
    parser.add_argument("--t-in", type=float, default=0.68, help="India threshold")
    parser.add_argument("--t-fr", type=float, default=0.66, help="France threshold")
    parser.add_argument("--t-us", type=float, default=0.65, help="US threshold")
    parser.add_argument("--gap", type=float, default=0.05, help="Minimum margin gap between p1 and p2")
    args = parser.parse_args()
    main(threshold=args.threshold, t_in=args.t_in, t_fr=args.t_fr, t_us=args.t_us, min_gap=args.gap)