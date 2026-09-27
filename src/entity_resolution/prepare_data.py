"""Step 1: read raw TSVs, write parquet for fast re-loading."""
import polars as pl

from .config import (TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
                    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3, cache_path)


def read_tsv(path, cols):
    return pl.read_csv(path, separator="\t", columns=cols, truncate_ragged_lines=True)


def main():
    print("Reading train sources...")
    pl.read_parquet(cache_path("train_source1.parquet")) if cache_path("train_source1.parquet").exists() else None
    for src, name in [(TRAIN_SOURCE1, "source1"), (TRAIN_SOURCE2, "source2"), (TRAIN_SOURCE3, "source3")]:
        out = cache_path(f"train_{name}.parquet")
        if not out.exists():
            df = read_tsv(src, ["entity_id", "business_name", "business_address", "country"])
            df.write_parquet(out)
            print(f"  wrote {out}")

    gt_out = cache_path("train_ground_truth.parquet")
    if not gt_out.exists():
        gt = pl.read_csv(TRAIN_GROUND_TRUTH, separator="\t", columns=["source1_entity_id", "matched_entity_ids"])
        gt.write_parquet(gt_out)
        print(f"  wrote {gt_out}")

    print("Reading test sources...")
    for src, name in [(TEST_SOURCE1, "source1"), (TEST_SOURCE2, "source2"), (TEST_SOURCE3, "source3")]:
        out = cache_path(f"test_{name}.parquet")
        if not out.exists():
            df = read_tsv(src, ["entity_id", "business_name", "business_address", "country"])
            df.write_parquet(out)
            print(f"  wrote {out}")

    print("Done.")


if __name__ == "__main__":
    main()