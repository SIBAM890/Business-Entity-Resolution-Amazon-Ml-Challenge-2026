"""Step 3: build pairwise features for a split."""
import polars as pl

from .config import cache_path
from .features import compute_features, s1_extra_table, FEATURES


def build(split: str):
    out_dir = cache_path(f"{split}_feats")
    out_dir.mkdir(parents=True, exist_ok=True)
    done_marker = out_dir / "done"
    if done_marker.exists():
        print(f"[{split}] features already built, skipping")
        return

    s1 = pl.read_parquet(cache_path(f"{split}_source1_norm.parquet")).with_row_index("s1_idx")
    s2 = pl.read_parquet(cache_path(f"{split}_source2_norm.parquet")).with_columns(is_s3=pl.lit(False))
    s3 = pl.read_parquet(cache_path(f"{split}_source3_norm.parquet")).with_columns(is_s3=pl.lit(True))
    q = pl.concat([s2, s3]).with_row_index("q_idx")

    s1_extra = s1_extra_table(s1, cache_path(f"{split}_cands"))

    for f in sorted(cache_path(f"{split}_cands").glob("*.parquet")):
        if f.name.endswith(".done"):
            continue
        out_file = out_dir / f.name
        if out_file.exists():
            continue
        c = pl.read_parquet(f)
        feats = compute_features(c, q, s1, s1_extra)
        feats.write_parquet(out_file.with_suffix(".tmp"))
        out_file.with_suffix(".tmp").replace(out_file)
        print(f"  wrote {out_file}")

    done_marker.touch()
    print(f"[{split}] features done")


def main():
    for split in ("train", "test"):
        build(split)


if __name__ == "__main__":
    main()