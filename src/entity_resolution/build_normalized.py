"""Step 2: normalize names and addresses for a split (train or test). Fast vectorized version."""
import polars as pl

from .config import cache_path
from .normalization import normalize_name, normalize_address


def normalize_split(split: str):
    out_path = cache_path(f"{split}_source1_norm.parquet")
    if out_path.exists():
        print(f"[{split}] normalized Source 1 already exists, skipping")
    else:
        s1 = pl.read_parquet(cache_path(f"{split}_source1.parquet"))
        # Process in chunks for memory efficiency
        name_results = [normalize_name(x) for x in s1["business_name"].to_list()]
        addr_results = [normalize_address(x) for x in s1["business_address"].to_list()]
        
        res = s1.with_columns(
            pl.Series("name_norm", [r[0] for r in name_results]),
            pl.Series("name_core", [r[1] for r in name_results]),
            pl.Series("is_domain", [r[2] for r in name_results]),
            pl.Series("has_indic", [r[3] for r in name_results]),
            pl.Series("addr_norm", [r[0] for r in addr_results]),
            pl.Series("addr_nums", [r[1] for r in addr_results]),
        ).drop(["business_name", "business_address"])
        res.write_parquet(out_path)
        print(f"[{split}] wrote {out_path} ({res.height:,} rows)")

    for name in ("source2", "source3"):
        out_path = cache_path(f"{split}_{name}_norm.parquet")
        if out_path.exists():
            print(f"[{split}] normalized {name} already exists, skipping")
            continue
        src = pl.read_parquet(cache_path(f"{split}_{name}.parquet"))
        name_results = [normalize_name(x) for x in src["business_name"].to_list()]
        addr_results = [normalize_address(x) for x in src["business_address"].to_list()]
        
        res = src.with_columns(
            pl.Series("name_norm", [r[0] for r in name_results]),
            pl.Series("name_core", [r[1] for r in name_results]),
            pl.Series("is_domain", [r[2] for r in name_results]),
            pl.Series("has_indic", [r[3] for r in name_results]),
            pl.Series("addr_norm", [r[0] for r in addr_results]),
            pl.Series("addr_nums", [r[1] for r in addr_results]),
        ).drop(["business_name", "business_address"])
        res = res.with_columns(addr_missing=(pl.col("addr_norm") == ""))
        res.write_parquet(out_path)
        print(f"[{split}] wrote {out_path} ({res.height:,} rows)")


def main():
    for split in ("train", "test"):
        normalize_split(split)


if __name__ == "__main__":
    main()