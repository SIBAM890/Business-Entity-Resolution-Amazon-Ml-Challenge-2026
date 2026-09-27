import time
import sys
import polars as pl
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

def test_prefix_city_pass(sample_size=5000, seed=42):
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet")
    gt_with_matches = gt_df.filter(pl.col("matched_entity_ids") != "")
    val_sample = gt_with_matches.sample(n=sample_size, seed=seed)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())
    
    y_true = {}
    total_true = 0
    all_true_pairs = set()
    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        s1_id = row["source1_entity_id"]
        y_true[s1_id] = matches
        total_true += len(matches)
        for m in matches:
            all_true_pairs.add((s1_id, m))
            
    s1_df = pl.read_parquet("cache/train_source1_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s2_df = pl.read_parquet("cache/train_source2_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s3_df = pl.read_parquet("cache/train_source3_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    queries_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    s1_val = s1_df.filter(pl.col("entity_id").is_in(val_s1_set)).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    del s1_df
    queries_df = queries_df.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    
    # Inverted index: (country, prefix4, addr_token) where addr_token count is between 100 and 10,000 (typical city/locality/street)
    addr_token_counts = Counter()
    for row in queries_df.select(["country", "addr_norm"]).iter_rows(named=True):
        c, an = row["country"], row["addr_norm"]
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit(): addr_token_counts[(c, atok)] += 1
            
    prefix_city_idx = defaultdict(list)
    for row in s1_val.iter_rows(named=True):
        eid, c, comp = row["entity_id"], row["country"], row["name_compact"]
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        if len(comp) >= 4:
            p4 = comp[:4]
            # Use last 2 tokens of address (usually city / state / district)
            for at in addr_toks[-3:]:
                if len(at) >= 4 and not at.isdigit() and addr_token_counts.get((c, at), 0) <= 20000:
                    prefix_city_idx[(c, p4, at)].append(eid)
                    
    print(f"Prefix+City index keys: {len(prefix_city_idx):,}")
    
    # Query streaming
    candidates = set()
    for row in queries_df.iter_rows(named=True):
        qid, c, comp = row["entity_id"], row["country"], row["name_compact"]
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        if len(comp) >= 4:
            p4 = comp[:4]
            for at in addr_toks[-3:]:
                if (c, p4, at) in prefix_city_idx:
                    for s1_id in prefix_city_idx[(c, p4, at)]:
                        candidates.add((s1_id, qid))
                        
    recovered = candidates.intersection(all_true_pairs)
    print(f"Prefix+City Pass alone:")
    print(f"  Total Candidates: {len(candidates):,} ({len(candidates)/sample_size:.1f} per S1)")
    print(f"  True Recovered:   {len(recovered):,} / {total_true:,} ({len(recovered)/total_true*100:.2f}%)")

if __name__ == "__main__":
    test_prefix_city_pass(sample_size=5000)
