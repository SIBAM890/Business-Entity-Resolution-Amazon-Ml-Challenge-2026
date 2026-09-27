import time
import sys
import os
import psutil
import polars as pl
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

def get_process_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def run_blocking_benchmark(sample_size=20000, seed=42):
    print("=" * 60)
    print(f"BENCHMARKING BLOCKING RECALL (Sample: {sample_size} S1 entities)")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load Ground Truth
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet")
    gt_with_matches = gt_df.filter(pl.col("matched_entity_ids") != "")
    print(f"Loaded GT: {gt_df.height:,} S1 entities, {gt_with_matches.height:,} non-empty.")
    
    # 2. Sample 20k validation S1 entities with matches (matches measure_blocking_recall.py)
    val_sample = gt_with_matches.sample(n=sample_size, seed=seed)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())
    
    # Build ground truth mapping for validation sample
    y_true = {}
    total_true = 0
    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        y_true[row["source1_entity_id"]] = matches
        total_true += len(matches)
    print(f"Sample contains {len(val_s1_set):,} S1 entities with {total_true:,} true matches.")
    
    # 3. Load Normalized Tables
    print("Loading normalized S1, S2, S3...")
    t_load = time.time()
    s1_df = pl.read_parquet("cache/train_source1_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s2_df = pl.read_parquet("cache/train_source2_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s3_df = pl.read_parquet("cache/train_source3_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    queries_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    print(f"Loaded all tables in {time.time() - t_load:.2f}s. RAM: {get_process_memory():.1f} MB")
    
    # 4. Filter S1 to the validation set for fast evaluation
    s1_val = s1_df.filter(pl.col("entity_id").is_in(val_s1_set))
    print(f"Filtered S1 to sample: {s1_val.height:,} rows.")
    
    # Precompute compressed names (no spaces)
    s1_val = s1_val.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    queries_df = queries_df.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    
    # Count token frequencies in queries for rare token indexing
    print("Counting name token frequencies in queries...")
    t_count = time.time()
    token_counts = Counter()
    for row in queries_df.select(["country", "name_core"]).iter_rows(named=True):
        c = row["country"]
        for tok in set(row["name_core"].split()):
            if len(tok) >= 3:
                token_counts[(c, tok)] += 1
    print(f"Token counts done in {time.time() - t_count:.2f}s. Distinct tokens: {len(token_counts):,}")
    
    # 5. Build Inverted Indexes on S1 validation set
    print("Building Inverted Indexes on S1 sample...")
    exact_idx = defaultdict(list)
    compact_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    
    for row in s1_val.iter_rows(named=True):
        eid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        
        # Pass 1: exact core name
        if nc:
            exact_idx[(c, nc)].append(eid)
            
        # Pass 2: compact name
        if comp:
            compact_idx[(c, comp)].append(eid)
            
        # Pass 3: rare name tokens
        for tok in set(nc.split()):
            if len(tok) >= 3 and token_counts.get((c, tok), 0) <= 500:
                rare_tok_idx[(c, tok)].append(eid)
                
        # Pass 4: PIN + first 3 chars of name
        first3 = comp[:3] if len(comp) >= 3 else comp
        for p in nums:
            if len(p) >= 5: # likely PIN/postal code
                pin_prefix_idx[(c, p, first3)].append(eid)
                
        # Pass 5: street number + first address token
        if nums and addr_toks:
            first_num = nums[0]
            # non-numeric address token
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 3:
                    num_addr_idx[(c, first_num, at)].append(eid)
                    break

    print(f"Indexes built. RAM: {get_process_memory():.1f} MB")
    
    # 6. Stream queries and collect candidates for S1 sample
    print("Matching queries against S1 sample...")
    t_match = time.time()
    
    cands_by_s1 = defaultdict(set)
    
    for row in queries_df.iter_rows(named=True):
        qid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        first3 = comp[:3] if len(comp) >= 3 else comp
        
        matched_s1s = set()
        
        # P1: Exact core name
        if nc and (c, nc) in exact_idx:
            matched_s1s.update(exact_idx[(c, nc)])
            
        # P2: Compact name
        if comp and (c, comp) in compact_idx:
            matched_s1s.update(compact_idx[(c, comp)])
            
        # P3: Rare token
        for tok in set(nc.split()):
            if len(tok) >= 3 and (c, tok) in rare_tok_idx:
                matched_s1s.update(rare_tok_idx[(c, tok)])
                
        # P4: PIN + first3
        for p in nums:
            if len(p) >= 5 and (c, p, first3) in pin_prefix_idx:
                matched_s1s.update(pin_prefix_idx[(c, p, first3)])
                
        # P5: Street number + address token
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 3:
                    if (c, first_num, at) in num_addr_idx:
                        matched_s1s.update(num_addr_idx[(c, first_num, at)])
                    break
                    
        for s1_id in matched_s1s:
            cands_by_s1[s1_id].add(qid)
            
    print(f"Matching finished in {time.time() - t_match:.2f}s.")
    
    # 7. Evaluate Recall and Statistics
    recovered = 0
    cand_counts = []
    
    for s1_id, true_matches in y_true.items():
        found = cands_by_s1.get(s1_id, set())
        recovered += len(found.intersection(true_matches))
        cand_counts.append(len(found))
        
    recall = recovered / total_true if total_true > 0 else 0.0
    c_arr = np.array(cand_counts)
    
    print("\n" + "=" * 60)
    print("BLOCKING BENCHMARK RESULTS:")
    print("=" * 60)
    print(f"Sample Size:         {sample_size:,} S1 entities")
    print(f"Total True Pairs:    {total_true:,}")
    print(f"Recovered Pairs:     {recovered:,}")
    print(f"Blocking Recall:     {recall:.4f} ({recall*100:.2f}%)")
    print(f"Total Candidates:    {c_arr.sum():,}")
    print(f"Mean Cands / S1:     {c_arr.mean():.2f}")
    print(f"Median Cands / S1:   {np.median(c_arr):.2f}")
    print(f"P95 Cands / S1:      {np.percentile(c_arr, 95):.2f}")
    print(f"Max Cands / S1:      {c_arr.max():,}")
    print(f"S1 with 0 Cands:     {(c_arr == 0).sum():,} ({(c_arr == 0).mean()*100:.2f}%)")
    print(f"Total Runtime:       {time.time() - t0:.2f}s")
    print(f"Peak RAM:            {get_process_memory():.1f} MB")
    print("=" * 60)

if __name__ == "__main__":
    run_blocking_benchmark(sample_size=20000)
