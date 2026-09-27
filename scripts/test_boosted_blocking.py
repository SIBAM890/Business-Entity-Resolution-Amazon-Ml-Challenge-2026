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

def run_boosted_blocking_benchmark(sample_size=20000, seed=42):
    print("=" * 60)
    print(f"BENCHMARKING BOOSTED BLOCKING (Sample: {sample_size} S1 entities)")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load Ground Truth
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet")
    gt_with_matches = gt_df.filter(pl.col("matched_entity_ids") != "")
    val_sample = gt_with_matches.sample(n=sample_size, seed=seed)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())
    
    y_true = {}
    total_true = 0
    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        y_true[row["source1_entity_id"]] = matches
        total_true += len(matches)
    print(f"Sample contains {len(val_s1_set):,} S1 entities with {total_true:,} true matches.")
    
    # 2. Load Normalized Tables
    s1_df = pl.read_parquet("cache/train_source1_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s2_df = pl.read_parquet("cache/train_source2_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    s3_df = pl.read_parquet("cache/train_source3_norm.parquet", 
                            columns=["entity_id", "country", "name_norm", "name_core", "addr_norm", "addr_nums"])
    queries_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    s1_val = s1_df.filter(pl.col("entity_id").is_in(val_s1_set))
    del s1_df
    
    s1_val = s1_val.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    queries_df = queries_df.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    
    # 3. Frequency Counting for Inverted Indexes
    print("Counting token frequencies...")
    t_count = time.time()
    name_token_counts = Counter()
    addr_token_counts = Counter()
    
    for row in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c = row["country"]
        nc = row["name_core"]
        an = row["addr_norm"]
        for tok in set(nc.split()):
            name_token_counts[(c, tok)] += 1
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit():
                addr_token_counts[(c, atok)] += 1
                
    print(f"Frequency counting complete in {time.time() - t_count:.2f}s. RAM: {get_process_memory():.1f} MB")
    
    # 4. Build Multi-Pass Inverted Indexes on S1 sample
    print("Building Inverted Indexes on S1 sample...")
    exact_idx = defaultdict(list)
    compact_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    name_bigram_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    rare_addr_idx = defaultdict(list)
    
    RARE_NAME_THRESH = 1000
    RARE_ADDR_THRESH = 200
    
    for row in s1_val.iter_rows(named=True):
        eid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        
        # Pass 1: exact core name
        if nc:
            exact_idx[(c, nc)].append(eid)
            
        # Pass 2: compact name
        if comp:
            compact_idx[(c, comp)].append(eid)
            
        # Pass 3: rare name token (len >= 2, count <= RARE_NAME_THRESH)
        for tok in set(toks):
            if len(tok) >= 2 and name_token_counts.get((c, tok), 0) <= RARE_NAME_THRESH:
                rare_tok_idx[(c, tok)].append(eid)
                
        # Pass 4: 2-token name combination (for names with 2 to 5 tokens)
        if 2 <= len(toks) <= 5:
            sorted_toks = sorted(toks)
            for i in range(len(sorted_toks)):
                for j in range(i + 1, len(sorted_toks)):
                    t1, t2 = sorted_toks[i], sorted_toks[j]
                    if len(t1) >= 3 and len(t2) >= 3:
                        name_bigram_idx[(c, t1, t2)].append(eid)
                        
        # Pass 5: PIN + 4-char name prefix (or full comp if < 4)
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        for p in nums:
            if len(p) >= 5:
                pin_prefix_idx[(c, p, prefix4)].append(eid)
                
        # Pass 6: Street number + 2 address tokens
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if not at.isdigit() and len(at) >= 3:
                    num_addr_idx[(c, first_num, at)].append(eid)
                    break
                    
        # Pass 7: Rare address token (count <= RARE_ADDR_THRESH, len >= 4)
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and addr_token_counts.get((c, at), 0) <= RARE_ADDR_THRESH:
                rare_addr_idx[(c, at)].append(eid)

    print(f"Indexes built. RAM: {get_process_memory():.1f} MB")
    
    # 5. Stream queries and collect candidates
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
        toks = nc.split()
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        
        matched_s1s = set()
        
        # P1: Exact core name
        if nc and (c, nc) in exact_idx:
            matched_s1s.update(exact_idx[(c, nc)])
            
        # P2: Compact name
        if comp and (c, comp) in compact_idx:
            matched_s1s.update(compact_idx[(c, comp)])
            
        # P3: Rare token
        for tok in set(toks):
            if len(tok) >= 2 and (c, tok) in rare_tok_idx:
                matched_s1s.update(rare_tok_idx[(c, tok)])
                
        # P4: 2-token name combination
        if 2 <= len(toks) <= 5:
            sorted_toks = sorted(toks)
            for i in range(len(sorted_toks)):
                for j in range(i + 1, len(sorted_toks)):
                    t1, t2 = sorted_toks[i], sorted_toks[j]
                    if (c, t1, t2) in name_bigram_idx:
                        matched_s1s.update(name_bigram_idx[(c, t1, t2)])
                        
        # P5: PIN + 4-char prefix
        for p in nums:
            if len(p) >= 5 and (c, p, prefix4) in pin_prefix_idx:
                matched_s1s.update(pin_prefix_idx[(c, p, prefix4)])
                
        # P6: Street number + address token
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if not at.isdigit() and len(at) >= 3:
                    if (c, first_num, at) in num_addr_idx:
                        matched_s1s.update(num_addr_idx[(c, first_num, at)])
                    break
                    
        # P7: Rare address token
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and (c, at) in rare_addr_idx:
                matched_s1s.update(rare_addr_idx[(c, at)])
                
        for s1_id in matched_s1s:
            cands_by_s1[s1_id].add(qid)
            
    print(f"Matching finished in {time.time() - t_match:.2f}s.")
    
    # 6. Evaluate Recall and Statistics
    recovered = 0
    cand_counts = []
    
    for s1_id, true_matches in y_true.items():
        found = cands_by_s1.get(s1_id, set())
        recovered += len(found.intersection(true_matches))
        cand_counts.append(len(found))
        
    recall = recovered / total_true if total_true > 0 else 0.0
    c_arr = np.array(cand_counts)
    
    print("\n" + "=" * 60)
    print("BOOSTED BLOCKING BENCHMARK RESULTS:")
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
    run_boosted_blocking_benchmark(sample_size=20000)
