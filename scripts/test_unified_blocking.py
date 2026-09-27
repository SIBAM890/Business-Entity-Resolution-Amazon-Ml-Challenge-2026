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

def run_unified_blocking_benchmark(sample_size=20000, seed=42):
    print("=" * 60)
    print(f"BENCHMARKING HIGH-PERFORMANCE MULTI-PASS BLOCKING")
    print(f"(Sample: {sample_size:,} S1 entities)")
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
        s1_id = row["source1_entity_id"]
        y_true[s1_id] = matches
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
    
    s1_val = s1_df.filter(pl.col("entity_id").is_in(val_s1_set)).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    del s1_df
    queries_df = queries_df.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    
    # 3. Frequency Counting
    print("Counting token frequencies...")
    t_count = time.time()
    name_token_counts = Counter()
    addr_token_counts = Counter()
    
    for row in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c, nc, an = row["country"], row["name_core"], row["addr_norm"]
        for tok in set(nc.split()):
            name_token_counts[(c, tok)] += 1
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit():
                addr_token_counts[(c, atok)] += 1
                
    print(f"Frequency counting complete in {time.time() - t_count:.2f}s.")
    
    # 4. Multi-Pass Inverted Indexes with Controlled Capacities
    print("Building Inverted Indexes on S1 sample...")
    exact_idx = defaultdict(list)
    compact_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    name_bigram_idx = defaultdict(list)
    prefix_city_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)
    street_pin_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    rare_addr_idx = defaultdict(list)
    
    RARE_NAME_THRESH = 1200
    RARE_ADDR_THRESH = 250
    
    for row in s1_val.iter_rows(named=True):
        eid, c, nc, comp = row["entity_id"], row["country"], row["name_core"], row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        prefix2 = comp[:2] if len(comp) >= 2 else comp
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        
        # P1: Exact core name
        if nc: exact_idx[(c, nc)].append(eid)
        
        # P2: Compact name
        if comp: compact_idx[(c, comp)].append(eid)
        
        # P3: Rare name tokens (count <= 1200, length >= 2)
        for t in set(toks):
            cnt = name_token_counts.get((c, t), 0)
            if len(t) >= 2 and cnt <= RARE_NAME_THRESH:
                rare_tok_idx[(c, t)].append(eid)
                
        # P4: Name Bigrams - with frequency guard
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    c1 = name_token_counts.get((c, st[i]), 0)
                    c2 = name_token_counts.get((c, st[j]), 0)
                    if min(c1, c2) <= 4500 and len(st[i]) >= 3 and len(st[j]) >= 3:
                        name_bigram_idx[(c, st[i], st[j])].append(eid)
                        
        # P5: Prefix4 + City/Locality
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if len(at) >= 4 and not at.isdigit() and addr_token_counts.get((c, at), 0) <= 25000:
                    prefix_city_idx[(c, prefix4, at)].append(eid)
                    
        # P6: PIN + 2-char prefix
        for p in nums:
            if len(p) >= 5:
                pin_prefix_idx[(c, p, prefix2)].append(eid)
                
        # P7: Street number + PIN (same physical building)
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if len(p) >= 5:
                    street_pin_idx[(c, first_num, p)].append(eid)
                    
        # P8: Street number + 2nd address token
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 4:
                    num_addr_idx[(c, first_num, at)].append(eid)
                    break
                    
        # P9: Rare address token (count <= 150)
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and addr_token_counts.get((c, at), 0) <= RARE_ADDR_THRESH:
                rare_addr_idx[(c, at)].append(eid)

    print(f"Indexes built. RAM: {get_process_memory():.1f} MB")
    
    # 5. Query Streaming
    print("Matching queries against S1 sample...")
    t_match = time.time()
    cands_by_s1 = defaultdict(set)
    pass_hits = Counter()
    
    for row in queries_df.iter_rows(named=True):
        qid, c, nc, comp = row["entity_id"], row["country"], row["name_core"], row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        prefix2 = comp[:2] if len(comp) >= 2 else comp
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        
        matched_s1s = set()
        
        if nc and (c, nc) in exact_idx:
            s_set = exact_idx[(c, nc)]
            matched_s1s.update(s_set)
            pass_hits["P1_Exact"] += len(s_set)
            
        if comp and (c, comp) in compact_idx:
            s_set = compact_idx[(c, comp)]
            matched_s1s.update(s_set)
            pass_hits["P2_Compact"] += len(s_set)
            
        for t in set(toks):
            if (c, t) in rare_tok_idx:
                s_set = rare_tok_idx[(c, t)]
                matched_s1s.update(s_set)
                pass_hits["P3_Rare_Tok"] += len(s_set)
                
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    if (c, st[i], st[j]) in name_bigram_idx:
                        s_set = name_bigram_idx[(c, st[i], st[j])]
                        matched_s1s.update(s_set)
                        pass_hits["P4_Bigram"] += len(s_set)
                        
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if (c, prefix4, at) in prefix_city_idx:
                    s_set = prefix_city_idx[(c, prefix4, at)]
                    matched_s1s.update(s_set)
                    pass_hits["P5_Prefix4_City"] += len(s_set)
                    
        for p in nums:
            if len(p) >= 5 and (c, p, prefix2) in pin_prefix_idx:
                s_set = pin_prefix_idx[(c, p, prefix2)]
                matched_s1s.update(s_set)
                pass_hits["P6_PIN_Prefix"] += len(s_set)
                
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if len(p) >= 5 and (c, first_num, p) in street_pin_idx:
                    s_set = street_pin_idx[(c, first_num, p)]
                    matched_s1s.update(s_set)
                    pass_hits["P7_Street_PIN"] += len(s_set)
                    
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 4 and (c, first_num, at) in num_addr_idx:
                    s_set = num_addr_idx[(c, first_num, at)]
                    matched_s1s.update(s_set)
                    pass_hits["P8_Street_AddrTok"] += len(s_set)
                    break
                    
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and (c, at) in rare_addr_idx:
                s_set = rare_addr_idx[(c, at)]
                matched_s1s.update(s_set)
                pass_hits["P9_Rare_Addr"] += len(s_set)
                
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
    print("UNIFIED HIGH-PERFORMANCE BLOCKING BENCHMARK RESULTS:")
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
    print("Pass hit totals across queries:")
    for k, v in pass_hits.most_common():
        print(f"  {k:<20}: {v:,}")

if __name__ == "__main__":
    run_unified_blocking_benchmark(sample_size=20000)
