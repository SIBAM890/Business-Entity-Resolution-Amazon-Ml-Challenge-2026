import time
import sys
import polars as pl
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

def profile_passes(sample_size=5000, seed=42):
    print("=" * 60)
    print(f"PROFILING PASS CONTRIBUTIONS (Sample: {sample_size} S1 entities)")
    print("=" * 60)
    
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
    
    name_token_counts = Counter()
    addr_token_counts = Counter()
    for row in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c, nc, an = row["country"], row["name_core"], row["addr_norm"]
        for tok in set(nc.split()): name_token_counts[(c, tok)] += 1
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit(): addr_token_counts[(c, atok)] += 1
            
    # Inverted indexes per pass
    pass_names = [
        "P1_Exact_Core",
        "P2_Compact",
        "P3_Rare_Tok_500",
        "P3b_Rare_Tok_1000",
        "P4_Name_Bigram",
        "P5_PIN_Prefix4",
        "P6_StreetNum_AddrTok",
        "P7_Rare_Addr_200",
    ]
    
    indexes = {p: defaultdict(list) for p in pass_names}
    
    for row in s1_val.iter_rows(named=True):
        eid, c, nc, comp = row["entity_id"], row["country"], row["name_core"], row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        
        if nc: indexes["P1_Exact_Core"][(c, nc)].append(eid)
        if comp: indexes["P2_Compact"][(c, comp)].append(eid)
        for t in set(toks):
            cnt = name_token_counts.get((c, t), 0)
            if len(t) >= 2 and cnt <= 500:
                indexes["P3_Rare_Tok_500"][(c, t)].append(eid)
            elif len(t) >= 2 and cnt <= 1000:
                indexes["P3b_Rare_Tok_1000"][(c, t)].append(eid)
                
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    indexes["P4_Name_Bigram"][(c, st[i], st[j])].append(eid)
                    
        for p in nums:
            if len(p) >= 5: indexes["P5_PIN_Prefix4"][(c, p, prefix4)].append(eid)
            
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if not at.isdigit() and len(at) >= 3:
                    indexes["P6_StreetNum_AddrTok"][(c, first_num, at)].append(eid)
                    break
                    
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and addr_token_counts.get((c, at), 0) <= 200:
                indexes["P7_Rare_Addr_200"][(c, at)].append(eid)
                
    # Track candidate pairs per pass
    pass_candidates = {p: set() for p in pass_names}
    
    print("Streaming queries...")
    for row in queries_df.iter_rows(named=True):
        qid, c, nc, comp = row["entity_id"], row["country"], row["name_core"], row["name_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        prefix4 = comp[:4] if len(comp) >= 4 else comp
        
        if nc and (c, nc) in indexes["P1_Exact_Core"]:
            for s1_id in indexes["P1_Exact_Core"][(c, nc)]: pass_candidates["P1_Exact_Core"].add((s1_id, qid))
            
        if comp and (c, comp) in indexes["P2_Compact"]:
            for s1_id in indexes["P2_Compact"][(c, comp)]: pass_candidates["P2_Compact"].add((s1_id, qid))
            
        for t in set(toks):
            if (c, t) in indexes["P3_Rare_Tok_500"]:
                for s1_id in indexes["P3_Rare_Tok_500"][(c, t)]: pass_candidates["P3_Rare_Tok_500"].add((s1_id, qid))
            if (c, t) in indexes["P3b_Rare_Tok_1000"]:
                for s1_id in indexes["P3b_Rare_Tok_1000"][(c, t)]: pass_candidates["P3b_Rare_Tok_1000"].add((s1_id, qid))
                
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    if (c, st[i], st[j]) in indexes["P4_Name_Bigram"]:
                        for s1_id in indexes["P4_Name_Bigram"][(c, st[i], st[j])]: pass_candidates["P4_Name_Bigram"].add((s1_id, qid))
                        
        for p in nums:
            if len(p) >= 5 and (c, p, prefix4) in indexes["P5_PIN_Prefix4"]:
                for s1_id in indexes["P5_PIN_Prefix4"][(c, p, prefix4)]: pass_candidates["P5_PIN_Prefix4"].add((s1_id, qid))
                
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if not at.isdigit() and len(at) >= 3:
                    if (c, first_num, at) in indexes["P6_StreetNum_AddrTok"]:
                        for s1_id in indexes["P6_StreetNum_AddrTok"][(c, first_num, at)]: pass_candidates["P6_StreetNum_AddrTok"].add((s1_id, qid))
                    break
                    
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and (c, at) in indexes["P7_Rare_Addr_200"]:
                for s1_id in indexes["P7_Rare_Addr_200"][(c, at)]: pass_candidates["P7_Rare_Addr_200"].add((s1_id, qid))
                
    print("\n" + "=" * 80)
    print(f"{'Pass Name':<25} | {'Cand Count':<12} | {'Cands/S1':<10} | {'True Recovered':<15} | {'Marginal Gain':<15}")
    print("=" * 80)
    
    cumulative_cands = set()
    cumulative_recovered = set()
    
    for p in pass_names:
        cands = pass_candidates[p]
        recovered = cands.intersection(all_true_pairs)
        new_recovered = recovered - cumulative_recovered
        cumulative_recovered.update(recovered)
        cumulative_cands.update(cands)
        
        print(f"{p:<25} | {len(cands):<12,} | {len(cands)/sample_size:<10.1f} | {len(recovered):<5,} ({len(recovered)/total_true*100:5.2f}%) | +{len(new_recovered):<5,} (+{len(new_recovered)/total_true*100:5.2f}%)")
        
    print("=" * 80)
    print(f"CUMULATIVE TOTAL:         | {len(cumulative_cands):<12,} | {len(cumulative_cands)/sample_size:<10.1f} | {len(cumulative_recovered):<5,} ({len(cumulative_recovered)/total_true*100:5.2f}%) |")
    print("=" * 80)

if __name__ == "__main__":
    profile_passes(sample_size=5000)
