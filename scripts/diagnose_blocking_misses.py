import time
import sys
import os
import psutil
import polars as pl
import numpy as np
from collections import defaultdict, Counter
from rapidfuzz import fuzz

sys.stdout.reconfigure(encoding='utf-8')

def diagnose_misses(sample_size=5000, seed=42):
    print("=" * 60)
    print(f"DIAGNOSING BLOCKING MISSES (Sample: {sample_size} S1 entities)")
    print("=" * 60)
    
    # 1. Load Ground Truth
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet")
    gt_with_matches = gt_df.filter(pl.col("matched_entity_ids") != "")
    val_sample = gt_with_matches.sample(n=sample_size, seed=seed)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())
    
    y_true = {}
    total_true = 0
    all_target_queries = set()
    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        y_true[row["source1_entity_id"]] = matches
        total_true += len(matches)
        all_target_queries.update(matches)
        
    print(f"Sample has {len(val_s1_set):,} S1 entities, {total_true:,} true matches, {len(all_target_queries):,} distinct target query IDs.")
    
    # 2. Load Normalized Tables
    s1_df = pl.read_parquet("cache/train_source1_norm.parquet")
    s2_df = pl.read_parquet("cache/train_source2_norm.parquet")
    s3_df = pl.read_parquet("cache/train_source3_norm.parquet")
    queries_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df
    
    # Filter S1 to sample
    s1_val = s1_df.filter(pl.col("entity_id").is_in(val_s1_set)).with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    queries_df = queries_df.with_columns(
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact")
    )
    
    # Also index the target queries for instant diagnosis
    target_q_df = queries_df.filter(pl.col("entity_id").is_in(all_target_queries))
    s1_lookup = {r["entity_id"]: r for r in s1_val.iter_rows(named=True)}
    q_lookup = {r["entity_id"]: r for r in target_q_df.iter_rows(named=True)}
    
    # Baseline Inverted Indexes on S1
    token_counts = Counter()
    for row in queries_df.select(["country", "name_core"]).iter_rows(named=True):
        c = row["country"]
        for tok in set(row["name_core"].split()):
            if len(tok) >= 3:
                token_counts[(c, tok)] += 1
                
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
        first3 = comp[:3] if len(comp) >= 3 else comp
        
        if nc: exact_idx[(c, nc)].append(eid)
        if comp: compact_idx[(c, comp)].append(eid)
        for tok in set(nc.split()):
            if len(tok) >= 3 and token_counts.get((c, tok), 0) <= 500:
                rare_tok_idx[(c, tok)].append(eid)
        for p in nums:
            if len(p) >= 5: pin_prefix_idx[(c, p, first3)].append(eid)
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 3:
                    num_addr_idx[(c, first_num, at)].append(eid)
                    break

    # Now evaluate directly on all (s1_id, true_qid) pairs to see which ones are caught by baseline vs missed
    caught = []
    missed = []
    
    for s1_id, true_qids in y_true.items():
        s1_row = s1_lookup[s1_id]
        c = s1_row["country"]
        nc = s1_row["name_core"]
        comp = s1_row["name_compact"]
        nums = s1_row["addr_nums"].split() if s1_row["addr_nums"] else []
        addr_toks = s1_row["addr_norm"].split() if s1_row["addr_norm"] else []
        first3 = comp[:3] if len(comp) >= 3 else comp
        
        for qid in true_qids:
            if qid not in q_lookup:
                missed.append((s1_id, qid, "query_not_found"))
                continue
            q_row = q_lookup[qid]
            q_nc = q_row["name_core"]
            q_comp = q_row["name_compact"]
            q_nums = q_row["addr_nums"].split() if q_row["addr_nums"] else []
            q_addr_toks = q_row["addr_norm"].split() if q_row["addr_norm"] else []
            q_first3 = q_comp[:3] if len(q_comp) >= 3 else q_comp
            
            # Check baseline passes
            is_caught = False
            # P1: exact name
            if nc and q_nc == nc: is_caught = True
            # P2: compact
            elif comp and q_comp == comp: is_caught = True
            # P3: rare token
            elif any(len(t) >= 3 and token_counts.get((c, t), 0) <= 500 and t in q_nc.split() for t in nc.split()):
                is_caught = True
            # P4: PIN + first3
            elif any(len(p) >= 5 and p in q_nums and (first3 == q_first3) for p in nums):
                is_caught = True
            # P5: num + addr
            elif nums and q_nums and nums[0] == q_nums[0] and any(at in q_addr_toks for at in addr_toks[:3] if not at.isdigit() and len(at) >= 3):
                is_caught = True
                
            if is_caught:
                caught.append((s1_id, qid))
            else:
                missed.append((s1_id, qid, s1_row, q_row))
                
    baseline_recall = len(caught) / total_true
    print(f"\nBaseline Caught: {len(caught):,} / {total_true:,} ({baseline_recall*100:.2f}%)")
    print(f"Baseline Missed: {len(missed):,} / {total_true:,} ({len(missed)/total_true*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("DETAILED ERROR ANALYSIS ON 30 MISSED PAIRS:")
    print("=" * 60)
    
    # Categorize misses
    miss_reasons = Counter()
    
    for item in missed[:30]:
        s1_id, qid, s1_r, q_r = item
        s1_name, q_name = s1_r["name_core"], q_r["name_core"]
        s1_addr, q_addr = s1_r["addr_norm"], q_r["addr_norm"]
        s1_nums, q_nums = s1_r["addr_nums"], q_r["addr_nums"]
        
        name_ratio = fuzz.ratio(s1_name, q_name)
        token_sort = fuzz.token_sort_ratio(s1_name, q_name)
        shared_name_toks = set(s1_name.split()).intersection(q_name.split())
        shared_addr_toks = set(s1_addr.split()).intersection(q_addr.split())
        shared_nums = set(s1_nums.split()).intersection(q_nums.split())
        
        print(f"\n[Pair: {s1_id} <-> {qid}] (Country: {s1_r['country']})")
        print(f"  S1 Name:  '{s1_name}'")
        print(f"  Q Name:   '{q_name}' (fuzz.ratio: {name_ratio}, token_sort: {token_sort}, shared_toks: {shared_name_toks})")
        print(f"  S1 Addr:  '{s1_addr}'")
        print(f"  Q Addr:   '{q_addr}' (shared_addr_toks: {shared_addr_toks}, shared_nums: {shared_nums})")
        
        # Determine why it missed and what pass could capture it
        potentials = []
        if token_sort >= 70 or name_ratio >= 70:
            potentials.append("fuzzy_name_sim")
        if shared_name_toks:
            potentials.append(f"shared_common_name_token({[t for t in shared_name_toks]})")
        if any(len(p) >= 5 for p in shared_nums):
            potentials.append("shared_pin_alone")
        if shared_nums:
            potentials.append("shared_house_number")
        if len(shared_addr_toks) >= 2:
            potentials.append(f"shared_multi_addr_toks({len(shared_addr_toks)})")
            
        print(f"  Potential Recovery Passes: {potentials}")

    # Now run quantitative audit across ALL misses to see what recovers them:
    print("\n" + "=" * 60)
    print("QUANTITATIVE AUDIT: WHAT PASSES RECOVER THE MISSED PAIRS?")
    print("=" * 60)
    
    can_recover_shared_pin = 0
    can_recover_rare_addr_tok = 0
    can_recover_name_prefix4 = 0
    can_recover_name_char3gram = 0
    can_recover_token_sort_75 = 0
    can_recover_street_num_city = 0
    can_recover_name_token_1000 = 0
    can_recover_name_token_2000 = 0
    
    for item in missed:
        s1_id, qid, s1_r, q_r = item
        s1_name, q_name = s1_r["name_core"], q_r["name_core"]
        s1_addr, q_addr = s1_r["addr_norm"], q_r["addr_norm"]
        s1_nums = s1_r["addr_nums"].split() if s1_r["addr_nums"] else []
        q_nums_list = q_r["addr_nums"].split() if q_r["addr_nums"] else []
        c = s1_r["country"]
        
        s1_toks = set(s1_name.split())
        q_toks = set(q_name.split())
        shared_toks = s1_toks.intersection(q_toks)
        
        # 1. Higher rare name token threshold (<= 1000, <= 2000)
        if any(token_counts.get((c, t), 0) <= 1000 for t in shared_toks if len(t) >= 2):
            can_recover_name_token_1000 += 1
        if any(token_counts.get((c, t), 0) <= 2000 for t in shared_toks if len(t) >= 2):
            can_recover_name_token_2000 += 1
            
        # 2. Shared PIN (length >= 5) without name prefix restriction
        shared_pins = [p for p in s1_nums if len(p) >= 5 and p in q_nums_list]
        if shared_pins:
            can_recover_shared_pin += 1
            
        # 3. Name token prefix (first 4 chars equal)
        s1_prefixes = {t[:4] for t in s1_toks if len(t) >= 4}
        q_prefixes = {t[:4] for t in q_toks if len(t) >= 4}
        if s1_prefixes.intersection(q_prefixes):
            can_recover_name_prefix4 += 1
            
        # 4. Token sort ratio >= 75
        if fuzz.token_sort_ratio(s1_name, q_name) >= 75:
            can_recover_token_sort_75 += 1
            
        # 5. Shared address tokens (>= 2 non-numeric tokens)
        shared_addr = set(s1_addr.split()).intersection(q_addr.split())
        shared_addr_non_num = [t for t in shared_addr if not t.isdigit() and len(t) >= 4]
        if len(shared_addr_non_num) >= 2:
            can_recover_rare_addr_tok += 1
            
        # 6. Shared pair of name tokens (2-token key)
        if len(shared_toks) >= 2:
            can_recover_street_num_city += 1 # reusing counter for 2-token name key
            
    n_miss = len(missed)
    print(f"Total Misses: {n_miss:,}")
    print(f"  • Recoverable by Name Token Threshold <= 1000:   {can_recover_name_token_1000:,} ({can_recover_name_token_1000/n_miss*100:.1f}%)")
    print(f"  • Recoverable by Name Token Threshold <= 2000:   {can_recover_name_token_2000:,} ({can_recover_name_token_2000/n_miss*100:.1f}%)")
    print(f"  • Recoverable by Shared PIN alone:               {can_recover_shared_pin:,} ({can_recover_shared_pin/n_miss*100:.1f}%)")
    print(f"  • Recoverable by 4-char Name Prefix:             {can_recover_name_prefix4:,} ({can_recover_name_prefix4/n_miss*100:.1f}%)")
    print(f"  • Recoverable by Token Sort Ratio >= 75:         {can_recover_token_sort_75:,} ({can_recover_token_sort_75/n_miss*100:.1f}%)")
    print(f"  • Recoverable by Shared 2+ Addr Tokens:          {can_recover_rare_addr_tok:,} ({can_recover_rare_addr_tok/n_miss*100:.1f}%)")
    print(f"  • Recoverable by Any 2 Shared Name Tokens:       {can_recover_street_num_city:,} ({can_recover_street_num_city/n_miss*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    diagnose_misses(sample_size=5000)
