"""
Fast Loop: 10k Diagnostic Sample Blocking Benchmark
Tests complementary passes to push candidate blocking recall from ~93.1% to >= 95.5%
Measures:
- Overall recall
- US recall vs India recall
- Total candidate volume, mean/S1, median/S1, p95/S1
- Pass-by-pass incremental contribution
"""
import time
import sys
import os
import psutil
import polars as pl
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_benchmark(sample_size=10000, seed=42):
    t0 = time.time()
    print("=" * 70)
    print(f"FAST LOOP BLOCKING BENCHMARK (Sample: {sample_size:,} S1 entities)")
    print("=" * 70)

    # 1. Load Ground Truth
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet")
    gt_with_matches = gt_df.filter(pl.col("matched_entity_ids") != "")
    val_sample = gt_with_matches.sample(n=sample_size, seed=seed)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())

    y_true = {}
    s1_to_country = {}
    total_true = 0
    true_by_country = {"US": 0, "India": 0}

    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        s1_id = row["source1_entity_id"]
        y_true[s1_id] = matches
        total_true += len(matches)

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

    for r in s1_val.select(["entity_id", "country"]).iter_rows(named=True):
        s1_to_country[r["entity_id"]] = r["country"]
        c = r["country"]
        if c in true_by_country:
            true_by_country[c] += len(y_true[r["entity_id"]])

    print(f"Sample has {len(val_s1_set):,} S1 entities ({s1_val.filter(pl.col('country')=='US').height} US, {s1_val.filter(pl.col('country')=='India').height} India)")
    print(f"Total True Pairs: {total_true:,} (US: {true_by_country['US']:,}, India: {true_by_country['India']:,})")

    # Add compact and sorted-tokens compact using native Polars
    s1_val = s1_val.with_columns([
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact"),
        pl.col("name_core").str.split(" ").list.sort().list.join("").alias("name_sorted_compact")
    ])
    queries_df = queries_df.with_columns([
        pl.col("name_core").str.replace_all(" ", "").alias("name_compact"),
        pl.col("name_core").str.split(" ").list.sort().list.join("").alias("name_sorted_compact")
    ])

    # 3. Frequency Counting
    t_fc = time.time()
    name_token_counts = Counter()
    addr_token_counts = Counter()
    for row in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c, nc, an = row["country"], row["name_core"], row["addr_norm"]
        for tok in set(nc.split()):
            name_token_counts[(c, tok)] += 1
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit():
                addr_token_counts[(c, atok)] += 1
    print(f"Frequency counting complete in {time.time()-t_fc:.1f}s. RAM: {get_ram_mb():.1f} MB")

    # 4. Build Multi-Pass Inverted Indexes on S1
    # We will test:
    # Baseline: Passes 1 to 9
    # Boosted: + Pass 10 (Sorted Compact Name), + Pass 11 (Prefix4 + Mid/Full Addr Token), + Pass 12 (Expanded Bigrams)
    print("Building Inverted Indexes on S1...")
    exact_idx = defaultdict(list)
    compact_idx = defaultdict(list)
    sorted_compact_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    name_bigram_idx = defaultdict(list)
    prefix_city_idx = defaultdict(list)
    prefix_any_addr_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)
    street_pin_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    rare_addr_idx = defaultdict(list)

    RARE_NAME_THRESH = 2200
    RARE_ADDR_THRESH = 350

    for row in s1_val.iter_rows(named=True):
        eid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = row["name_compact"]
        s_comp = row["name_sorted_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        p2 = comp[:2] if len(comp) >= 2 else comp
        p4 = comp[:4] if len(comp) >= 4 else comp

        # P1: Exact
        if nc: exact_idx[(c, nc)].append(eid)
        # P2: Compact
        if comp: compact_idx[(c, comp)].append(eid)
        # P10 (Boosted): Sorted Compact (reverses word order: "diaz herman" == "herman diaz")
        if s_comp and s_comp != comp: sorted_compact_idx[(c, s_comp)].append(eid)

        # P3: Rare token
        for t in set(toks):
            if len(t) >= 2 and name_token_counts.get((c, t), 0) <= RARE_NAME_THRESH:
                rare_tok_idx[(c, t)].append(eid)

        # P4: Name Bigrams (min count <= 5500)
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    c1 = name_token_counts.get((c, st[i]), 0)
                    c2 = name_token_counts.get((c, st[j]), 0)
                    if min(c1, c2) <= 5500 and len(st[i]) >= 3 and len(st[j]) >= 3:
                        name_bigram_idx[(c, st[i], st[j])].append(eid)

        # P5: Prefix4 + City / Locality (last 3 tokens)
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if len(at) >= 4 and not at.isdigit() and addr_token_counts.get((c, at), 0) <= 25000:
                    prefix_city_idx[(c, p4, at)].append(eid)

        # P11 (Boosted): Prefix4 + Any Addr Token in first 6 tokens (count <= 10000)
        if len(comp) >= 4 and addr_toks:
            for at in addr_toks[:6]:
                if len(at) >= 4 and not at.isdigit() and addr_token_counts.get((c, at), 0) <= 10000:
                    prefix_any_addr_idx[(c, p4, at)].append(eid)

        # P6: PIN + Prefix2
        for p in nums:
            if len(p) >= 5: pin_prefix_idx[(c, p, p2)].append(eid)

        # P7: Street Num + PIN
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if len(p) >= 5: street_pin_idx[(c, first_num, p)].append(eid)

        # P8: Street Num + Addr Token
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if not at.isdigit() and len(at) >= 4:
                    num_addr_idx[(c, first_num, at)].append(eid)
                    break

        # P9: Rare Addr Token
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and addr_token_counts.get((c, at), 0) <= RARE_ADDR_THRESH:
                rare_addr_idx[(c, at)].append(eid)

    # 5. Stream queries and match against indexes
    print("Streaming queries against S1 indexes...")
    t_match = time.time()
    
    # Store candidates separately for baseline vs boosted
    cands_base = defaultdict(set)
    cands_boosted = defaultdict(set)
    pass_hits = Counter()

    for row in queries_df.iter_rows(named=True):
        qid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = row["name_compact"]
        s_comp = row["name_sorted_compact"]
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        toks = nc.split()
        p2 = comp[:2] if len(comp) >= 2 else comp
        p4 = comp[:4] if len(comp) >= 4 else comp

        m_base = set()
        m_extra = set()

        # P1
        if nc and (c, nc) in exact_idx:
            m_base.update(exact_idx[(c, nc)])
        # P2
        if comp and (c, comp) in compact_idx:
            m_base.update(compact_idx[(c, comp)])
        # P3
        for t in set(toks):
            if (c, t) in rare_tok_idx: m_base.update(rare_tok_idx[(c, t)])
        # P4
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    if (c, st[i], st[j]) in name_bigram_idx:
                        m_base.update(name_bigram_idx[(c, st[i], st[j])])
        # P5
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if (c, p4, at) in prefix_city_idx:
                    m_base.update(prefix_city_idx[(c, p4, at)])
        # P6
        for p in nums:
            if len(p) >= 5 and (c, p, p2) in pin_prefix_idx:
                m_base.update(pin_prefix_idx[(c, p, p2)])
        # P7
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if (c, first_num, p) in street_pin_idx:
                    m_base.update(street_pin_idx[(c, first_num, p)])
        # P8
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:4]:
                if (c, first_num, at) in num_addr_idx:
                    m_base.update(num_addr_idx[(c, first_num, at)])
                    break
        # P9
        for at in set(addr_toks):
            if (c, at) in rare_addr_idx:
                m_base.update(rare_addr_idx[(c, at)])

        # BOOSTED EXTRA PASSES:
        # P10: Sorted Compact Name
        if s_comp and (c, s_comp) in sorted_compact_idx:
            hit = sorted_compact_idx[(c, s_comp)]
            m_extra.update(hit)
            pass_hits["P10_Sorted_Compact"] += len(hit)
        # P11: Prefix4 + Mid/Early Addr Token
        if len(comp) >= 4 and addr_toks:
            for at in addr_toks[:6]:
                if (c, p4, at) in prefix_any_addr_idx:
                    hit = prefix_any_addr_idx[(c, p4, at)]
                    m_extra.update(hit)
                    pass_hits["P11_Prefix4_AnyAddr"] += len(hit)

        # Record candidates
        for s1_id in m_base:
            cands_base[s1_id].add(qid)
        for s1_id in (m_base | m_extra):
            cands_boosted[s1_id].add(qid)

    print(f"Matching finished in {time.time()-t_match:.1f}s. RAM: {get_ram_mb():.1f} MB")

    # 6. Evaluate Baseline vs Boosted
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON: BASELINE (9 PASSES) VS BOOSTED (11 PASSES)")
    print("=" * 70)

    for mode, c_dict in [("BASELINE (9 Passes)", cands_base), ("BOOSTED (11 Passes)", cands_boosted)]:
        rec_total = 0
        rec_us = 0
        rec_in = 0
        c_counts = []

        for s1_id, matches in y_true.items():
            f = c_dict.get(s1_id, set())
            hits = len(f.intersection(matches))
            rec_total += hits
            c = s1_to_country[s1_id]
            if c == "US": rec_us += hits
            elif c == "India": rec_in += hits
            c_counts.append(len(f))

        c_arr = np.array(c_counts)
        r_tot = rec_total / total_true
        r_us = rec_us / true_by_country["US"] if true_by_country["US"] > 0 else 0
        r_in = rec_in / true_by_country["India"] if true_by_country["India"] > 0 else 0

        print(f"\n--- {mode} ---")
        print(f"  Overall Blocking Recall:   {r_tot*100:.2f}% ({rec_total:,} / {total_true:,})")
        print(f"    US Blocking Recall:      {r_us*100:.2f}% ({rec_us:,} / {true_by_country['US']:,})")
        print(f"    India Blocking Recall:   {r_in*100:.2f}% ({rec_in:,} / {true_by_country['India']:,})")
        print(f"  Candidate Statistics:")
        print(f"    Total Candidates:        {c_arr.sum():,}")
        print(f"    Mean Candidates / S1:    {c_arr.mean():.1f}")
        print(f"    Median Candidates / S1:  {np.median(c_arr):.1f}")
        print(f"    P95 Candidates / S1:     {np.percentile(c_arr, 95):.1f}")
        print(f"    Max Candidates / S1:     {c_arr.max():,}")
        print(f"    S1 with 0 Candidates:    {(c_arr == 0).sum():,} ({(c_arr == 0).mean()*100:.2f}%)")

    print("\nBoosted Pass Hits:")
    for k, v in pass_hits.items():
        print(f"  {k}: {v:,} hits")

    print("\n" + "=" * 70)
    print(f"Total benchmark time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    run_benchmark(sample_size=10000)
