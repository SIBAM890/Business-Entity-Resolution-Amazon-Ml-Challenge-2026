"""
Test Pass 12: Rare Address Pair on the 10k Diagnostic Sample
Tests whether adding address token pairs (min count <= 2500) pushes India recall beyond 92%
"""
import time
import sys
import polars as pl
import numpy as np
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

def main():
    t0 = time.time()
    print("=" * 70)
    print("TESTING PASS 12: RARE ADDRESS PAIR ON 10K SAMPLE")
    print("=" * 70)

    # Load 10k sample
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet").filter(pl.col("matched_entity_ids") != "")
    val_sample = gt_df.sample(n=10000, seed=42)
    val_s1_set = set(val_sample["source1_entity_id"].to_list())

    y_true = {}
    s1_to_country = {}
    total_true = 0
    true_by_c = {"US": 0, "India": 0}

    for row in val_sample.iter_rows(named=True):
        matches = set(row["matched_entity_ids"].split(","))
        s1_id = row["source1_entity_id"]
        y_true[s1_id] = matches
        total_true += len(matches)

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
        if c in true_by_c: true_by_c[c] += len(y_true[r["entity_id"]])

    # Frequency counting
    addr_token_counts = Counter()
    name_token_counts = Counter()
    for row in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c, nc, an = row["country"], row["name_core"], row["addr_norm"]
        for tok in set(nc.split()): name_token_counts[(c, tok)] += 1
        for atok in set(an.split()):
            if len(atok) >= 4 and not atok.isdigit(): addr_token_counts[(c, atok)] += 1

    # Inverted index for Pass 12: Addr Token Pairs (len >= 4, min count <= 2500)
    addr_pair_idx = defaultdict(list)
    for row in s1_val.iter_rows(named=True):
        eid = row["entity_id"]
        c = row["country"]
        addr_toks = [t for t in set(row["addr_norm"].split()) if len(t) >= 4 and not t.isdigit()]
        if 2 <= len(addr_toks) <= 8:
            st = sorted(addr_toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    c1 = addr_token_counts.get((c, st[i]), 0)
                    c2 = addr_token_counts.get((c, st[j]), 0)
                    if min(c1, c2) <= 2500:
                        addr_pair_idx[(c, st[i], st[j])].append(eid)

    print(f"Address pair index built: {len(addr_pair_idx):,} unique pairs.")

    # Match queries against addr_pair_idx
    recovered_by_p12 = 0
    p12_cands_by_s1 = defaultdict(set)

    for row in queries_df.iter_rows(named=True):
        qid = row["entity_id"]
        c = row["country"]
        addr_toks = [t for t in set(row["addr_norm"].split()) if len(t) >= 4 and not t.isdigit()]
        if 2 <= len(addr_toks) <= 8:
            st = sorted(addr_toks)
            matched = set()
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    pair = (c, st[i], st[j])
                    if pair in addr_pair_idx:
                        matched.update(addr_pair_idx[pair])
            for s1_id in matched:
                p12_cands_by_s1[s1_id].add(qid)

    # Evaluate how many true pairs are caught by P12
    p12_total_hits = 0
    p12_in_hits = 0
    p12_us_hits = 0
    cand_counts = [len(p12_cands_by_s1.get(s1, set())) for s1 in val_s1_set]

    for s1_id, matches in y_true.items():
        found = p12_cands_by_s1.get(s1_id, set())
        hits = len(found.intersection(matches))
        p12_total_hits += hits
        c = s1_to_country[s1_id]
        if c == "India": p12_in_hits += hits
        elif c == "US": p12_us_hits += hits

    print("\n--- PASS 12 STANDALONE CAPTURE ---")
    print(f"  True Pairs Recovered by P12 alone: {p12_total_hits:,} / {total_true:,} ({p12_total_hits/total_true*100:.2f}%)")
    print(f"    India True Pairs:                {p12_in_hits:,} / {true_by_c['India']:,} ({p12_in_hits/true_by_c['India']*100:.2f}%)")
    print(f"    US True Pairs:                   {p12_us_hits:,} / {true_by_c['US']:,} ({p12_us_hits/true_by_c['US']*100:.2f}%)")
    print(f"  P12 Candidates / S1:               Mean={np.mean(cand_counts):.1f}, Median={np.median(cand_counts):.1f}, P95={np.percentile(cand_counts, 95):.1f}")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
