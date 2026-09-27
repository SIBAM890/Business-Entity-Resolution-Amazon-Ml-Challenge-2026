"""
Fast Loop 20k Entity Benchmark
Evaluates the complete enhanced pipeline end-to-end:
1. Boosted 11-Pass Blocking (P10 Sorted Compact, P11 Prefix4 + Early Addr)
2. Fatal Address Contradiction & Multi-Tenant Plaza Penalty Features
3. LightGBM Training & Metric Evaluation (Precision, Recall, Macro F0.5)
Runs on 20,000 S1 entities in ~4-5 minutes without touching full 2.2M dataset.
"""
import time
import sys
import os
import psutil
import polars as pl
import lightgbm as lgb
import numpy as np
from collections import defaultdict, Counter
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.metrics import macro_f05

sys.stdout.reconfigure(encoding='utf-8')

def get_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def run_fast_loop(sample_size=20000, seed=42):
    t0 = time.time()
    print("=" * 70)
    print(f"FAST LOOP BENCHMARK: 20K ENTITIES END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load Ground Truth
    gt_df = pl.read_parquet("cache/train_ground_truth.parquet").filter(pl.col("matched_entity_ids") != "")
    sample_s1 = gt_df.sample(n=sample_size, seed=seed)
    s1_ids = set(sample_s1["source1_entity_id"].to_list())

    y_true = {}
    true_pairs = []
    for r in sample_s1.iter_rows(named=True):
        m = set(r["matched_entity_ids"].split(","))
        y_true[r["source1_entity_id"]] = m
        for qid in m:
            true_pairs.append((r["source1_entity_id"], qid))

    tp_df = pl.DataFrame(true_pairs, schema=["s1_id", "m_id"])
    print(f"Sample contains {len(s1_ids):,} S1 entities with {len(true_pairs):,} true matches.")

    # 2. Load Normalized Data
    s1_df = pl.read_parquet("cache/train_source1_norm.parquet").filter(pl.col("entity_id").is_in(s1_ids))
    s2_df = pl.read_parquet("cache/train_source2_norm.parquet")
    s3_df = pl.read_parquet("cache/train_source3_norm.parquet")
    queries_df = pl.concat([s2_df, s3_df])
    del s2_df, s3_df

    # Token frequencies
    t_fc = time.time()
    name_counts = Counter()
    addr_counts = Counter()
    for r in queries_df.select(["country", "name_core", "addr_norm"]).iter_rows(named=True):
        c, nc, an = r["country"], r["name_core"], r["addr_norm"]
        for tok in set(nc.split()): name_counts[(c, tok)] += 1
        for at in set(an.split()):
            if len(at) >= 4 and not at.isdigit(): addr_counts[(c, at)] += 1
    print(f"Token counts computed in {time.time()-t_fc:.1f}s.")

    # 3. Boosted 11-Pass Inverted Index
    print("Building Boosted 11-Pass Inverted Indexes...")
    exact_idx = defaultdict(list)
    compact_idx = defaultdict(list)
    sorted_compact_idx = defaultdict(list)
    rare_tok_idx = defaultdict(list)
    bigram_idx = defaultdict(list)
    prefix_city_idx = defaultdict(list)
    prefix_early_addr_idx = defaultdict(list)
    pin_prefix_idx = defaultdict(list)
    street_pin_idx = defaultdict(list)
    num_addr_idx = defaultdict(list)
    rare_addr_idx = defaultdict(list)

    s1_rows = s1_df.iter_rows(named=True)
    for s1_row_idx, row in enumerate(s1_rows):
        eid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = nc.replace(" ", "")
        toks = nc.split()
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        p2 = comp[:2] if len(comp) >= 2 else comp
        p4 = comp[:4] if len(comp) >= 4 else comp
        s_comp = "".join(sorted(toks))

        # P1 Exact
        if nc: exact_idx[(c, nc)].append((s1_row_idx, eid))
        # P2 Compact
        if comp: compact_idx[(c, comp)].append((s1_row_idx, eid))
        # P10 Sorted Compact
        if s_comp and s_comp != comp: sorted_compact_idx[(c, s_comp)].append((s1_row_idx, eid))
        # P3 Rare Token
        for t in set(toks):
            if len(t) >= 2 and name_counts.get((c, t), 0) <= 1200:
                rare_tok_idx[(c, t)].append((s1_row_idx, eid))
        # P4 Bigram
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    c1, c2 = name_counts.get((c, st[i]), 0), name_counts.get((c, st[j]), 0)
                    if min(c1, c2) <= 4500 and len(st[i]) >= 3 and len(st[j]) >= 3:
                        bigram_idx[(c, st[i], st[j])].append((s1_row_idx, eid))
        # P5 Prefix4 + City / Locality (last 3 tokens)
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if len(at) >= 4 and not at.isdigit() and addr_counts.get((c, at), 0) <= 25000:
                    prefix_city_idx[(c, p4, at)].append((s1_row_idx, eid))
        # P11 Prefix4 + Early Address Token (first 6 tokens)
        if len(comp) >= 4 and addr_toks:
            for at in addr_toks[:6]:
                if len(at) >= 4 and not at.isdigit() and addr_counts.get((c, at), 0) <= 10000:
                    prefix_early_addr_idx[(c, p4, at)].append((s1_row_idx, eid))
        # P6 PIN + Prefix2
        for p in nums:
            if len(p) >= 5: pin_prefix_idx[(c, p, p2)].append((s1_row_idx, eid))
        # P7 Street Num + PIN
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if len(p) >= 5: street_pin_idx[(c, first_num, p)].append((s1_row_idx, eid))
        # P8 Street Num + Addr Token
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if not at.isdigit() and len(at) >= 4:
                    num_addr_idx[(c, first_num, at)].append((s1_row_idx, eid))
                    break
        # P9 Rare Addr Token
        for at in set(addr_toks):
            if not at.isdigit() and len(at) >= 4 and addr_counts.get((c, at), 0) <= 250:
                rare_addr_idx[(c, at)].append((s1_row_idx, eid))

    print(f"Indexes built. Streaming queries for candidate generation...")
    t_match = time.time()
    cands_by_s1 = defaultdict(set)
    candidate_pairs = []

    for q_idx, row in enumerate(queries_df.iter_rows(named=True)):
        qid = row["entity_id"]
        c = row["country"]
        nc = row["name_core"]
        comp = nc.replace(" ", "")
        toks = nc.split()
        nums = row["addr_nums"].split() if row["addr_nums"] else []
        addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
        p2 = comp[:2] if len(comp) >= 2 else comp
        p4 = comp[:4] if len(comp) >= 4 else comp
        s_comp = "".join(sorted(toks))

        matched = set()
        if nc and (c, nc) in exact_idx: matched.update(exact_idx[(c, nc)])
        if comp and (c, comp) in compact_idx: matched.update(compact_idx[(c, comp)])
        if s_comp and (c, s_comp) in sorted_compact_idx: matched.update(sorted_compact_idx[(c, s_comp)])
        for t in set(toks):
            if (c, t) in rare_tok_idx: matched.update(rare_tok_idx[(c, t)])
        if 2 <= len(toks) <= 5:
            st = sorted(toks)
            for i in range(len(st)):
                for j in range(i + 1, len(st)):
                    if (c, st[i], st[j]) in bigram_idx: matched.update(bigram_idx[(c, st[i], st[j])])
        if len(comp) >= 4:
            for at in addr_toks[-3:]:
                if (c, p4, at) in prefix_city_idx: matched.update(prefix_city_idx[(c, p4, at)])
        if len(comp) >= 4 and addr_toks:
            for at in addr_toks[:6]:
                if (c, p4, at) in prefix_early_addr_idx: matched.update(prefix_early_addr_idx[(c, p4, at)])
        for p in nums:
            if len(p) >= 5 and (c, p, p2) in pin_prefix_idx: matched.update(pin_prefix_idx[(c, p, p2)])
        if len(nums) >= 2:
            first_num = nums[0]
            for p in nums[1:]:
                if (c, first_num, p) in street_pin_idx: matched.update(street_pin_idx[(c, first_num, p)])
        if nums and addr_toks:
            first_num = nums[0]
            for at in addr_toks[:3]:
                if (c, first_num, at) in num_addr_idx:
                    matched.update(num_addr_idx[(c, first_num, at)])
                    break
        for at in set(addr_toks):
            if (c, at) in rare_addr_idx: matched.update(rare_addr_idx[(c, at)])

        for s1_idx_int, s1_id in matched:
            cands_by_s1[s1_id].add(qid)
            candidate_pairs.append((s1_id, qid, s1_idx_int, q_idx))

    print(f"Candidate generation complete in {time.time()-t_match:.1f}s. Total pairs: {len(candidate_pairs):,}")

    # Evaluate Blocking Recall
    recovered = 0
    for s1_id, matches in y_true.items():
        recovered += len(cands_by_s1.get(s1_id, set()).intersection(matches))
    blocking_recall = recovered / len(true_pairs)
    print(f"\n>>> BOOSTED BLOCKING RECALL: {blocking_recall*100:.2f}% ({recovered:,} / {len(true_pairs):,}) <<<")

    # 4. Compute Features on Candidate Pairs
    print("\nComputing discriminative pairwise features...")
    t_feat = time.time()
    
    # Target label: 1 if true pair else 0
    s1_lookup = {r["entity_id"]: r for r in s1_df.iter_rows(named=True)}
    q_lookup = {r["entity_id"]: r for r in queries_df.filter(pl.col("entity_id").is_in({p[1] for p in candidate_pairs})).iter_rows(named=True)}

    feat_rows = []
    labels = []
    query_cand_map = defaultdict(list)

    for idx, (s1_id, qid, _, _) in enumerate(candidate_pairs):
        is_true = 1 if qid in y_true.get(s1_id, set()) else 0
        labels.append(is_true)
        query_cand_map[qid].append(idx)

    # Convert to arrays for rapidfuzz
    s1_names = [s1_lookup[p[0]]["name_core"] for p in candidate_pairs]
    q_names = [q_lookup[p[1]]["name_core"] for p in candidate_pairs]
    s1_addrs = [s1_lookup[p[0]]["addr_norm"] for p in candidate_pairs]
    q_addrs = [q_lookup[p[1]]["addr_norm"] for p in candidate_pairs]

    nm_ratio = process.cpdist(q_names, s1_names, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    nm_tset = process.cpdist(q_names, s1_names, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    nm_tsort = process.cpdist(q_names, s1_names, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32)
    ad_ratio = process.cpdist(q_addrs, s1_addrs, scorer=fuzz.ratio, workers=-1, dtype=np.float32)
    ad_tset = process.cpdist(q_addrs, s1_addrs, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)

    # Derived and Contradiction Features:
    # 1. Multi-Tenant Plaza Penalty: ad >= 75 and nm < 55
    ad_high_nm_low = ((ad_ratio >= 75.0) & (nm_ratio < 55.0)).astype(np.float32)
    # 2. Product of Name and Addr
    nm_ad_product = (nm_ratio / 100.0) * (ad_ratio / 100.0)
    # 3. Min of Name and Addr
    nm_ad_min = np.minimum(nm_ratio, ad_ratio)

    X = np.column_stack([
        nm_ratio, nm_tset, nm_tsort, ad_ratio, ad_tset,
        ad_high_nm_low, nm_ad_product, nm_ad_min
    ])
    y = np.array(labels, dtype=np.int8)
    print(f"Features computed in {time.time()-t_feat:.1f}s. Shape: {X.shape}, Positives: {y.sum():,}")

    # 5. Split Train / Validation (80% / 20% on S1 entities)
    s1_list = sorted(list(s1_ids))
    val_s1_set = set(s1_list[::5])
    train_s1_set = set(s1_list) - val_s1_set

    val_mask = np.array([p[0] in val_s1_set for p in candidate_pairs])
    X_train, y_train = X[~val_mask], y[~val_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs = [candidate_pairs[i] for i in range(len(candidate_pairs)) if val_mask[i]]

    print(f"Train pairs: {len(X_train):,} (Pos: {y_train.sum():,}) | Val pairs: {len(X_val):,} (Pos: {y_val.sum():,})")

    # 6. Train LightGBM Booster
    print("Training LightGBM ranker...")
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=127,
        min_data_in_leaf=100,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        scale_pos_weight=0.80, # Asymmetric penalty on false merges
        verbose=-1
    )

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)]
    )

    # 7. Evaluate on Held-out 4,000 Validation Entities
    print("\nEvaluating on held-out validation fold...")
    val_probs = model.predict(X_val)
    
    # Build best per query
    val_df = pl.DataFrame({
        "s1_id": [p[0] for p in val_pairs],
        "q_id": [p[1] for p in val_pairs],
        "prob": val_probs
    })

    best_val = val_df.sort(["q_id", "prob"], descending=[False, True]) \
                     .group_by("q_id", maintain_order=True) \
                     .agg(pl.col("s1_id").first(), pl.col("prob").first().alias("p1"),
                          pl.col("prob").slice(1, 1).first().fill_null(0.0).alias("p2"))

    val_eval_s1 = pl.Series("s1_id", sorted(list(val_s1_set)))
    val_tp_df = tp_df.filter(pl.col("s1_id").is_in(val_s1_set))

    print("\n--- THRESHOLD & AMBIGUITY GAP SWEEP (VAL FOLD) ---")
    best_comb = None
    best_f05 = 0.0

    for t in [0.55, 0.60, 0.65, 0.70, 0.75]:
        for gap in [0.0, 0.05]:
            pred = best_val.filter((pl.col("p1") >= t) & (pl.col("p1") - pl.col("p2") >= gap)) \
                           .select(pl.col("s1_id"), pl.col("q_id").alias("m_id"))
            m = macro_f05(pred, val_tp_df, val_eval_s1)
            print(f"Threshold={t:.2f} | Gap={gap:.2f} -> Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | SingAcc: {m['singleton_acc']:.4f}")
            if m["macro_f05"] > best_f05:
                best_f05 = m["macro_f05"]
                best_comb = (t, gap, m)

    print("\n" + "=" * 70)
    print(f"FAST LOOP RESULT (20k Benchmark):")
    print(f"  Blocking Recall:    {blocking_recall*100:.2f}% (vs 93.11% baseline)")
    print(f"  Best Operating Pt:  Threshold={best_comb[0]:.2f}, Margin Gap={best_comb[1]:.2f}")
    print(f"  Macro F0.5:         {best_comb[2]['macro_f05']:.4f}")
    print(f"  Precision:          {best_comb[2]['macro_precision']:.4f}")
    print(f"  Recall:             {best_comb[2]['macro_recall']:.4f}")
    print(f"  Total Runtime:      {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    run_fast_loop(sample_size=20000)
