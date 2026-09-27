"""
Fast Experiment: Asymmetric Loss & Precision-Weighted LightGBM
Trains an updated LightGBM model on the existing 9.27M feature rows in cache/train_feats/
with asymmetric weighting to align directly with Macro F0.5 (penalizing false merges).
Evaluates on the full 441,655 entity validation fold.
"""
import time
import sys
import os
import polars as pl
import lightgbm as lgb
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.entity_resolution.config import cache_path
from src.entity_resolution.metrics import macro_f05
from src.entity_resolution.train import (
    FEATURES, SEED, USE_GPU, label_table, score_split, best_per_query
)

sys.stdout.reconfigure(encoding='utf-8')

def main(scale_pos_weight=0.75):
    t0 = time.time()
    print("=" * 70)
    print(f"FAST EXPERIMENT: ASYMMETRIC LIGHTGBM (scale_pos_weight={scale_pos_weight})")
    print("=" * 70)

    # 1. Load label tables
    s1, q, lab, tp = label_table()
    val_s1 = s1.filter(pl.col("is_val"))
    print(f"Loaded metadata. Validation entities: {val_s1.height:,} ({time.time()-t0:.1f}s)")

    # 2. Load cached training features
    print("Loading cached train_feats...")
    t_load = time.time()
    feat_paths = sorted(cache_path("train_feats").glob("part_*.parquet"))
    dfs = [pl.read_parquet(p) for p in feat_paths]
    df = pl.concat(dfs)
    del dfs
    print(f"Loaded {df.height:,} candidate pairs ({time.time()-t_load:.1f}s)")

    # 3. Join labels and split train / val
    df = df.join(lab, on="q_idx", how="left")
    y = (pl.col("true_s1_idx").is_not_null() & (pl.col("true_s1_idx") == pl.col("s1_idx"))).cast(pl.Int8)
    df = df.with_columns(y.alias("target"))

    # Map validation flag from s1
    val_s1_indices = set(val_s1["s1_idx"].to_list())
    is_val_mask = df["s1_idx"].is_in(val_s1_indices).to_numpy()
    val_ids = df.select("q_idx", "s1_idx").filter(is_val_mask)

    X = df.select(FEATURES).to_numpy()
    y_arr = df["target"].to_numpy()
    del df

    X_train, y_train = X[~is_val_mask], y_arr[~is_val_mask]
    X_val, y_val = X[is_val_mask], y_arr[is_val_mask]
    del X, y_arr
    print(f"Train split: {len(X_train):,} pairs (Pos: {y_train.sum():,}) | Val split: {len(X_val):,} pairs (Pos: {y_val.sum():,})")

    # 4. Train LightGBM with asymmetric weight
    params = dict(
        objective="binary",
        learning_rate=0.05,
        num_leaves=255,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        max_bin=255,
        num_threads=16,
        seed=SEED,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
        **({"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0} if USE_GPU else {"device": "cpu"})
    )

    print(f"Training LightGBM with scale_pos_weight={scale_pos_weight}...")
    t_tr = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)
    del X_train, y_train

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=200)]
    )
    print(f"Training complete in {time.time()-t_tr:.1f}s. Best iteration: {model.best_iteration}")

    # 5. Score full validation split and evaluate
    print("Scoring validation split...")
    val_probs = model.predict(X_val, num_iteration=model.best_iteration)
    del X_val, y_val, dtrain, dval

    # Re-evaluate on validation fold
    df_scored = val_ids.with_columns(pl.Series("prob", val_probs))
    del val_probs, val_ids

    # Best per query
    best_val = best_per_query(df_scored)
    del df_scored

    # Join metadata
    ids = best_val.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
                  .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")

    print("\n--- THRESHOLD SWEEP ON ASYMMETRIC MODEL ---")
    best_res = None
    best_f05 = 0.0
    best_t = 0.50

    for t in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        pred = ids.filter(pl.col("p1") >= t).select("s1_id", "m_id")
        m = macro_f05(pred, tp, val_s1["entity_id"])
        m_in = macro_f05(pred, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])
        m_us = macro_f05(pred, tp, val_s1.filter(pl.col("country") == "US")["entity_id"])
        print(f"Threshold {t:.2f} -> Macro F0.5: {m['macro_f05']:.4f} | Prec: {m['macro_precision']:.4f} | Rec: {m['macro_recall']:.4f} | SingAcc: {m['singleton_acc']:.4f} | IN: {m_in['macro_f05']:.4f} | US: {m_us['macro_f05']:.4f}")
        if m["macro_f05"] > best_f05:
            best_f05 = m["macro_f05"]
            best_t = t
            best_res = m

    print("\n" + "=" * 70)
    print(f"BEST RESULT (scale_pos_weight={scale_pos_weight}): Threshold={best_t:.2f} -> Macro F0.5={best_f05:.4f}")
    print(f"Total experiment time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == "__main__":
    spw = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
    main(spw)
