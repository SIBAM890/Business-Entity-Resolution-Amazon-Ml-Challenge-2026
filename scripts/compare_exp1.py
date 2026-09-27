import polars as pl
from src.entity_resolution.metrics import macro_f05
from src.entity_resolution.train import label_table

s1, q, lab, tp = label_table()
val_s1 = s1.filter(pl.col("is_val"))
best = pl.read_parquet("cache/train_best.parquet")

ids = best.join(s1.select("s1_idx", pl.col("entity_id").alias("s1_id"), "country"), on="s1_idx") \
          .join(q.select("q_idx", pl.col("entity_id").alias("m_id")), on="q_idx")

p_base = ids.filter(pl.col("p1") >= 0.65).select("s1_id", "m_id")
p_tuned = ids.filter(
    (pl.col("p1") - pl.col("p2") >= 0.05) & 
    (
        ((pl.col("country") == "India") & (pl.col("p1") >= 0.68)) | 
        ((pl.col("country") == "US") & (pl.col("p1") >= 0.65))
    )
).select("s1_id", "m_id")

r_base = macro_f05(p_base, tp, val_s1["entity_id"])
r_tuned = macro_f05(p_tuned, tp, val_s1["entity_id"])

print("--- EXPERIMENT 1 COMPARISON ---")
print(f"Base (0.65):      F0.5={r_base['macro_f05']:.5f} | Prec={r_base['macro_precision']:.5f} | Rec={r_base['macro_recall']:.5f} | SingAcc={r_base['singleton_acc']:.5f}")
print(f"Tuned (Adaptive): F0.5={r_tuned['macro_f05']:.5f} | Prec={r_tuned['macro_precision']:.5f} | Rec={r_tuned['macro_recall']:.5f} | SingAcc={r_tuned['singleton_acc']:.5f}")

# India specific
r_base_in = macro_f05(p_base, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])
r_tuned_in = macro_f05(p_tuned, tp, val_s1.filter(pl.col("country") == "India")["entity_id"])
print(f"\nIndia Base:       F0.5={r_base_in['macro_f05']:.5f} | Prec={r_base_in['macro_precision']:.5f} | Rec={r_base_in['macro_recall']:.5f} | SingAcc={r_base_in['singleton_acc']:.5f}")
print(f"India Tuned:      F0.5={r_tuned_in['macro_f05']:.5f} | Prec={r_tuned_in['macro_precision']:.5f} | Rec={r_tuned_in['macro_recall']:.5f} | SingAcc={r_tuned_in['singleton_acc']:.5f}")
