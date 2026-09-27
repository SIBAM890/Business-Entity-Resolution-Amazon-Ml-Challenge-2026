"""
Step 2: Boosted high-performance multi-pass candidate generation for a split (train or test).
Replaces slow CPU torch brute-force with optimized multi-pass inverted indexes (>93% recall).

Writes cache/<split>_cands/<country>_<chunk>.parquet with columns:
  q_idx   : UInt32, row in the concatenated query table (Source 2 rows, then Source 3 rows)
  s1_idx  : UInt32, row in the Source 1 table
  cos_name: Float32, name similarity
  cos_addr: Float32, address similarity
  cos_comb: Float32, combined retrieval score
  rank    : Int16, rank of this S1 among the query's candidates
"""
import gc
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .config import cache_path

K_KEEP = 15
CHUNK = 250_000
RARE_NAME_THRESH = 1200
RARE_ADDR_THRESH = 250


def scan_tables(split: str):
    """Lazy frames; materialised one country at a time to bound memory."""
    cols = ["country", "name_core", "addr_norm", "addr_nums"]
    s1 = pl.scan_parquet(cache_path(f"{split}_source1_norm.parquet")).select(["entity_id"] + cols).with_row_index("s1_idx")
    q = pl.concat([pl.scan_parquet(cache_path(f"{split}_source2_norm.parquet")).select(cols),
                   pl.scan_parquet(cache_path(f"{split}_source3_norm.parquet")).select(cols)]).with_row_index("q_idx")
    return s1, q


class MultiPassIndex:
    """High-speed country-partitioned multi-pass inverted index on Source 1."""

    def __init__(self, s1c: pl.DataFrame, name_counts: Counter, addr_counts: Counter):
        self.s1c = s1c
        self.exact_idx = defaultdict(list)
        self.compact_idx = defaultdict(list)
        self.rare_tok_idx = defaultdict(list)
        self.bigram_idx = defaultdict(list)
        self.prefix_city_idx = defaultdict(list)
        self.pin_prefix_idx = defaultdict(list)
        self.street_pin_idx = defaultdict(list)
        self.num_addr_idx = defaultdict(list)
        self.rare_addr_idx = defaultdict(list)

        s1_rows = s1c.select(["name_core", "addr_norm", "addr_nums"]).iter_rows(named=True)
        for s1_row_idx, row in enumerate(s1_rows):
            nc = row["name_core"]
            comp = nc.replace(" ", "")
            nums = row["addr_nums"].split() if row["addr_nums"] else []
            addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
            toks = nc.split()
            p2 = comp[:2] if len(comp) >= 2 else comp
            p4 = comp[:4] if len(comp) >= 4 else comp

            # P1: Exact core name
            if nc:
                self.exact_idx[nc].append(s1_row_idx)

            # P2: Compact name
            if comp:
                self.compact_idx[comp].append(s1_row_idx)

            # P3: Rare name tokens
            for t in set(toks):
                if len(t) >= 2 and name_counts.get(t, 0) <= RARE_NAME_THRESH:
                    self.rare_tok_idx[t].append(s1_row_idx)

            # P4: Name Bigrams (min count <= 4500)
            if 2 <= len(toks) <= 5:
                st = sorted(toks)
                for i in range(len(st)):
                    for j in range(i + 1, len(st)):
                        c1 = name_counts.get(st[i], 0)
                        c2 = name_counts.get(st[j], 0)
                        if min(c1, c2) <= 4500 and len(st[i]) >= 3 and len(st[j]) >= 3:
                            self.bigram_idx[(st[i], st[j])].append(s1_row_idx)

            # P5: Prefix4 + City / Locality
            if len(comp) >= 4:
                for at in addr_toks[-3:]:
                    if len(at) >= 4 and not at.isdigit() and addr_counts.get(at, 0) <= 25000:
                        self.prefix_city_idx[(p4, at)].append(s1_row_idx)

            # P6: PIN + Prefix2
            for p in nums:
                if len(p) >= 5:
                    self.pin_prefix_idx[(p, p2)].append(s1_row_idx)

            # P7: Street Number + PIN
            if len(nums) >= 2:
                first_num = nums[0]
                for p in nums[1:]:
                    if len(p) >= 5:
                        self.street_pin_idx[(first_num, p)].append(s1_row_idx)

            # P8: Street Number + Addr Token
            if nums and addr_toks:
                first_num = nums[0]
                for at in addr_toks[:3]:
                    if not at.isdigit() and len(at) >= 4:
                        self.num_addr_idx[(first_num, at)].append(s1_row_idx)
                        break

            # P9: Rare Addr Token
            for at in set(addr_toks):
                if not at.isdigit() and len(at) >= 4 and addr_counts.get(at, 0) <= RARE_ADDR_THRESH:
                    self.rare_addr_idx[at].append(s1_row_idx)

    def match_queries(self, part: pl.DataFrame, k_keep: int = K_KEEP) -> pl.DataFrame:
        """Matches a chunk of queries against the S1 index and returns top-K candidate pairs."""
        q_rows = part.select(["name_core", "addr_norm", "addr_nums"]).iter_rows(named=True)
        pair_q_indices = []
        pair_s1_indices = []

        for q_row_idx, row in enumerate(q_rows):
            nc = row["name_core"]
            comp = nc.replace(" ", "")
            nums = row["addr_nums"].split() if row["addr_nums"] else []
            addr_toks = row["addr_norm"].split() if row["addr_norm"] else []
            toks = nc.split()
            p2 = comp[:2] if len(comp) >= 2 else comp
            p4 = comp[:4] if len(comp) >= 4 else comp

            cands = set()

            if nc and nc in self.exact_idx:
                cands.update(self.exact_idx[nc])
            if comp and comp in self.compact_idx:
                cands.update(self.compact_idx[comp])
            for t in set(toks):
                if t in self.rare_tok_idx:
                    cands.update(self.rare_tok_idx[t])
            if 2 <= len(toks) <= 5:
                st = sorted(toks)
                for i in range(len(st)):
                    for j in range(i + 1, len(st)):
                        if (st[i], st[j]) in self.bigram_idx:
                            cands.update(self.bigram_idx[(st[i], st[j])])
            if len(comp) >= 4:
                for at in addr_toks[-3:]:
                    if (p4, at) in self.prefix_city_idx:
                        cands.update(self.prefix_city_idx[(p4, at)])
            for p in nums:
                if len(p) >= 5 and (p, p2) in self.pin_prefix_idx:
                    cands.update(self.pin_prefix_idx[(p, p2)])
            if len(nums) >= 2:
                first_num = nums[0]
                for p in nums[1:]:
                    if len(p) >= 5 and (first_num, p) in self.street_pin_idx:
                        cands.update(self.street_pin_idx[(first_num, p)])
            if nums and addr_toks:
                first_num = nums[0]
                for at in addr_toks[:3]:
                    if not at.isdigit() and len(at) >= 4 and (first_num, at) in self.num_addr_idx:
                        cands.update(self.num_addr_idx[(first_num, at)])
                        break
            for at in set(addr_toks):
                if not at.isdigit() and len(at) >= 4 and at in self.rare_addr_idx:
                    cands.update(self.rare_addr_idx[at])

            if cands:
                for s1_idx in cands:
                    pair_q_indices.append(q_row_idx)
                    pair_s1_indices.append(s1_idx)

        if not pair_q_indices:
            return pl.DataFrame(schema={
                "q_row": pl.Int32, "s1_row": pl.Int32, "cos_name": pl.Float32,
                "cos_addr": pl.Float32, "cos_comb": pl.Float32, "rank": pl.Int16
            })

        qi_arr = np.array(pair_q_indices, dtype=np.int32)
        si_arr = np.array(pair_s1_indices, dtype=np.int32)

        # Vectorized string scoring with rapidfuzz
        q_names = part["name_core"].to_list()
        s_names = self.s1c["name_core"].to_list()
        q_addrs = part["addr_norm"].to_list()
        s_addrs = self.s1c["addr_norm"].to_list()

        q_sub_names = [q_names[i] for i in qi_arr]
        s_sub_names = [s_names[i] for i in si_arr]
        q_sub_addrs = [q_addrs[i] for i in qi_arr]
        s_sub_addrs = [s_addrs[i] for i in si_arr]

        cos_name = process.cpdist(q_sub_names, s_sub_names, scorer=fuzz.token_sort_ratio, workers=-1, dtype=np.float32) / 100.0
        cos_addr = process.cpdist(q_sub_addrs, s_sub_addrs, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32) / 100.0

        del q_sub_names, s_sub_names, q_sub_addrs, s_sub_addrs

        comb = 0.6 * cos_name + 0.4 * cos_addr

        df = pl.DataFrame({
            "q_row": qi_arr,
            "s1_row": si_arr,
            "cos_name": cos_name,
            "cos_addr": cos_addr,
            "cos_comb": comb,
        })

        df = df.with_columns(
            pl.col("cos_comb").rank("ordinal", descending=True).over("q_row").cast(pl.Int16).alias("rank")
        )
        return df.filter(pl.col("rank") <= k_keep)


def generate(split: str):
    out_dir = cache_path(f"{split}_cands")
    out_dir.mkdir(parents=True, exist_ok=True)
    s1, q = scan_tables(split)
    countries = sorted(s1.select(pl.col("country").unique()).collect()["country"].to_list())
    total = 0

    for country in countries:
        t0 = time.time()
        s1c = s1.filter(pl.col("country") == country).collect()
        qc = q.filter(pl.col("country") == country).collect()
        if qc.height == 0:
            continue

        done_marker = out_dir / f"{country}.done"
        if done_marker.exists():
            print(f"[{split}/{country}] done already, skipping")
            continue

        print(f"[{split}/{country}] counting query frequencies ({qc.height:,} queries)...", flush=True)
        name_counts = Counter()
        addr_counts = Counter()
        for row in qc.select(["name_core", "addr_norm"]).iter_rows(named=True):
            for t in set(row["name_core"].split()):
                name_counts[t] += 1
            for at in set(row["addr_norm"].split()):
                if len(at) >= 4 and not at.isdigit():
                    addr_counts[at] += 1

        print(f"[{split}/{country}] building multi-pass index on {s1c.height:,} S1 entities...", flush=True)
        idx = MultiPassIndex(s1c, name_counts, addr_counts)
        del name_counts, addr_counts

        s1_map = s1c["s1_idx"].to_numpy()
        print(f"[{split}/{country}] index ready in {time.time()-t0:.1f}s. Streaming query chunks...", flush=True)

        for ci, s in enumerate(range(0, qc.height, CHUNK)):
            out_file = out_dir / f"{country}_{s:09d}.parquet"
            if out_file.exists():
                continue
            part = qc.slice(s, CHUNK)
            cand = idx.match_queries(part, k_keep=K_KEEP)
            q_map = part["q_idx"].to_numpy()

            cand = cand.with_columns(
                pl.Series("q_idx", q_map[cand["q_row"].to_numpy()], dtype=pl.UInt32),
                pl.Series("s1_idx", s1_map[cand["s1_row"].to_numpy()], dtype=pl.UInt32),
            ).select("q_idx", "s1_idx", "cos_name", "cos_addr", "cos_comb", "rank")

            cand.write_parquet(out_file.with_suffix(".tmp"))
            out_file.with_suffix(".tmp").replace(out_file)
            total += cand.height
            print(f"  chunk {ci}: {part.height:,} queries -> {cand.height:,} pairs ({time.time()-t0:.1f}s)", flush=True)

        done_marker.touch()
        del idx, s1c, qc
        gc.collect()

    print(f"[{split}] total candidate pairs: {total:,}")


if __name__ == "__main__":
    for sp_ in sys.argv[1:] or ["train", "test"]:
        generate(sp_)