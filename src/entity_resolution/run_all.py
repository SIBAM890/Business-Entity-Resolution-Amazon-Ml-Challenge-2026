"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

Each stage caches its result under cache/ and is skipped/resumed when the cache exists.
"""
import time

from . import prepare_data, build_features, candidates, indic_dictionary, predict, train
from .build_normalized import main as build_normalized_main
from .apply_indic_dict import main as apply_indic_dict_main
from .config import cache_path


def main():
    t0 = time.time()
    if not cache_path("test_source3.parquet").exists():
        prepare_data.main()
    build_normalized_main()
    # Indic-script names: learn token dictionary on the training fold, apply to both splits
    if not indic_dictionary.DICT_PATH.exists():
        indic_dictionary.learn()
    apply_indic_dict_main()
    for split in ("train", "test"):
        candidates.generate(split)
        build_features.build(split)
    if not train.MODEL_PATH.exists():
        train.main()
    predict.main()
    print(f"pipeline finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()