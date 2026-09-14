"""Exhaustively compare original and rebuilt serving storage, with exact floats."""

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def equal_csr(a, b):
    assert a.shape == b.shape
    for field in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--rebuilt", type=Path, default=ROOT / "artifacts/serving_bundle.joblib")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, new = joblib.load(args.original), joblib.load(args.rebuilt)
    assert old.product_metadata == new.product_metadata
    assert len(old.seen_by_user) == len(new.seen_by_user)
    assert old.positive_history_users == set(new.positive_history_users)
    for user, seen in old.seen_by_user.items():
        assert new.seen_by_user[user] == seen
        assert new.profile._users[user] == old.profile._users[user]
    assert old.collaborative._user_index == new.collaborative._user_index
    equal_csr(old.collaborative._positive_weights, new.collaborative._positive_weights)
    equal_csr(old.collaborative.item_similarity, new.collaborative.item_similarity)
    # Compare all content rows in their original row order, including weights
    # that intentionally differ from collaborative for duplicate input pairs.
    row_order = np.empty(len(old.content._user_index), dtype=np.int64)
    for user, row in old.content._user_index.items():
        row_order[row] = new.content._user_index[user]
    equal_csr(old.content._positive_weights, new.content._positive_weights[row_order])
    equal_csr(old.content.product_vectors, new.content.product_vectors)
    equal_csr(old.content._product_transpose, new.content._product_transpose)
    np.testing.assert_array_equal(old.content._terms, new.content._terms)
    np.testing.assert_array_equal(old.popularity._scores, new.popularity._scores)
    assert old.profile.global_positive_rate == new.profile.global_positive_rate
    assert old.profile._rankings == new.profile._rankings
    assert old.profile._stats.keys() == new.profile._stats.keys()
    for key, values in old.profile._stats.items():
        for expected, actual in zip(values, new.profile._stats[key]):
            np.testing.assert_array_equal(expected, actual)
    for name in ("popularity", "content", "profile", "collaborative"):
        assert getattr(old, name).product_ids == getattr(new, name).product_ids
    result = {"result": "exact equality", "users_checked": len(old.seen_by_user),
              "positive_users_checked": len(old.content._user_index),
              "metadata_products_checked": len(old.product_metadata),
              "seen_pairs_checked": sum(map(len, old.seen_by_user.values())),
              "all_learned_score_arrays": "exact", "all_inferred_profiles": "exact",
              "shared_positive_weights": new.content._positive_weights is new.collaborative._positive_weights,
              "seen_indptr_dtype": str(new.seen_by_user.indptr.dtype),
              "seen_product_dtype": str(new.seen_by_user.indices.dtype),
              "seen_array_bytes": new.seen_by_user.indptr.nbytes + new.seen_by_user.indices.nbytes,
              "profile_code_bytes": new.profile._users.codes.nbytes,
              "positive_mask_bytes": new.positive_history_users.mask.nbytes}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
