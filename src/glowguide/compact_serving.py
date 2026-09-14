"""Lossless serving storage adapters; experimentation models stay untouched.

Adapters preserve the Mapping/Set interfaces consumed by existing scoring
methods. Only one user's strings/sets are materialized per request. Conversion
uses shallow serving copies and never mutates the fitted source models.
"""

from collections.abc import Mapping, Set
from copy import copy

import numpy as np
from scipy.sparse import csr_matrix


def unsigned_dtype(maximum):
    for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
        if maximum <= np.iinfo(dtype).max:
            return dtype
    raise ValueError("Index exceeds uint64 capacity")


class CompactUserHistoryIndex(Mapping):
    def __init__(self, user_index, product_ids, histories):
        self.user_index = user_index
        self.product_ids = product_ids
        product_index = {p: i for i, p in enumerate(product_ids)}
        lengths = np.zeros(len(user_index), dtype=np.int64)
        for user, row in user_index.items():
            lengths[row] = len(histories[user])
        total = int(lengths.sum())
        self.indptr = np.empty(len(user_index) + 1, dtype=unsigned_dtype(total))
        self.indptr[0] = 0
        np.cumsum(lengths, out=self.indptr[1:])
        self.indices = np.empty(total, dtype=unsigned_dtype(len(product_ids) - 1))
        for user, row in user_index.items():
            self.indices[self.indptr[row]:self.indptr[row + 1]] = sorted(product_index[p] for p in histories[user])

    def __len__(self):
        return len(self.user_index)

    def __iter__(self):
        return iter(self.user_index)

    def __contains__(self, user):
        return user in self.user_index

    def __getitem__(self, user):
        row = self.user_index[user]
        return frozenset(self.product_ids[i] for i in self.indices[self.indptr[row]:self.indptr[row + 1]])


class CompactPositiveUsers(Set):
    def __init__(self, user_index, users):
        self.user_index = user_index
        self.mask = np.zeros(len(user_index), dtype=bool)
        for user in users:
            self.mask[user_index[user]] = True
        self.count = int(self.mask.sum())

    def __len__(self):
        return self.count

    def __iter__(self):
        return (u for u, i in self.user_index.items() if self.mask[i])

    def __contains__(self, user):
        row = self.user_index.get(user)
        return row is not None and bool(self.mask[row])


class CompactPositiveUserIndex(Mapping):
    """Map positive users to the shared row space, excluding negative-only IDs."""

    def __init__(self, users):
        self.users = users

    def __len__(self):
        return len(self.users)

    def __iter__(self):
        return iter(self.users)

    def __contains__(self, user):
        return user in self.users

    def __getitem__(self, user):
        if user not in self.users:
            raise KeyError(user)
        return self.users.user_index[user]


class CompactUserProfiles(Mapping):
    def __init__(self, user_index, profiles):
        self.user_index = user_index
        table = dict.fromkeys(profiles.values())
        self.profiles = tuple(table)
        lookup = {profile: i for i, profile in enumerate(self.profiles)}
        self.codes = np.empty(len(user_index), dtype=unsigned_dtype(len(table) - 1))
        for user, row in user_index.items():
            self.codes[row] = lookup[profiles[user]]

    def __len__(self):
        return len(self.user_index)

    def __iter__(self):
        return iter(self.user_index)

    def __getitem__(self, user):
        return self.profiles[self.codes[self.user_index[user]]]


def compact_serving_bundle(source):
    """Pack fitted serving copies without changing any learned float values.

    Content sums duplicate positive ratings while collaborative uses their
    maximum. Share weights ONLY after exact CSR equality; otherwise retain
    each model's original weights in the common row space.
    """
    if isinstance(source.seen_by_user, CompactUserHistoryIndex):
        return source
    bundle = copy(source)
    for name in ("popularity", "content", "profile", "collaborative"):
        setattr(bundle, name, copy(getattr(source, name)))
    product_ids = source.popularity.product_ids
    product_index = {p: i for i, p in enumerate(product_ids)}
    catalog = frozenset(product_ids)
    for name in ("content", "profile", "collaborative"):
        model = getattr(bundle, name)
        model.product_ids = product_ids
        model._product_index = product_index
        model.candidate_product_ids = catalog
    bundle.popularity.candidate_product_ids = catalog
    user_index = source.collaborative._user_index
    bundle.seen_by_user = CompactUserHistoryIndex(user_index, product_ids, source.seen_by_user)
    bundle.positive_history_users = CompactPositiveUsers(user_index, source.positive_history_users)
    bundle.profile._users = CompactUserProfiles(user_index, source.profile._users)
    del bundle.profile.user_profiles  # Fit-only pandas copy of the same profiles.
    del bundle.content.vectorizer  # Serving uses fitted vectors and _terms only.
    del bundle.collaborative.positive_matrix  # Fit-only binary co-occurrence input.
    old_weights = source.content._positive_weights.tocoo()
    row_map = np.empty(old_weights.shape[0], dtype=np.int64)
    for user, row in source.content._user_index.items():
        row_map[row] = user_index[user]
    aligned = csr_matrix((old_weights.data, (row_map[old_weights.row], old_weights.col)),
                         shape=source.collaborative._positive_weights.shape)
    collaborative = bundle.collaborative._positive_weights
    if all(np.array_equal(getattr(aligned, field), getattr(collaborative, field))
           for field in ("indptr", "indices", "data")):
        aligned = collaborative
    bundle.content._positive_weights = aligned
    bundle.content._user_index = CompactPositiveUserIndex(bundle.positive_history_users)
    bundle.build_metadata = {**source.build_metadata, "serving_representation": "compact-v1"}
    return bundle
