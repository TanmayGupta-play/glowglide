"""Infer normalized skin attributes exclusively from training interactions."""

import pandas as pd


def infer_user_profiles(train: pd.DataFrame) -> pd.DataFrame:
    """Return a user-indexed frame with independent modal non-null attributes.

    Frequency ties use the latest observation of a tied value. If timestamps
    also tie, lexical value order resolves the ambiguity deterministically.
    Missing attributes remain nullable strings; values are not re-normalized.
    """
    if not {"author_id", "submission_time"} <= set(train.columns):
        raise ValueError("Train profiles require author_id and submission_time")
    frame = train.reindex(columns=["author_id", "submission_time", "skin_type", "skin_tone"]).copy()
    frame["author_id"] = frame["author_id"].astype("string")
    if frame["author_id"].isna().any() or frame["author_id"].eq("").any():
        raise ValueError("Training author IDs must be nonempty")
    frame["submission_time"] = pd.to_datetime(frame["submission_time"], errors="coerce", format="mixed")
    if frame["submission_time"].isna().any():
        raise ValueError("Training profile timestamps must be valid and nonmissing")
    profiles = pd.DataFrame(index=pd.Index(sorted(frame["author_id"].unique()), name="author_id"))
    for attribute in ("skin_type", "skin_tone"):
        frame[attribute] = frame[attribute].astype("string")
        counts = (
            frame.dropna(subset=[attribute]).groupby(["author_id", attribute])["submission_time"]
            .agg(count="size", latest="max").reset_index()
        )
        winners = counts.sort_values(
            ["author_id", "count", "latest", attribute], ascending=[True, False, False, True], kind="stable"
        ).drop_duplicates("author_id").set_index("author_id")[attribute]
        profiles[attribute] = winners.reindex(profiles.index).astype("string")
    return profiles
