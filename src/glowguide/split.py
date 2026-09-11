"""Global chronological splitting, independent of evaluation eligibility."""

from dataclasses import dataclass
from numbers import Real

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: pd.Timestamp


def temporal_split(interactions: pd.DataFrame, split_quantile: float = 0.80) -> TemporalSplit:
    """Split at one global timestamp quantile, keeping cutoff ties in train.

    Quantiles use pandas' linear interpolation. Timestamp ties can make the
    realized train fraction exceed the requested fraction or leave test empty.
    The source frame is never modified and no users are filtered here.
    """
    if isinstance(split_quantile, bool) or not isinstance(split_quantile, Real) or not 0 < split_quantile < 1:
        raise ValueError("split_quantile must be a number strictly between 0 and 1")
    if interactions.empty:
        raise ValueError("Cannot split an empty interaction dataset")
    if "submission_time" not in interactions:
        raise ValueError("Interactions must contain submission_time")
    dates = pd.to_datetime(interactions["submission_time"], errors="coerce", format="mixed")
    if dates.isna().any():
        raise ValueError("submission_time contains invalid or missing timestamps")
    frame = interactions.copy()
    frame["submission_time"] = dates
    cutoff = dates.quantile(split_quantile)
    return TemporalSplit(
        train=frame.loc[dates.le(cutoff)].copy(),
        test=frame.loc[dates.gt(cutoff)].copy(),
        cutoff=cutoff,
    )
