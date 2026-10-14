"""Fixed reference/test split.

The split is decided once, by a seeded shuffle done separately inside each cheat
type, so both halves contain the same mix of honest and cheating sittings.

Everything a detector learns (per-question averages, spreads, thresholds) comes
from the HONEST sittings of the REFERENCE half only. The TEST half is only used to
measure how well the finished detectors work.
"""
import numpy as np

SPLIT_SEED = 2024
REFERENCE_FRACTION = 0.5


def assign_split(df, seed=SPLIT_SEED, reference_fraction=REFERENCE_FRACTION):
    """Return a copy of df with a 'split' column: 'reference' or 'test'."""
    rng = np.random.default_rng(seed)
    cheat_type = df["cheat_type"].to_numpy()
    is_reference = np.zeros(len(df), dtype=bool)
    for kind in sorted(set(cheat_type)):
        positions = np.flatnonzero(cheat_type == kind)
        shuffled = rng.permutation(positions)
        n_reference = int(round(len(shuffled) * reference_fraction))
        is_reference[shuffled[:n_reference]] = True
    out = df.copy()
    out["split"] = np.where(is_reference, "reference", "test")
    return out


def select_reference_honest(df):
    """Honest sittings of the reference half: the only data detectors may learn from."""
    return df[(df["split"] == "reference") & (df["is_cheat"] == 0)]


def select_test(df):
    """The held-out test half (honest and cheating sittings)."""
    return df[df["split"] == "test"]
