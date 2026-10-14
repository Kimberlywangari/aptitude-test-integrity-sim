from pathlib import Path

import pandas as pd

from src.dataset import build_dataset
from src.generator import CHEAT_TYPES, generate_sittings
from src.make_data import make_sample
from src.split import assign_split, select_reference_honest, select_test

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_sittings.csv"


def test_split_is_deterministic():
    df = generate_sittings(n_sittings=1000, seed=3)
    first = assign_split(df)
    second = assign_split(df)
    pd.testing.assert_frame_equal(first, second)


def test_split_values_and_sizes():
    df = build_dataset()
    assert set(df["split"]) == {"reference", "test"}
    n_reference = (df["split"] == "reference").sum()
    n_test = (df["split"] == "test").sum()
    assert n_reference + n_test == len(df)
    # two cheat types have an odd size (225) so the halves differ by a few sittings
    assert abs(n_reference - n_test) <= 4


def test_split_is_stratified_by_cheat_type():
    df = build_dataset()
    counts = df.groupby(["cheat_type", "split"]).size().unstack()
    # every type is divided evenly between the two halves (within one sitting)
    assert ((counts["reference"] - counts["test"]).abs() <= 1).all()


def test_split_does_not_change_the_sittings():
    df = generate_sittings(n_sittings=1000, seed=3)
    split_df = assign_split(df)
    pd.testing.assert_frame_equal(split_df.drop(columns="split"), df)


def test_reference_honest_contains_only_honest_reference_rows():
    df = build_dataset()
    ref = select_reference_honest(df)
    assert len(ref) == 2125
    assert (ref["is_cheat"] == 0).all()
    assert (ref["split"] == "reference").all()


def test_test_set_keeps_the_cheat_rate_and_every_type():
    df = build_dataset()
    test = select_test(df)
    assert abs(test["is_cheat"].mean() - 0.15) < 0.002
    assert set(test["cheat_type"]) >= set(CHEAT_TYPES)


def test_reference_and_test_do_not_overlap():
    df = build_dataset()
    ref_ids = set(df.loc[df["split"] == "reference", "sitting_id"])
    test_ids = set(select_test(df)["sitting_id"])
    assert ref_ids.isdisjoint(test_ids)
    assert len(ref_ids) + len(test_ids) == len(df)


def test_committed_sample_matches_regenerated_data():
    committed = pd.read_csv(SAMPLE_PATH)
    regenerated = make_sample(build_dataset()).reset_index(drop=True)
    pd.testing.assert_frame_equal(committed, regenerated)
