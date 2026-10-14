import pandas as pd

from src.generator import (
    CHEAT_RATE,
    CHEAT_TYPE_SHARES,
    CHEAT_TYPES,
    MIN_SECONDS,
    N_QUESTIONS,
    N_SITTINGS,
    NONE_TYPE,
    generate_sittings,
    time_columns,
)


def test_same_seed_gives_identical_table():
    first = generate_sittings(n_sittings=500, seed=123)
    second = generate_sittings(n_sittings=500, seed=123)
    pd.testing.assert_frame_equal(first, second)


def test_different_seed_gives_different_table():
    first = generate_sittings(n_sittings=500, seed=123)
    other = generate_sittings(n_sittings=500, seed=124)
    assert not first.equals(other)


def test_table_shape_and_columns():
    df = generate_sittings(n_sittings=200, seed=1)
    assert len(df) == 200
    expected = ["sitting_id", "is_cheat", "cheat_type", "focus_loss_events"] + time_columns()
    assert list(df.columns) == expected
    assert df["sitting_id"].is_unique


def test_cheat_proportion_is_exact():
    df = generate_sittings()
    assert len(df) == N_SITTINGS
    assert df["is_cheat"].sum() == round(N_SITTINGS * CHEAT_RATE)
    assert df["is_cheat"].mean() == CHEAT_RATE


def test_cheat_type_counts_match_shares():
    df = generate_sittings()
    n_cheats = int(df["is_cheat"].sum())
    counts = df["cheat_type"].value_counts()
    assert set(counts.index) == set(CHEAT_TYPES) | {NONE_TYPE}
    assert counts[NONE_TYPE] == N_SITTINGS - n_cheats
    for cheat_type, share in CHEAT_TYPE_SHARES.items():
        # the last type absorbs rounding, so allow a difference of one sitting
        assert abs(counts[cheat_type] - share * n_cheats) <= 1


def test_label_columns_agree():
    df = generate_sittings(n_sittings=1000, seed=5)
    assert ((df["is_cheat"] == 1) == (df["cheat_type"] != NONE_TYPE)).all()


def test_times_are_in_valid_range():
    df = generate_sittings(n_sittings=1000, seed=5)
    times = df[time_columns()]
    assert times.shape[1] == N_QUESTIONS
    assert times.min().min() >= MIN_SECONDS
    assert not times.isna().any().any()
    assert (df["focus_loss_events"] >= 0).all()


def test_cheat_signals_differ_from_honest_in_the_intended_way():
    df = generate_sittings()
    df["mean_time"] = df[time_columns()].mean(axis=1)
    by_type = df.groupby("cheat_type")
    honest_mean_time = by_type["mean_time"].mean()[NONE_TYPE]
    honest_focus = by_type["focus_loss_events"].mean()[NONE_TYPE]
    # too_fast sittings are much quicker than honest ones on average
    assert by_type["mean_time"].mean()["too_fast"] < 0.6 * honest_mean_time
    # repeated focus loss really does have far more focus-loss events
    assert by_type["focus_loss_events"].mean()["focus_loss"] > 4 * honest_focus
