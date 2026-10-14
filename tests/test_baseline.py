import pandas as pd

from src.detectors import (
    BaselineThresholds,
    average_time,
    baseline_flags,
    fit_baseline,
    fit_focus_threshold,
    flag_high_focus_loss,
    flag_low_average_time,
)


def make_sittings(times, focus):
    """Tiny table: one row per sitting, 3 questions, no label columns at all."""
    df = pd.DataFrame(times, columns=["t_q01", "t_q02", "t_q03"])
    df["focus_loss_events"] = focus
    return df


def test_average_time_is_row_mean_of_question_columns_only():
    df = make_sittings([[10, 20, 30], [5, 5, 5]], [9, 9])
    assert list(average_time(df)) == [20.0, 5.0]


def test_low_average_time_rule_is_strictly_below_threshold():
    df = make_sittings([[10, 10, 10], [20, 20, 20], [50, 50, 50]], [0, 0, 0])
    flags = flag_low_average_time(df, min_avg_seconds=20.0)
    # 10 < 20 is flagged; exactly 20 is not; 50 is not
    assert list(flags) == [True, False, False]


def test_high_focus_loss_rule_is_strictly_above_threshold():
    df = make_sittings([[30, 30, 30]] * 4, [0, 3, 4, 5])
    flags = flag_high_focus_loss(df, max_focus_loss=4)
    assert list(flags) == [False, False, False, True]


def test_baseline_flags_if_either_rule_fires():
    df = make_sittings(
        [[10, 10, 10], [50, 50, 50], [50, 50, 50], [10, 10, 10]],
        [0, 0, 9, 9],
    )
    thresholds = BaselineThresholds(min_avg_seconds=20.0, max_focus_loss=4)
    # fast only, normal, focus-loss only, both
    assert list(baseline_flags(df, thresholds)) == [True, False, True, True]


def test_baseline_does_not_read_label_columns():
    df = make_sittings([[10, 10, 10], [50, 50, 50]], [0, 0])
    labelled = df.copy()
    labelled["is_cheat"] = [0, 1]
    labelled["cheat_type"] = ["none", "too_fast"]
    thresholds = BaselineThresholds(min_avg_seconds=20.0, max_focus_loss=4)
    assert list(baseline_flags(df, thresholds)) == list(baseline_flags(labelled, thresholds))


def test_fit_baseline_hand_checked_example():
    # 100 honest sittings where sitting i takes i seconds on every question (i = 1..100)
    # so their average times are exactly 1..100.
    times = [[i, i, i] for i in range(1, 101)]
    # focus loss: 98 sittings with 0, one with 1, one with 7
    focus = [0] * 98 + [1, 7]
    reference = make_sittings(times, focus)
    thresholds = fit_baseline(reference, target_fpr=0.01)
    # 1% quantile of 1..100 by linear interpolation = 1 + 0.99 * (2 - 1) = 1.99
    assert abs(thresholds.min_avg_seconds - 1.99) < 1e-9
    # Y = 0 would flag 2 of 100 (2% > 1%); Y = 1 flags only the sitting with 7 (1%)
    assert thresholds.max_focus_loss == 1


def test_fit_focus_threshold_respects_the_budget():
    focus = pd.Series([0] * 90 + [1] * 5 + [2] * 3 + [10] * 2)
    # exceeding 0: 10/100, exceeding 1: 5/100, exceeding 2: 2/100, exceeding 3: 2/100
    assert fit_focus_threshold(focus, target_fpr=0.05) == 1
    assert fit_focus_threshold(focus, target_fpr=0.02) == 2
    assert fit_focus_threshold(focus, target_fpr=0.20) == 0
