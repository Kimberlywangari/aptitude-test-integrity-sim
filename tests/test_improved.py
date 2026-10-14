import math

import pandas as pd
import pytest

from src.detectors import (
    ImprovedThresholds,
    baseline_flags,
    count_fast_answers,
    fit_baseline,
    fit_focus_threshold,
    fit_improved,
    fit_min_fast_answers,
    fit_question_stats,
    flag_many_fast_answers,
    improved_flags,
    timing_z_scores,
)


def table(times, focus=None, n_questions=2):
    """Tiny table of sittings. times is a list of per-question seconds lists."""
    columns = [f"t_q{j:02d}" for j in range(1, n_questions + 1)]
    df = pd.DataFrame(times, columns=columns)
    df["focus_loss_events"] = focus if focus is not None else 0
    return df


# Hand-checked reference: 4 honest sittings, 2 questions.
#   question 1 seconds: 10, 20, 10, 20  -> mean 15, population std 5
#   question 2 seconds: 40, 60, 40, 60  -> mean 50, population std 10
@pytest.fixture
def reference():
    return table([[10, 40], [20, 60], [10, 40], [20, 60]])


@pytest.fixture
def test_rows():
    # A: z = (-1.5, 0)    exactly on the boundary, not beyond it
    # B: z = (-2.0, -2.0) two unusually fast answers
    # C: z = ( 0.0, 0.0)  typical
    # D: z = (+1.0, +2.0) slow answers, which are never "too fast"
    return table([[7.5, 50], [5, 30], [15, 50], [20, 70]])


def test_question_stats_hand_checked(reference):
    means, stds = fit_question_stats(reference)
    assert list(means) == [15.0, 50.0]
    assert list(stds) == [5.0, 10.0]


def test_z_scores_hand_checked(reference, test_rows):
    means, stds = fit_question_stats(reference)
    z = timing_z_scores(test_rows, means, stds)
    assert z.values.tolist() == [[-1.5, 0.0], [-2.0, -2.0], [0.0, 0.0], [1.0, 2.0]]


def test_fast_answer_counts_hand_checked(reference, test_rows):
    means, stds = fit_question_stats(reference)
    counts = count_fast_answers(test_rows, means, stds, z_threshold=1.5)
    # boundary z = -1.5 is not beyond -1.5; slow answers are not counted
    assert list(counts) == [0, 2, 0, 0]


def test_stricter_threshold_counts_fewer_answers(reference, test_rows):
    means, stds = fit_question_stats(reference)
    counts = count_fast_answers(test_rows, means, stds, z_threshold=2.5)
    assert list(counts) == [0, 0, 0, 0]


def test_zero_spread_question_is_never_counted():
    reference = table([[10, 40], [10, 60], [10, 40], [10, 60]])  # question 1 never varies
    means, stds = fit_question_stats(reference)
    assert stds.iloc[0] == 0
    sitting = table([[1, 50]])
    z = timing_z_scores(sitting, means, stds)
    assert math.isnan(z.iloc[0, 0])
    assert count_fast_answers(sitting, means, stds, 1.5).iloc[0] == 0


def test_fit_min_fast_answers_respects_the_budget():
    counts = pd.Series([0] * 90 + [1] * 5 + [3] * 3 + [6] * 2)
    # reaching >=1: 10%, >=2: 5%, >=3: 5%, >=4: 2%, >=7: 0%
    assert fit_min_fast_answers(counts, target_fpr=0.50) == 1
    assert fit_min_fast_answers(counts, target_fpr=0.05) == 2
    assert fit_min_fast_answers(counts, target_fpr=0.02) == 4
    assert fit_min_fast_answers(counts, target_fpr=0.0) == 7


def test_flag_many_fast_answers_uses_k(reference, test_rows):
    means, stds = fit_question_stats(reference)
    thresholds = ImprovedThresholds(means, stds, z_threshold=1.5, min_fast_answers=2,
                                    max_focus_loss=5)
    assert list(flag_many_fast_answers(test_rows, thresholds)) == [False, True, False, False]
    thresholds.min_fast_answers = 3
    assert list(flag_many_fast_answers(test_rows, thresholds)) == [False, False, False, False]


def test_improved_flags_if_timing_or_focus_rule_fires(reference):
    means, stds = fit_question_stats(reference)
    thresholds = ImprovedThresholds(means, stds, z_threshold=1.5, min_fast_answers=2,
                                    max_focus_loss=4)
    df = table([[5, 30], [15, 50], [15, 50], [5, 30]], focus=[0, 0, 9, 9])
    # fast only, normal, focus-loss only, both
    assert list(improved_flags(df, thresholds)) == [True, False, True, True]


def test_fit_improved_learns_only_from_the_reference_it_is_given(reference):
    thresholds = fit_improved(reference)
    assert list(thresholds.question_mean) == [15.0, 50.0]
    assert list(thresholds.question_std) == [5.0, 10.0]
    assert thresholds.z_threshold == 1.5
    # inside the reference nobody has an answer beyond z = -1.5, so K = 1 is within budget
    assert thresholds.min_fast_answers == 1
    # all-zero focus loss: Y = 0, the same as the baseline's focus rule would learn
    assert thresholds.max_focus_loss == fit_focus_threshold(reference["focus_loss_events"]) == 0


def test_improved_does_not_read_label_columns(reference, test_rows):
    thresholds = fit_improved(reference)
    labelled = test_rows.copy()
    labelled["is_cheat"] = [0, 1, 0, 1]
    labelled["cheat_type"] = ["none", "too_fast", "none", "idle_burst"]
    assert list(improved_flags(test_rows, thresholds)) == list(improved_flags(labelled, thresholds))


def test_per_question_rule_catches_a_burst_that_hides_in_a_normal_average():
    # 100 honest sittings, 20 questions, every question mean 50 and std 10
    # (sitting i answers question j in 40 s or 60 s, alternating), so every
    # honest sitting averages exactly 50 s per question.
    rows = [[40 if (i + j) % 2 == 0 else 60 for j in range(20)] for i in range(100)]
    reference = table(rows, n_questions=20)

    # Idle-then-burst: 10 answers in 20 s (z = -3) and 10 long ones (z = +7).
    # Its average is 70 s, which looks perfectly normal to an average-time rule.
    burst = table([[20] * 10 + [120] * 10], n_questions=20)
    normal = table([[50] * 20], n_questions=20)
    both = pd.concat([burst, normal], ignore_index=True)

    baseline = fit_baseline(reference)
    improved = fit_improved(reference)
    assert list(baseline_flags(both, baseline)) == [False, False]
    assert list(improved_flags(both, improved)) == [True, False]
