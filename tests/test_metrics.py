import math

import pandas as pd
import pytest

from eval.metrics import (
    confusion_counts,
    display_table,
    divide,
    metrics_table,
    ratio_text,
    write_csv,
)

# Hand-checked example, 10 sittings:
#
#   row  true type     flagged   outcome
#    0   too_fast      yes       TP
#    1   too_fast      yes       TP
#    2   too_fast      no        FN
#    3   idle_burst    yes       TP
#    4   idle_burst    no        FN
#    5   none          yes       FP
#    6   none          no        TN
#    7   none          no        TN
#    8   none          no        TN
#    9   none          yes       FP
#
# Overall: TP=3 FP=2 FN=2 TN=3
#   precision = 3/5 = 0.6, recall = 3/5 = 0.6, FPR = 2/5 = 0.4, F1 = 6/10 = 0.6
# too_fast (its 3 cheats + the 5 honest): TP=2 FP=2 FN=1 TN=3
#   precision = 2/4 = 0.5, recall = 2/3, FPR = 2/5 = 0.4, F1 = 4/7
# idle_burst (its 2 cheats + the 5 honest): TP=1 FP=2 FN=1 TN=3
#   precision = 1/3, recall = 1/2, FPR = 0.4, F1 = 2/5
# focus_loss (no cheats, only the 5 honest): TP=0 FP=2 FN=0 TN=3
#   precision = 0/2 = 0, recall undefined (0/0), FPR = 0.4, F1 = 0/2 = 0


@pytest.fixture
def example():
    df = pd.DataFrame({
        "is_cheat": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "cheat_type": ["too_fast"] * 3 + ["idle_burst"] * 2 + ["none"] * 5,
    })
    flags = pd.Series([True, True, False, True, False, True, False, False, False, True])
    return df, flags


def row(table, group):
    return table[table["group"] == group].iloc[0]


def test_confusion_counts_hand_checked():
    is_cheat = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    flagged = [True, True, False, True, False, True, False, False, False, True]
    assert confusion_counts(is_cheat, flagged) == (3, 2, 2, 3)


def test_overall_metrics_hand_checked(example):
    df, flags = example
    r = row(metrics_table(df, flags), "overall")
    assert (r.tp, r.fp, r.fn, r.tn) == (3, 2, 2, 3)
    assert (r.n_sittings, r.n_honest, r.n_cheat) == (10, 5, 5)
    assert r.precision == pytest.approx(0.6)
    assert r.recall == pytest.approx(0.6)
    assert r.fpr == pytest.approx(0.4)
    assert r.f1 == pytest.approx(0.6)


def test_per_cheat_type_metrics_hand_checked(example):
    df, flags = example
    table = metrics_table(df, flags)

    r = row(table, "too_fast")
    assert (r.tp, r.fp, r.fn, r.tn) == (2, 2, 1, 3)
    assert r.precision == pytest.approx(0.5)
    assert r.recall == pytest.approx(2 / 3)
    assert r.fpr == pytest.approx(0.4)
    assert r.f1 == pytest.approx(4 / 7)

    r = row(table, "idle_burst")
    assert (r.tp, r.fp, r.fn, r.tn) == (1, 2, 1, 3)
    assert r.precision == pytest.approx(1 / 3)
    assert r.recall == pytest.approx(0.5)
    assert r.f1 == pytest.approx(0.4)


def test_type_with_no_sittings_has_undefined_recall(example):
    df, flags = example
    r = row(metrics_table(df, flags), "focus_loss")
    assert (r.tp, r.fp, r.fn, r.tn) == (0, 2, 0, 3)
    assert math.isnan(r.recall)
    assert r.precision == 0.0
    assert r.f1 == 0.0
    assert r.fpr == pytest.approx(0.4)


def test_nothing_flagged_gives_undefined_precision(example):
    df, _ = example
    flags = pd.Series([False] * 10)
    r = row(metrics_table(df, flags), "overall")
    assert (r.tp, r.fp, r.fn, r.tn) == (0, 0, 5, 5)
    assert math.isnan(r.precision)
    assert r.recall == 0.0
    assert r.fpr == 0.0
    assert r.f1 == 0.0


def test_perfect_flags_give_perfect_scores(example):
    df, _ = example
    flags = pd.Series(df["is_cheat"].astype(bool).to_list())
    r = row(metrics_table(df, flags), "overall")
    assert (r.precision, r.recall, r.fpr, r.f1) == (1.0, 1.0, 0.0, 1.0)


def test_flags_are_matched_to_sittings_by_index_not_position(example):
    df, flags = example
    shuffled = flags.sample(frac=1.0, random_state=0)  # same values, different order
    assert metrics_table(df, shuffled).equals(metrics_table(df, flags))


def test_counts_add_up_in_every_row(example):
    df, flags = example
    table = metrics_table(df, flags)
    assert (table["tp"] + table["fn"] == table["n_cheat"]).all()
    assert (table["fp"] + table["tn"] == table["n_honest"]).all()
    assert (table["n_honest"] + table["n_cheat"] == table["n_sittings"]).all()


def test_divide_returns_nan_for_zero_denominator():
    assert divide(3, 4) == 0.75
    assert math.isnan(divide(0, 0))


def test_ratio_text_puts_counts_beside_percentage():
    assert ratio_text(192, 225) == "192/225 (85.3%)"
    assert ratio_text(0, 0) == "n/a (0/0)"


def test_display_table_shows_raw_counts(example):
    df, flags = example
    shown = display_table(metrics_table(df, flags))
    overall = shown[shown["group"] == "overall"].iloc[0]
    assert overall["precision"] == "3/5 (60.0%)"
    assert overall["recall"] == "3/5 (60.0%)"
    assert overall["FPR"] == "2/5 (40.0%)"
    assert overall["F1"] == "6/10 (60.0%)"


def test_write_csv_uses_lf_line_endings(example, tmp_path):
    df, flags = example
    path = tmp_path / "metrics.csv"
    write_csv(metrics_table(df, flags), path)
    content = path.read_bytes()
    assert b"\r\n" not in content
    assert content.startswith(b"group,n_sittings,n_honest,n_cheat,tp,fp,fn,tn")
