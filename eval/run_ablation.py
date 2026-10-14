"""Ablation: switch the detector's rules on and off on the SAME test set.

Run from the repo root:  python -m eval.run_ablation

Three rules ("components"), each fitted from honest reference sittings only:
    avg_time    baseline rule: average seconds per question below X
    z_score     improved rule: at least K answers with a per-question z-score below -z
    focus_loss  focus-loss events above Y (identical in baseline and improved)

A variant flags a sitting if ANY of its switched-on rules fires.

Writes:
    results/ablation_metrics.csv           every on/off combination
    results/z_threshold_sensitivity.csv    improved detector with different z cutoffs
"""
import pandas as pd

from eval.common import RESULTS_DIR, load_reference_and_test
from eval.metrics import display_table, metrics_table, write_csv
from src.detectors import (
    fit_baseline,
    fit_improved,
    flag_high_focus_loss,
    flag_low_average_time,
    flag_many_fast_answers,
)

# (name, avg_time on, z_score on, focus_loss on)
VARIANTS = [
    ("focus_loss only", False, False, True),
    ("avg_time only", True, False, False),
    ("z_score only", False, True, False),
    ("avg_time + focus_loss (baseline)", True, False, True),
    ("z_score + focus_loss (improved)", False, True, True),
    ("avg_time + z_score", True, True, False),
    ("avg_time + z_score + focus_loss", True, True, True),
]

Z_VALUES = [1.0, 1.5, 2.0, 2.5]


def component_flags(df, baseline_thresholds, improved_thresholds):
    """Each rule's flags on its own, as {rule name: boolean Series}."""
    return {
        "avg_time": flag_low_average_time(df, baseline_thresholds.min_avg_seconds),
        "z_score": flag_many_fast_answers(df, improved_thresholds),
        "focus_loss": flag_high_focus_loss(df, improved_thresholds.max_focus_loss),
    }


def combine(components, avg_time, z_score, focus_loss):
    """OR together the rules that are switched on."""
    switched_on = {"avg_time": avg_time, "z_score": z_score, "focus_loss": focus_loss}
    flags = None
    for name, is_on in switched_on.items():
        if is_on:
            flags = components[name] if flags is None else flags | components[name]
    return flags


def label_columns(table, **columns):
    """Insert identifying columns (in the given order) at the front of a table."""
    for name, value in reversed(list(columns.items())):
        table.insert(0, name, value)
    return table


def main():
    reference, test = load_reference_and_test()
    baseline_thresholds = fit_baseline(reference)
    improved_thresholds = fit_improved(reference)
    assert baseline_thresholds.max_focus_loss == improved_thresholds.max_focus_loss

    components = component_flags(test, baseline_thresholds, improved_thresholds)

    tables = []
    for name, avg_time, z_score, focus_loss in VARIANTS:
        flags = combine(components, avg_time, z_score, focus_loss)
        table = metrics_table(test, flags)
        label_columns(table, variant=name, avg_time_rule=int(avg_time),
                      z_score_rule=int(z_score), focus_loss_rule=int(focus_loss))
        tables.append(table)
    ablation = pd.concat(tables, ignore_index=True)
    write_csv(ablation, RESULTS_DIR / "ablation_metrics.csv")

    # How much does the result depend on the z cutoff? K is re-fitted for each one.
    sensitivity_tables = []
    for z in Z_VALUES:
        thresholds = fit_improved(reference, z_threshold=z)
        fast = flag_many_fast_answers(test, thresholds)
        focus = flag_high_focus_loss(test, thresholds.max_focus_loss)
        table = metrics_table(test, fast | focus)
        label_columns(table, z_threshold=z, min_fast_answers=int(thresholds.min_fast_answers))
        sensitivity_tables.append(table)
    sensitivity = pd.concat(sensitivity_tables, ignore_index=True)
    write_csv(sensitivity, RESULTS_DIR / "z_threshold_sensitivity.csv")

    print(f"Test set: {len(test)} sittings (identical for every variant)\n")
    overall = ablation[ablation["group"] == "overall"]
    shown = display_table(overall.drop(columns=["group"]).rename(columns={"variant": "group"}))
    print("ABLATION (overall)")
    print(shown.drop(columns=["honest", "cheats"]).to_string(index=False))
    print("\nZ-THRESHOLD SENSITIVITY (z_score + focus_loss, overall)")
    overall = sensitivity[sensitivity["group"] == "overall"].copy()
    overall["group"] = [f"z<-{z} (K={k})" for z, k in
                        zip(overall["z_threshold"], overall["min_fast_answers"])]
    print(display_table(overall.drop(columns=["z_threshold", "min_fast_answers"]))
          .drop(columns=["honest", "cheats"]).to_string(index=False))


if __name__ == "__main__":
    main()
