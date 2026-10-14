"""Run the BASELINE detector on the fixed test set and write results/.

Run from the repo root:  python -m eval.run_baseline

Writes:
    results/baseline_metrics.csv     metrics overall and per cheat type
    results/baseline_thresholds.json thresholds learned from honest reference sittings
"""
from dataclasses import asdict

from eval.common import RESULTS_DIR, data_summary, load_reference_and_test, write_json
from eval.metrics import display_table, metrics_table, write_csv
from src.detectors import (
    TARGET_FPR_PER_RULE,
    baseline_flags,
    fit_baseline,
    flag_high_focus_loss,
    flag_low_average_time,
)


def main():
    reference, test = load_reference_and_test()

    # Learn X and Y from honest reference sittings only.
    thresholds = fit_baseline(reference)

    # Apply them to the test half, then score against the ground-truth labels.
    flags = baseline_flags(test, thresholds)
    table = metrics_table(test, flags)
    table.insert(0, "detector", "baseline")

    RESULTS_DIR.mkdir(exist_ok=True)
    write_csv(table, RESULTS_DIR / "baseline_metrics.csv")

    reference_flagged = {
        "low_average_time": int(flag_low_average_time(reference, thresholds.min_avg_seconds).sum()),
        "high_focus_loss": int(flag_high_focus_loss(reference, thresholds.max_focus_loss).sum()),
        "either_rule": int(baseline_flags(reference, thresholds).sum()),
    }
    write_json(
        {
            "detector": "baseline",
            "rule": "flag if average seconds per question < min_avg_seconds "
                    "OR focus_loss_events > max_focus_loss",
            "target_fpr_per_rule": TARGET_FPR_PER_RULE,
            "thresholds": asdict(thresholds),
            "honest_reference_sittings_flagged": reference_flagged,
            "data": data_summary(reference, test),
        },
        RESULTS_DIR / "baseline_thresholds.json",
    )

    print("Baseline thresholds (from honest reference sittings):")
    print(f"  flag if average time per question < {thresholds.min_avg_seconds:.2f} s")
    print(f"  flag if focus-loss events > {thresholds.max_focus_loss}")
    print(f"Test set: {len(test)} sittings")
    print(display_table(table.drop(columns="detector")).to_string(index=False))


if __name__ == "__main__":
    main()
