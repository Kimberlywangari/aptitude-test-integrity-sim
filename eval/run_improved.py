"""Run the IMPROVED detector and compare it with the baseline on the SAME test set.

Run from the repo root:  python -m eval.run_improved

Writes:
    results/improved_metrics.csv      metrics overall and per cheat type
    results/improved_thresholds.json  K, Y, z threshold and per-question stats
    results/before_after.csv          baseline and improved rows from one run
"""
import pandas as pd

from eval.common import RESULTS_DIR, data_summary, load_reference_and_test, write_json
from eval.metrics import display_table, metrics_table, write_csv
from src.detectors import (
    TARGET_FPR_PER_RULE,
    baseline_flags,
    fit_baseline,
    fit_improved,
    flag_high_focus_loss,
    flag_many_fast_answers,
    improved_flags,
)

COUNT_COLUMNS = ["group", "n_sittings", "n_honest", "n_cheat", "tp", "fp", "fn", "tn"]


def main():
    reference, test = load_reference_and_test()

    # Both detectors learn from the same honest reference sittings ...
    baseline_thresholds = fit_baseline(reference)
    improved_thresholds = fit_improved(reference)

    # ... and are scored on the identical test sittings.
    before = metrics_table(test, baseline_flags(test, baseline_thresholds))
    before.insert(0, "detector", "baseline")
    after = metrics_table(test, improved_flags(test, improved_thresholds))
    after.insert(0, "detector", "improved")

    # The baseline counts recomputed here must equal the committed baseline results.
    recorded = pd.read_csv(RESULTS_DIR / "baseline_metrics.csv")
    if not recorded[COUNT_COLUMNS].equals(before[COUNT_COLUMNS]):
        raise SystemExit("baseline counts differ from results/baseline_metrics.csv; "
                         "rerun python -m eval.run_baseline")

    write_csv(after, RESULTS_DIR / "improved_metrics.csv")
    write_csv(pd.concat([before, after], ignore_index=True), RESULTS_DIR / "before_after.csv")

    t = improved_thresholds
    write_json(
        {
            "detector": "improved",
            "rule": "flag if at least min_fast_answers answers have a per-question z-score "
                    "below -z_threshold OR focus_loss_events > max_focus_loss",
            "z_score": "(seconds - question reference mean) / question reference std, "
                       "population std, honest reference sittings only",
            "target_fpr_per_rule": TARGET_FPR_PER_RULE,
            "thresholds": {
                "z_threshold": t.z_threshold,
                "min_fast_answers": int(t.min_fast_answers),
                "max_focus_loss": int(t.max_focus_loss),
            },
            "question_mean_seconds": {c: round(float(v), 2) for c, v in t.question_mean.items()},
            "question_std_seconds": {c: round(float(v), 2) for c, v in t.question_std.items()},
            "honest_reference_sittings_flagged": {
                "many_fast_answers": int(flag_many_fast_answers(reference, t).sum()),
                "high_focus_loss": int(flag_high_focus_loss(reference, t.max_focus_loss).sum()),
                "either_rule": int(improved_flags(reference, t).sum()),
            },
            "data": data_summary(reference, test),
        },
        RESULTS_DIR / "improved_thresholds.json",
    )

    print(f"Improved detector: flag if >= {t.min_fast_answers} answers have z < -{t.z_threshold} "
          f"OR focus-loss events > {t.max_focus_loss}")
    print(f"Test set: {len(test)} sittings (identical for both detectors)")
    print("\nBASELINE")
    print(display_table(before.drop(columns="detector")).to_string(index=False))
    print("\nIMPROVED")
    print(display_table(after.drop(columns="detector")).to_string(index=False))


if __name__ == "__main__":
    main()
