"""Render the markdown tables used in the docs, straight from results/ files.

Run from the repo root:  python -m eval.make_tables
(after eval.run_baseline, eval.run_improved and eval.run_ablation)

Writes results/comparison_tables.md. The docs copy tables from that file, so no
number in them is typed by hand.
"""
import json
from pathlib import Path

import pandas as pd

from eval.common import RESULTS_DIR
from eval.metrics import ratio_text
from src.generator import CHEAT_TYPES, MIN_SECONDS

OUT_PATH = RESULTS_DIR / "comparison_tables.md"


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def metric_cells(r):
    """TP FP FN TN and the four rates as 'count/total (pct)'."""
    return [
        r.tp, r.fp, r.fn, r.tn,
        ratio_text(r.tp, r.tp + r.fp),
        ratio_text(r.tp, r.tp + r.fn),
        ratio_text(r.fp, r.fp + r.tn),
        ratio_text(2 * r.tp, 2 * r.tp + r.fp + r.fn),
    ]


METRIC_HEADERS = ["TP", "FP", "FN", "TN", "Precision", "Recall", "False-positive rate", "F1"]


def group_row(table, group):
    return next(table[table["group"] == group].itertuples(index=False))


def caught(r):
    return ratio_text(r.tp, r.tp + r.fn)


def overall_before_after(before_after):
    rows = []
    for detector in ["baseline", "improved"]:
        r = group_row(before_after[before_after["detector"] == detector], "overall")
        rows.append([detector.capitalize(), r.tp + r.fp] + metric_cells(r))
    return md_table(["Detector", "Flagged"] + METRIC_HEADERS, rows)


def per_type_before_after(before_after):
    base = before_after[before_after["detector"] == "baseline"]
    impr = before_after[before_after["detector"] == "improved"]
    rows = []
    for group in CHEAT_TYPES + ["overall"]:
        b, i = group_row(base, group), group_row(impr, group)
        change = f"{i.tp - b.tp:+d} ({100 * (i.recall - b.recall):+.1f} pp)"
        label = "all cheat types" if group == "overall" else group
        rows.append([label, b.n_cheat, caught(b), caught(i), change])
    return md_table(["Cheat type", "Cheats in test", "Baseline caught", "Improved caught",
                     "Change in caught"], rows)


def per_type_full(before_after, detector):
    table = before_after[before_after["detector"] == detector]
    rows = []
    for group in CHEAT_TYPES:
        r = group_row(table, group)
        rows.append([group, r.n_cheat] + metric_cells(r))
    return md_table(["Cheat type", "Cheats"] + METRIC_HEADERS, rows)


def ablation_overall(ablation):
    rows = []
    for variant in ablation["variant"].unique():
        part = ablation[ablation["variant"] == variant]
        r = group_row(part, "overall")
        rules = "".join(
            letter if on else "-"
            for letter, on in zip("AZF", [r.avg_time_rule, r.z_score_rule, r.focus_loss_rule])
        )
        rows.append([variant, rules, r.tp + r.fp] + metric_cells(r))
    return md_table(["Variant", "Rules (A/Z/F)", "Flagged"] + METRIC_HEADERS, rows)


def ablation_per_type(ablation):
    rows = []
    for variant in ablation["variant"].unique():
        part = ablation[ablation["variant"] == variant]
        cells = [caught(group_row(part, t)) for t in CHEAT_TYPES]
        honest_flagged = group_row(part, "overall")
        rows.append([variant] + cells + [ratio_text(honest_flagged.fp, honest_flagged.n_honest)])
    return md_table(["Variant"] + CHEAT_TYPES + ["Honest flagged (FP)"], rows)


def sensitivity(sens):
    rows = []
    for (z, k), part in sens.groupby(["z_threshold", "min_fast_answers"], sort=True):
        r = group_row(part, "overall")
        cells = [caught(group_row(part, t)) for t in CHEAT_TYPES]
        rows.append([f"{z:g}", k] + metric_cells(r) + cells)
    return md_table(["z cutoff", "K"] + METRIC_HEADERS + [f"{t} caught" for t in CHEAT_TYPES], rows)


def settings_table(baseline_info, improved_info):
    bt, it = baseline_info["thresholds"], improved_info["thresholds"]
    bf, i_f = baseline_info["honest_reference_sittings_flagged"], \
        improved_info["honest_reference_sittings_flagged"]
    n_ref = baseline_info["data"]["n_reference_honest"]
    rows = [
        ["Timing rule",
         f"average seconds per question < {bt['min_avg_seconds']:.2f}",
         f"at least {it['min_fast_answers']} answers with z < -{it['z_threshold']:g}"],
        ["Focus-loss rule", f"events > {bt['max_focus_loss']}", f"events > {it['max_focus_loss']}"],
        ["Honest reference sittings flagged by timing rule",
         ratio_text(bf["low_average_time"], n_ref), ratio_text(i_f["many_fast_answers"], n_ref)],
        ["Honest reference sittings flagged by focus-loss rule",
         ratio_text(bf["high_focus_loss"], n_ref), ratio_text(i_f["high_focus_loss"], n_ref)],
        ["Honest reference sittings flagged by either rule",
         ratio_text(bf["either_rule"], n_ref), ratio_text(i_f["either_rule"], n_ref)],
    ]
    return md_table(["Setting", "Baseline", "Improved"], rows)


def lowest_possible_z(improved_info):
    """Lowest z an answer could get on each question, given the 2-second floor."""
    means = improved_info["question_mean_seconds"]
    stds = improved_info["question_std_seconds"]
    values = sorted((MIN_SECONDS - means[q]) / stds[q] for q in means)
    n = len(values)
    return {
        "lowest": values[0],
        "median": values[n // 2],
        "highest": values[-1],
        "n_questions": n,
        "possible_at_1.5": sum(v < -1.5 for v in values),
        "possible_at_2.0": sum(v < -2.0 for v in values),
        "possible_at_2.5": sum(v < -2.5 for v in values),
    }


def main():
    before_after = pd.read_csv(RESULTS_DIR / "before_after.csv")
    ablation = pd.read_csv(RESULTS_DIR / "ablation_metrics.csv")
    sens = pd.read_csv(RESULTS_DIR / "z_threshold_sensitivity.csv")
    baseline_info = json.loads((RESULTS_DIR / "baseline_thresholds.json").read_text())
    improved_info = json.loads((RESULTS_DIR / "improved_thresholds.json").read_text())

    data = improved_info["data"]
    z = lowest_possible_z(improved_info)
    sections = [
        "<!-- Generated by python -m eval.make_tables from files in results/. Do not edit. -->",
        f"Test set: {data['n_test']} sittings ({data['n_test_honest']} honest, "
        f"{data['n_test_cheat']} cheating). Thresholds learned from "
        f"{data['n_reference_honest']} honest reference sittings.",
        "### Table 1. Before/after, overall\n\n" + overall_before_after(before_after),
        "### Table 2. Before/after, cheats caught per cheat type\n\n"
        + per_type_before_after(before_after),
        "### Table 3. Baseline, per cheat type\n\n" + per_type_full(before_after, "baseline"),
        "### Table 4. Improved, per cheat type\n\n" + per_type_full(before_after, "improved"),
        "### Table 5. Thresholds and reference false-alarm budget\n\n"
        + settings_table(baseline_info, improved_info),
        "### Table 6. Ablation, overall\n\n" + ablation_overall(ablation),
        "### Table 7. Ablation, cheats caught per cheat type\n\n" + ablation_per_type(ablation),
        "### Table 8. Sensitivity to the z cutoff (z_score + focus_loss, K re-fitted each time)"
        "\n\n" + sensitivity(sens),
        "### Lowest z-score any answer can reach\n\n"
        f"With a {MIN_SECONDS:g} s floor on answer time and the per-question reference means "
        f"and stds in improved_thresholds.json, the lowest possible z on a question is "
        f"(2 - mean) / std. Across the {z['n_questions']} questions it ranges from "
        f"{z['lowest']:.2f} to {z['highest']:.2f} (median {z['median']:.2f}). "
        f"Questions where z < -1.5 is reachable: {z['possible_at_1.5']} of {z['n_questions']}; "
        f"z < -2.0: {z['possible_at_2.0']} of {z['n_questions']}; "
        f"z < -2.5: {z['possible_at_2.5']} of {z['n_questions']}.",
    ]
    OUT_PATH.write_text("\n\n".join(sections) + "\n", newline="\n")
    print(f"wrote {OUT_PATH.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
