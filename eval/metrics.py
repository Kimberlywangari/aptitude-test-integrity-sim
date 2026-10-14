"""Flag metrics: TP, FP, FN, TN, precision, recall, false-positive rate and F1.

Definitions (a "positive" is a flagged sitting, a "cheat" is is_cheat == 1):
    TP  cheating sitting that was flagged
    FP  honest sitting that was flagged
    FN  cheating sitting that was NOT flagged
    TN  honest sitting that was not flagged
    precision = TP / (TP + FP)       of the flagged sittings, how many were cheats
    recall    = TP / (TP + FN)       of the cheating sittings, how many were flagged
    FPR       = FP / (FP + TN)       of the honest sittings, how many were flagged
    F1        = 2TP / (2TP + FP + FN)

A rate whose denominator is 0 is NaN (undefined), never 0 or 1.

Per-cheat-type rows are computed on "all honest sittings + only the sittings of
that one cheat type". Recall and FPR mean what they say. Precision there depends on
that smaller mix, so it is not comparable with the overall precision.
"""
import numpy as np
import pandas as pd

from src.generator import CHEAT_TYPES, NONE_TYPE

TABLE_COLUMNS = [
    "group", "n_sittings", "n_honest", "n_cheat",
    "tp", "fp", "fn", "tn", "precision", "recall", "fpr", "f1",
]


def confusion_counts(is_cheat, flagged):
    """Return (tp, fp, fn, tn) from two parallel true/false (or 0/1) sequences."""
    is_cheat = np.asarray(is_cheat).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int((is_cheat & flagged).sum())
    fp = int((~is_cheat & flagged).sum())
    fn = int((is_cheat & ~flagged).sum())
    tn = int((~is_cheat & ~flagged).sum())
    return tp, fp, fn, tn


def divide(numerator, denominator):
    """numerator / denominator, or NaN when the denominator is 0."""
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def rates_from_counts(tp, fp, fn, tn):
    return {
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "fpr": divide(fp, fp + tn),
        "f1": divide(2 * tp, 2 * tp + fp + fn),
    }


def evaluate_group(group, is_cheat, flagged):
    """One row of the metrics table for one group of sittings."""
    tp, fp, fn, tn = confusion_counts(is_cheat, flagged)
    row = {
        "group": group,
        "n_sittings": tp + fp + fn + tn,
        "n_honest": fp + tn,
        "n_cheat": tp + fn,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
    row.update(rates_from_counts(tp, fp, fn, tn))
    return row


def metrics_table(df, flags):
    """Metrics for the whole of df ('overall') and for each cheat type.

    df must have is_cheat and cheat_type. flags is a boolean Series indexed like df.
    """
    flagged = flags.loc[df.index].to_numpy(dtype=bool)
    is_honest = (df["cheat_type"] == NONE_TYPE).to_numpy()
    rows = [evaluate_group("overall", df["is_cheat"].to_numpy(), flagged)]
    for cheat_type in CHEAT_TYPES:
        keep = is_honest | (df["cheat_type"] == cheat_type).to_numpy()
        rows.append(evaluate_group(cheat_type, df["is_cheat"].to_numpy()[keep], flagged[keep]))
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)


def ratio_text(numerator, denominator):
    """'192/225 (85.3%)', with the raw counts beside the percentage."""
    if denominator == 0:
        return f"n/a ({numerator}/{denominator})"
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def display_table(table):
    """Human-readable version of a metrics table: counts beside percentages."""
    rows = []
    for r in table.itertuples(index=False):
        rows.append({
            "group": r.group,
            "honest": r.n_honest,
            "cheats": r.n_cheat,
            "TP": r.tp, "FP": r.fp, "FN": r.fn, "TN": r.tn,
            "precision": ratio_text(r.tp, r.tp + r.fp),
            "recall": ratio_text(r.tp, r.tp + r.fn),
            "FPR": ratio_text(r.fp, r.fp + r.tn),
            "F1": ratio_text(2 * r.tp, 2 * r.tp + r.fp + r.fn),
        })
    return pd.DataFrame(rows)


def write_csv(table, path):
    """Write a results table the same way on every OS (LF line endings, 4 decimals)."""
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.4f")
