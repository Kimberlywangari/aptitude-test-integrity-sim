"""Cheat-flagging detectors for simulated sittings.

A flag means "worth a human look". Nothing here rejects anyone.

Detectors never read the ground-truth columns (is_cheat, cheat_type). Everything a
detector learns comes from the honest sittings of the reference half (see split.py).

Every rule gets the same false-alarm budget: its threshold is set so that at most
TARGET_FPR_PER_RULE of the honest REFERENCE sittings would trip that rule. That keeps
comparisons between detectors about detection power, not about who is looser.
"""
from dataclasses import dataclass

import numpy as np

from src.generator import TIME_PREFIX

TARGET_FPR_PER_RULE = 0.01


def question_columns(df):
    """The per-question time columns present in df."""
    return [c for c in df.columns if c.startswith(TIME_PREFIX)]


def average_time(df):
    """Average seconds per question, one value per sitting."""
    return df[question_columns(df)].mean(axis=1)


# ---------------------------------------------------------------- baseline


@dataclass
class BaselineThresholds:
    min_avg_seconds: float  # X: flag if average seconds per question is below this
    max_focus_loss: int     # Y: flag if focus-loss events exceed this


def fit_focus_threshold(focus_events, target_fpr=TARGET_FPR_PER_RULE):
    """Smallest Y such that at most target_fpr of the given (honest) sittings exceed Y."""
    y = 0
    while (focus_events > y).mean() > target_fpr:
        y += 1
    return y


def fit_baseline(reference_honest, target_fpr=TARGET_FPR_PER_RULE):
    """Set X and Y from honest reference sittings only."""
    min_avg_seconds = float(np.quantile(average_time(reference_honest), target_fpr))
    max_focus_loss = fit_focus_threshold(reference_honest["focus_loss_events"], target_fpr)
    return BaselineThresholds(min_avg_seconds, max_focus_loss)


def flag_low_average_time(df, min_avg_seconds):
    """Rule 1: average time per question below X seconds."""
    return average_time(df) < min_avg_seconds


def flag_high_focus_loss(df, max_focus_loss):
    """Rule 2: more than Y focus-loss events."""
    return df["focus_loss_events"] > max_focus_loss


def baseline_flags(df, thresholds):
    """Baseline detector: flag if either fixed-threshold rule fires."""
    low_time = flag_low_average_time(df, thresholds.min_avg_seconds)
    high_focus = flag_high_focus_loss(df, thresholds.max_focus_loss)
    return low_time | high_focus
