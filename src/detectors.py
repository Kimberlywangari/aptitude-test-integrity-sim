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
import pandas as pd

from src.generator import TIME_PREFIX

TARGET_FPR_PER_RULE = 0.01
Z_THRESHOLD = 1.5  # an answer is "unusually fast" if its z-score is below -Z_THRESHOLD


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


# ---------------------------------------------------------------- improved


@dataclass
class ImprovedThresholds:
    question_mean: pd.Series  # per-question mean seconds, from honest reference sittings
    question_std: pd.Series   # per-question std of seconds, from honest reference sittings
    z_threshold: float        # answer is "unusually fast" if z < -z_threshold
    min_fast_answers: int     # K: flag if at least this many answers are unusually fast
    max_focus_loss: int       # Y: same focus-loss rule as the baseline


def fit_question_stats(reference_honest):
    """Mean and standard deviation of seconds for each question (population std)."""
    columns = question_columns(reference_honest)
    means = reference_honest[columns].mean()
    stds = reference_honest[columns].std(ddof=0)
    return means, stds


def timing_z_scores(df, question_mean, question_std):
    """z = (seconds - that question's reference mean) / that question's reference std.

    A question with zero spread in the reference has no meaningful z-score, so it
    becomes NaN and is never counted as unusually fast.
    """
    columns = question_columns(df)
    std = question_std.replace(0, np.nan)
    return (df[columns] - question_mean) / std


def count_fast_answers(df, question_mean, question_std, z_threshold):
    """Number of answers per sitting with z below -z_threshold."""
    z = timing_z_scores(df, question_mean, question_std)
    return (z < -z_threshold).sum(axis=1)


def fit_min_fast_answers(fast_counts, target_fpr=TARGET_FPR_PER_RULE):
    """Smallest K such that at most target_fpr of the given (honest) sittings reach K."""
    k = 1
    while (fast_counts >= k).mean() > target_fpr:
        k += 1
    return k


def fit_improved(reference_honest, z_threshold=Z_THRESHOLD, target_fpr=TARGET_FPR_PER_RULE):
    """Learn per-question stats, K and Y from honest reference sittings only."""
    means, stds = fit_question_stats(reference_honest)
    fast_counts = count_fast_answers(reference_honest, means, stds, z_threshold)
    min_fast_answers = fit_min_fast_answers(fast_counts, target_fpr)
    max_focus_loss = fit_focus_threshold(reference_honest["focus_loss_events"], target_fpr)
    return ImprovedThresholds(means, stds, z_threshold, min_fast_answers, max_focus_loss)


def flag_many_fast_answers(df, thresholds):
    """Rule: at least K answers are unusually fast for their own question."""
    fast_counts = count_fast_answers(
        df, thresholds.question_mean, thresholds.question_std, thresholds.z_threshold
    )
    return fast_counts >= thresholds.min_fast_answers


def improved_flags(df, thresholds):
    """Improved detector: per-question timing rule OR the same focus-loss rule."""
    many_fast = flag_many_fast_answers(df, thresholds)
    high_focus = flag_high_focus_loss(df, thresholds.max_focus_loss)
    return many_fast | high_focus
