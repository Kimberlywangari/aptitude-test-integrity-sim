"""Seeded generator for SIMULATED test sittings.

Each sitting is one invented candidate taking a 40-question multiple-choice test.
Only two signals are simulated: seconds spent on each question, and how many times
the browser window lost focus (tab switch / window blur). There are no real
candidates and no real data anywhere in this project.

All behaviour parameters below were fixed once, before any detector was written or
run, and were not adjusted afterwards to make a detector look better.
"""
import numpy as np
import pandas as pd

SEED = 42
N_SITTINGS = 5000
N_QUESTIONS = 40
CHEAT_RATE = 0.15

# Share of all cheats given to each injected type (must sum to 1).
CHEAT_TYPE_SHARES = {
    "too_fast": 0.30,
    "idle_burst": 0.30,
    "focus_loss": 0.20,
    "uniform_timing": 0.20,
}
CHEAT_TYPES = list(CHEAT_TYPE_SHARES)

TIME_PREFIX = "t_q"      # time columns are t_q01 ... t_q40
NONE_TYPE = "none"       # cheat_type value for honest sittings
MIN_SECONDS = 2.0        # nobody reads and answers a question faster than this

# Honest behaviour
QUESTION_MEDIAN_RANGE = (20.0, 90.0)  # typical seconds per question
SPEED_SIGMA = 0.30       # spread of candidates' overall pace (log scale)
NOISE_SIGMA = 0.35       # question-to-question wobble within a sitting (log scale)
FOCUS_RATE_SHAPE = 2.0   # honest focus-loss rate ~ Gamma(shape, scale), mean 1.0
FOCUS_RATE_SCALE = 0.5
DISTRACTED_RATE = 0.08   # honest sittings with one long interruption
PAUSE_SECONDS = (60.0, 240.0)
RUSHED_RATE = 0.10       # honest sittings that rush the last few questions
RUSH_QUESTIONS = (3, 8)
RUSH_FACTOR = (0.35, 0.70)

# Injected cheat behaviour
TOO_FAST_SPEED = (0.15, 0.45)     # pace as a fraction of normal
BURST_QUESTIONS = (10, 20)        # how many final questions are clicked through
BURST_SECONDS = (3.0, 8.0)        # seconds per answer inside the burst
IDLE_GAP_SECONDS = (120.0, 300.0) # extra idle time just before the burst
FOCUS_CHEAT_EVENTS = (4, 15)      # focus-loss events for a repeated-focus-loss cheat
UNIFORM_SECONDS = (15.0, 45.0)    # constant pace of a uniform/robotic cheat
UNIFORM_JITTER = 1.0              # std of seconds around that pace


def time_columns(n_questions=N_QUESTIONS):
    """Names of the per-question time columns: t_q01, t_q02, ..."""
    return [f"{TIME_PREFIX}{j:02d}" for j in range(1, n_questions + 1)]


def _cheat_counts(n_sittings, cheat_rate):
    """Exact number of sittings per cheat type (last type absorbs rounding)."""
    n_cheats = int(round(n_sittings * cheat_rate))
    counts = {}
    used = 0
    for cheat_type in CHEAT_TYPES[:-1]:
        counts[cheat_type] = int(round(n_cheats * CHEAT_TYPE_SHARES[cheat_type]))
        used += counts[cheat_type]
    counts[CHEAT_TYPES[-1]] = n_cheats - used
    return counts


def _apply_honest_quirks(row, rng):
    """Add ordinary human messiness to one honest sitting's times (edits in place)."""
    # Distracted: one question gets a long pause (an interruption), no burst after it.
    if rng.random() < DISTRACTED_RATE:
        question = rng.integers(0, len(row))
        row[question] += rng.uniform(*PAUSE_SECONDS)
    # Rushed: the candidate ran short of time and hurried the last few questions.
    if rng.random() < RUSHED_RATE:
        n_rushed = rng.integers(RUSH_QUESTIONS[0], RUSH_QUESTIONS[1] + 1)
        row[-n_rushed:] *= rng.uniform(*RUSH_FACTOR, size=n_rushed)


def generate_sittings(n_sittings=N_SITTINGS, seed=SEED, n_questions=N_QUESTIONS,
                      cheat_rate=CHEAT_RATE):
    """Return a DataFrame with one row per simulated sitting.

    Columns: sitting_id, is_cheat (0/1), cheat_type, focus_loss_events,
    then one seconds-spent column per question. The same seed always gives the
    same table.
    """
    rng = np.random.default_rng(seed)

    # Each question has a typical time (an easy item ~20s, a hard one ~90s).
    question_median = rng.uniform(*QUESTION_MEDIAN_RANGE, size=n_questions)

    # Ground truth first: exact counts per type, then shuffled across sittings.
    counts = _cheat_counts(n_sittings, cheat_rate)
    types = [NONE_TYPE] * (n_sittings - sum(counts.values()))
    for cheat_type, count in counts.items():
        types += [cheat_type] * count
    cheat_type = np.array(types)
    rng.shuffle(cheat_type)

    # Honest timing for every sitting: question typical time x candidate pace x wobble.
    speed = rng.lognormal(0.0, SPEED_SIGMA, size=(n_sittings, 1))
    noise = rng.lognormal(0.0, NOISE_SIGMA, size=(n_sittings, n_questions))
    times = question_median * speed * noise

    # Honest focus loss: each candidate has their own rate, count is Poisson at that rate.
    focus_rate = rng.gamma(FOCUS_RATE_SHAPE, FOCUS_RATE_SCALE, size=n_sittings)
    focus_loss = rng.poisson(focus_rate)

    # Now edit each sitting according to its ground-truth type.
    for i in range(n_sittings):
        kind = cheat_type[i]
        if kind == NONE_TYPE:
            _apply_honest_quirks(times[i], rng)
        elif kind == "too_fast":
            # Answers everything at 15-45% of the normal pace (e.g. has the answers).
            fast_speed = rng.uniform(*TOO_FAST_SPEED)
            wobble = rng.lognormal(0.0, NOISE_SIGMA, size=n_questions)
            times[i] = question_median * fast_speed * wobble
        elif kind == "idle_burst":
            # Normal pace at first, then an idle gap, then a rapid burst to the end.
            burst_len = rng.integers(BURST_QUESTIONS[0], BURST_QUESTIONS[1] + 1)
            burst_len = min(burst_len, n_questions - 1)
            times[i, -burst_len:] = rng.uniform(*BURST_SECONDS, size=burst_len)
            times[i, -burst_len - 1] += rng.uniform(*IDLE_GAP_SECONDS)
        elif kind == "focus_loss":
            # Timing stays honest-looking, but the window loses focus many times.
            focus_loss[i] = rng.integers(FOCUS_CHEAT_EVENTS[0], FOCUS_CHEAT_EVENTS[1] + 1)
        elif kind == "uniform_timing":
            # Every answer takes almost the same time, whatever the question.
            pace = rng.uniform(*UNIFORM_SECONDS)
            times[i] = pace + rng.normal(0.0, UNIFORM_JITTER, size=n_questions)

    times = np.clip(times, MIN_SECONDS, None).round(1)

    df = pd.DataFrame(times, columns=time_columns(n_questions))
    df.insert(0, "focus_loss_events", focus_loss)
    df.insert(0, "cheat_type", cheat_type)
    df.insert(0, "is_cheat", (cheat_type != NONE_TYPE).astype(int))
    df.insert(0, "sitting_id", [f"S{i:05d}" for i in range(1, n_sittings + 1)])
    return df
