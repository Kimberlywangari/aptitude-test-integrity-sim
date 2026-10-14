"""Shared setup for the evaluation scripts, so every detector sees the same data."""
import json
from pathlib import Path

from src.dataset import build_dataset
from src.generator import CHEAT_RATE, N_QUESTIONS, N_SITTINGS, SEED
from src.split import SPLIT_SEED, select_reference_honest, select_test

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_reference_and_test():
    """Return (honest reference sittings, test sittings) from the fixed split."""
    df = build_dataset()
    return select_reference_honest(df), select_test(df)


def data_summary(reference_honest, test):
    """Sizes and seeds of the run, recorded next to every set of results."""
    return {
        "generator_seed": SEED,
        "split_seed": SPLIT_SEED,
        "n_sittings": N_SITTINGS,
        "n_questions": N_QUESTIONS,
        "cheat_rate": CHEAT_RATE,
        "n_reference_honest": int(len(reference_honest)),
        "n_test": int(len(test)),
        "n_test_honest": int((test["is_cheat"] == 0).sum()),
        "n_test_cheat": int(test["is_cheat"].sum()),
    }


def write_json(obj, path):
    """Write JSON with LF line endings so files are identical on every OS."""
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", newline="\n")
