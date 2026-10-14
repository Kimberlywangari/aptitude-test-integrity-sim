"""One place that builds the dataset every script and test uses."""
from src.generator import generate_sittings
from src.split import assign_split


def build_dataset():
    """Generate the simulated sittings and attach the fixed reference/test split.

    Uses the default seeds, so every call returns the identical table.
    """
    return assign_split(generate_sittings())
