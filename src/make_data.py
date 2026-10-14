"""Write the simulated data to disk.

Run from the repo root:  python -m src.make_data

data/sittings_full.csv   all 5,000 sittings (gitignored, regenerates from the seed)
data/sample_sittings.csv small illustrative sample (committed)
"""
from pathlib import Path

import pandas as pd

from src.dataset import build_dataset
from src.generator import NONE_TYPE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_HONEST = 40
SAMPLE_PER_CHEAT_TYPE = 10


def make_sample(df):
    """Take the first few rows of each group. Meant for inspection, not statistics:
    cheats are over-represented compared with the full data set."""
    honest = df[df["cheat_type"] == NONE_TYPE].head(SAMPLE_HONEST)
    cheats = df[df["cheat_type"] != NONE_TYPE]
    cheats = cheats.groupby("cheat_type").head(SAMPLE_PER_CHEAT_TYPE)
    return pd.concat([honest, cheats]).sort_values("sitting_id")


def main():
    df = build_dataset()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "sittings_full.csv", index=False, lineterminator="\n")
    sample = make_sample(df)
    sample.to_csv(DATA_DIR / "sample_sittings.csv", index=False, lineterminator="\n")
    print(f"wrote {len(df)} sittings to data/sittings_full.csv")
    print(f"wrote {len(sample)} sittings to data/sample_sittings.csv")
    print(df.groupby(["split", "cheat_type"]).size().unstack(0).to_string())


if __name__ == "__main__":
    main()
