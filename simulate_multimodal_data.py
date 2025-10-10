"""Generate synthetic multimodal survival-analysis data for local testing.

This script produces three CSV files under ``synthetic_multimodal/``:
    - ``tabular.csv``: duration/event labels + tabular features.
    - ``image.csv``: image embedding features keyed by ``patient_id``.
    - ``sensor.csv``: sensor summary features keyed by ``patient_id``.

Example::
    python simulate_multimodal_data.py --samples 200 --seed 13
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def make_tabular(ids: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    age = rng.normal(62, 10, size=n)
    bmi = rng.normal(26, 4, size=n)
    systolic_bp = rng.normal(130, 15, size=n)
    cholesterol = rng.normal(190, 35, size=n)
    diabetes = rng.binomial(1, 0.25, size=n)

    # survival labels
    baseline_hazard = 0.03 + 0.002 * (age - 60) + 0.003 * diabetes
    durations = rng.exponential(scale=1.0 / np.clip(baseline_hazard, 1e-3, None))
    durations = np.clip(durations, 0.1, None)
    censoring = rng.exponential(scale=20.0, size=n)
    observed = np.minimum(durations, censoring)
    events = (durations <= censoring).astype(int)

    return pd.DataFrame(
        {
            "patient_id": ids,
            "duration": observed.round(2),
            "event": events,
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "diabetes": diabetes,
        }
    )


def make_embedding(ids: pd.Series, dim: int, prefix: str, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    mat = rng.normal(0, 1, size=(n, dim))
    cols = [f"{prefix}{i:03d}" for i in range(dim)]
    df = pd.DataFrame(mat, columns=cols)
    df.insert(0, "patient_id", ids)
    return df


def make_sensor(ids: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    n = len(ids)
    feats = {
        "patient_id": ids,
        "accel_mean": rng.normal(0, 1, size=n),
        "accel_std": rng.gamma(2.0, 0.5, size=n),
        "hr_mean": rng.normal(72, 8, size=n),
        "hr_std": rng.gamma(2.0, 1.0, size=n),
        "step_count": rng.integers(2000, 12000, size=n),
        "sleep_hours": rng.normal(6.5, 1.0, size=n),
    }
    return pd.DataFrame(feats)


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    ids = pd.Series([f"PT_{i:04d}" for i in range(args.samples)], name="patient_id")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tab = make_tabular(ids, rng)
    img = make_embedding(ids, dim=args.image_dim, prefix="img_f", rng=rng)
    sens = make_sensor(ids, rng)

    tab.to_csv(out_dir / "tabular.csv", index=False)
    img.to_csv(out_dir / "image.csv", index=False)
    sens.to_csv(out_dir / "sensor.csv", index=False)

    print(f"Saved tabular data to {out_dir / 'tabular.csv'}")
    print(f"Saved image embeddings to {out_dir / 'image.csv'}")
    print(f"Saved sensor features to {out_dir / 'sensor.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal survival data for MySA testing.")
    parser.add_argument("--samples", type=int, default=200, help="Number of patients to simulate")
    parser.add_argument("--image-dim", type=int, default=128, help="Dimensionality of simulated image embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="synthetic_multimodal", help="Output directory")
    main(parser.parse_args())
