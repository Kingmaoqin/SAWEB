"""Generate a full multimodal toy dataset for local MySA experiments.

The script writes a directory (``--output``) containing artefacts ready for the
Streamlit workbench:

* ``tabular.csv`` – core clinical table with ``id``, ``duration`` and ``event``.
* ``image.csv`` – pre-computed embedding features keyed by ``id``.
* ``sensor.csv`` – aggregated sensor statistics keyed by ``id``.
* ``images/`` – RGB PNG files, plus ``image_manifest.csv`` for the image wizard.
* ``sensor_sequences/`` – per-subject time-series CSVs, plus
  ``sensor_manifest.csv`` and a zipped bundle ``sensor_sequences.zip`` for the
  sensor wizard.

Example::

    python simulate_multimodal_data.py --samples 200 --seed 13
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import zipfile

try:  # Pillow is only needed when actually writing image files.
    from PIL import Image
except Exception as exc:  # pragma: no cover - CLI guard
    raise RuntimeError(
        "Pillow is required to generate synthetic image files. "
        "Install it via `pip install pillow`."
    ) from exc


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
            "id": ids,
            "duration": observed.round(2),
            "event": events,
            "age": age,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "cholesterol": cholesterol,
            "diabetes": diabetes,
        }
    )


def _make_rgb_image(
    base_mean: float,
    rng: np.random.Generator,
    image_size: Tuple[int, int],
) -> np.ndarray:
    h, w = image_size
    pattern = rng.normal(loc=base_mean, scale=0.08, size=(h, w, 3))
    # Add a smooth radial gradient so that embeddings vary smoothly with base_mean.
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    radius = np.sqrt(xx**2 + yy**2)
    pattern += 0.15 * np.cos(3 * math.pi * radius)[..., None]
    pattern = np.clip(pattern, 0.0, 1.0)
    return (pattern * 255).astype(np.uint8)


def make_image_assets(
    tabular: pd.DataFrame,
    rng: np.random.Generator,
    out_dir: Path,
    *,
    image_dim: int,
    image_size: Tuple[int, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic PNG files and deterministic embeddings."""

    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_size
    # Global random projection to derive embeddings reproducibly from raw pixels.
    proj = rng.normal(0, 1, size=(image_dim, h * w * 3))

    feat_rows = []
    manifest_rows = []

    for row in tabular.itertuples(index=False):
        base_mean = 0.45 + 0.002 * (float(row.age) - 60.0) + 0.05 * float(row.diabetes)
        arr = _make_rgb_image(base_mean, rng, image_size)
        fname = f"{row.id}.png"
        Image.fromarray(arr).save(img_dir / fname)

        flat = arr.astype(np.float32).reshape(-1) / 255.0
        embedding = proj @ flat
        feat_rows.append([row.id, *embedding.tolist()])
        manifest_rows.append(
            {
                "id": row.id,
                "image": f"images/{fname}",
                "duration": row.duration,
                "event": row.event,
            }
        )

    cols = ["id"] + [f"img_feat_{i:03d}" for i in range(image_dim)]
    feat_df = pd.DataFrame(feat_rows, columns=cols)
    manifest_df = pd.DataFrame(manifest_rows)
    return feat_df, manifest_df


def make_sensor_assets(
    tabular: pd.DataFrame,
    rng: np.random.Generator,
    out_dir: Path,
    *,
    sequence_length: int,
    sampling_rate: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic sensor sequences + aggregated statistics."""

    sens_dir = out_dir / "sensor_sequences"
    sens_dir.mkdir(parents=True, exist_ok=True)

    time = np.arange(sequence_length) / max(sampling_rate, 1.0)

    feat_rows = []
    manifest_rows = []

    for row in tabular.itertuples(index=False):
        base_hr = 72 + 6 * float(row.diabetes) + 0.2 * (float(row.bmi) - 25)
        hr = base_hr + 3 * np.sin(2 * np.pi * time / 24) + rng.normal(0, 2, size=sequence_length)

        activity = rng.normal(0, 1, size=(sequence_length, 3))
        # Modulate activity by age (older → lower magnitude) and event occurrence (event → more variation).
        activity *= 1.2 - 0.004 * (float(row.age) - 60)
        if int(row.event) == 1:
            activity += rng.normal(0, 0.5, size=(sequence_length, 3))

        steps = np.maximum(0.0, rng.normal(0.6, 0.2, size=sequence_length))
        sleep = np.clip(
            7.5 - 0.03 * (float(row.bmi) - 25) - 0.5 * float(row.diabetes) + rng.normal(0, 0.3),
            4.5,
            9.0,
        )

        seq_df = pd.DataFrame(
            {
                "time": time,
                "accel_x": activity[:, 0],
                "accel_y": activity[:, 1],
                "accel_z": activity[:, 2],
                "heart_rate": hr,
                "step_rate": steps,
            }
        )
        fname = f"{row.id}.csv"
        seq_df.to_csv(sens_dir / fname, index=False)

        accel_mag = np.linalg.norm(activity, axis=1)
        feat_rows.append(
            {
                "id": row.id,
                "accel_mean": float(np.mean(accel_mag)),
                "accel_std": float(np.std(accel_mag, ddof=0)),
                "hr_mean": float(np.mean(hr)),
                "hr_std": float(np.std(hr, ddof=0)),
                "step_count": float(np.sum(steps) * (24 / sequence_length)),
                "sleep_hours": float(sleep),
            }
        )
        manifest_rows.append(
            {
                "id": row.id,
                "file": f"sensor_sequences/{fname}",
                "duration": row.duration,
                "event": row.event,
            }
        )

    feat_df = pd.DataFrame(feat_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    return feat_df, manifest_df


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    ids = pd.Series([f"PT_{i:04d}" for i in range(args.samples)], name="id")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    tab = make_tabular(ids, rng)
    img_feats, img_manifest = make_image_assets(
        tab,
        rng,
        out_dir,
        image_dim=args.image_dim,
        image_size=(args.image_size, args.image_size),
    )
    sens_feats, sens_manifest = make_sensor_assets(
        tab,
        rng,
        out_dir,
        sequence_length=args.sensor_length,
        sampling_rate=args.sensor_rate,
    )

    tab.to_csv(out_dir / "tabular.csv", index=False)
    img_feats.to_csv(out_dir / "image.csv", index=False)
    sens_feats.to_csv(out_dir / "sensor.csv", index=False)
    img_manifest.to_csv(out_dir / "image_manifest.csv", index=False)
    sens_manifest.to_csv(out_dir / "sensor_manifest.csv", index=False)

    # Also ship a zipped version of the sensor sequences for convenience.
    zip_path = out_dir / "sensor_sequences.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in (out_dir / "sensor_sequences").glob("*.csv"):
            zf.write(csv_path, arcname=csv_path.name)

    print(f"Saved tabular data to {out_dir / 'tabular.csv'}")
    print(f"Saved image embeddings to {out_dir / 'image.csv'} (raw files in images/)")
    print(
        "Saved sensor features to {0} (time-series in sensor_sequences/ and zipped at {1})"
        .format(out_dir / "sensor.csv", zip_path)
    )
    print(f"Image manifest written to {out_dir / 'image_manifest.csv'}")
    print(f"Sensor manifest written to {out_dir / 'sensor_manifest.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic multimodal survival data for MySA testing.")
    parser.add_argument("--samples", type=int, default=200, help="Number of patients to simulate")
    parser.add_argument("--image-dim", type=int, default=128, help="Dimensionality of simulated image embeddings")
    parser.add_argument("--image-size", type=int, default=64, help="Height/width of generated square images in pixels")
    parser.add_argument("--sensor-length", type=int, default=512, help="Number of timesteps per synthetic sensor recording")
    parser.add_argument("--sensor-rate", type=float, default=1.0, help="Sampling rate (Hz) for synthetic sensors")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="synthetic_multimodal", help="Output directory")
    main(parser.parse_args())
