"""Create a reproducible, clearly labelled survival-analysis demonstration CSV.

All predictors are sampled before the event time. No outcome-derived column is
included among the model inputs. The deliberately strong signal makes it easy
to demonstrate discrimination and separated survival curves; it is not a
clinical performance benchmark.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20261116
ROWS = 320
OUTPUT = Path(__file__).resolve().parents[1] / "demo_data" / "synthetic_survival_demo.csv"


def generate(seed: int = SEED, rows: int = ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stage = rng.choice([1, 2, 3, 4], size=rows, p=[0.23, 0.28, 0.28, 0.21])
    age = np.clip(rng.normal(61, 10, rows), 35, 85).round().astype(int)
    tumor_size = np.clip(rng.normal(3.1 + 0.65 * (stage - 2), 0.65, rows), 0.5, 8.0)
    biomarker = np.clip(rng.normal(48 + 12 * (stage - 2), 12, rows), 5, 110)
    treatment = rng.binomial(1, 0.55, rows)

    # Latent risk is built only from baseline predictors. A modest independent
    # noise term prevents the outcomes from being a deterministic lookup.
    risk = (
        0.78 * (stage - 2.5)
        + 0.63 * ((biomarker - 48) / 20)
        + 0.43 * (tumor_size - 3.1)
        + 0.22 * ((age - 61) / 10)
        - 0.60 * treatment
    )
    event_time = np.maximum(0.5, 27 * np.exp(-0.57 * risk + rng.normal(0, 0.28, rows)))
    censor_time = rng.uniform(28, 78, rows)
    duration = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)

    return pd.DataFrame({
        "duration": duration.round(2),
        "event": event,
        "age": age,
        "stage": stage,
        "tumor_size_cm": tumor_size.round(2),
        "biomarker_level": biomarker.round(1),
        "received_treatment": treatment,
    })


if __name__ == "__main__":
    data = generate()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}: {len(data)} rows, {int(data.event.sum())} events")
