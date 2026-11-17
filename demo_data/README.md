# Synthetic survival demonstration dataset

`synthetic_survival_demo.csv` is an upload-ready CSV with **320 simulated records**, generated reproducibly by `scripts/generate_synthetic_survival.py` with seed `20261116`. These are fictional records for demonstrating the app and its figures.

| Column | Meaning |
| --- | --- |
| `duration` | Observed follow-up time in simulated months, rounded to 2 decimals. |
| `event` | 1 if the simulated event was observed; 0 if the record was right-censored. |
| `age` | Simulated age in years. |
| `stage` | Simulated baseline stage from 1 through 4. |
| `tumor_size_cm` | Simulated baseline tumor size in centimetres. |
| `biomarker_level` | Simulated baseline biomarker measurement in arbitrary units. |
| `received_treatment` | Simulated treatment indicator, 0 or 1. |

The five predictor columns are created **before** the event time. A latent risk score combines them; event time is then drawn from a log-normal process with independent random variation. Censoring time is sampled independently, between 28 and 78 months. The generator deliberately makes the predictor–outcome relationship clear, so models can display strong discrimination and visibly separated survival curves. No event time, observed duration, censoring indicator, or score derived from them is included as a predictor.

This CSV has **225 observed events and 95 censored records**. Its C-index is a demonstration of the software on constructed data, not an estimate of clinical performance. Train, validation, and test results vary with the model and training settings.

In **Model Training**, click **Load synthetic survival demo**. The same file can be uploaded from **Chat with Agent**, or downloaded in the app using **Download synthetic demo CSV**.
