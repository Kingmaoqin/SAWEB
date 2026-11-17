# Synthetic survival demonstration dataset

`synthetic_survival_demo.csv` contains 160 simulated patient records generated with seed `20261116`. It is for interface testing and demonstration; no record represents a real patient.

- `duration`: observed follow-up duration in simulated months (minimum 0.2).
- `event`: 1 for an observed event before censoring; 0 for right-censoring.
- `age`, `stage`, `tumor_size_cm`, `biomarker_level`, `received_treatment`: numerical model inputs. Treatment is 0 or 1.

Event times are drawn from an exponential distribution whose rate depends on the five inputs. Independent censoring times are drawn uniformly from 14 to 62 months. The CSV has 103 observed events and 57 censored records. Model metrics on this small synthetic sample are demonstration outputs, not clinical validation.

In **Model Training**, click **Load synthetic survival demo**. The same CSV is available through **Download synthetic demo CSV** for use on the Chat page.
