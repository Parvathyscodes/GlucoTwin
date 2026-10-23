# GlucoTwin 20-Minute Demo Script

**Before recording:** Run the tests, launch the app, confirm the screen is readable, and ensure no personal information is visible. Use the exact submitted code and report. Do not claim clinical validation.

## 0:00–2:00 — Introduction
“GlucoTwin is a proof of concept exploring how static patient history and changing physiological readings can be combined in a patient-specific digital representation. Our chosen demonstrator is short-horizon glucose-excursion risk in a synthetic cohort.”

## 2:00–4:00 — Define the use case
Explain the two-hour horizon and illustrative 180 mg/dL target. Explicitly say this is an engineering label for the demo, not an emergency definition or medical recommendation.

## 4:00–6:00 — Data sources and limitations
Show the profile fields and time-series fields. Explain that the current reproducible demo uses synthetic data generated in code, with no real patient records. Describe the plan for a licensed open dataset and careful data integration.

## 6:00–9:00 — Architecture
Walk through `docs/architecture.md`: data streams, join, past-window features, patient-level split, model, evaluation and dashboard.

## 9:00–12:00 — Model and evaluation
Show `src/data_pipeline.py`, `src/modeling.py` and `train_model.py`. Explain why patient-level splitting helps reduce leakage. Show the generated metrics and explain that synthetic metrics are not clinical performance estimates.

## 12:00–16:00 — Live application
Select a synthetic patient. Show the profile, timeline, supporting signals and model-estimated event probability. Move the replay slider and point out how the latest state and estimate change.

## 16:00–18:00 — Responsible use
Discuss synthetic-data limitations, lack of external validation, missing-data considerations and why this prototype must not guide treatment.

## 18:00–20:00 — Roadmap and close
Outline the next steps: integrate a suitable licensed public time-series dataset, document provenance, compare against a strong baseline, validate across held-out patients, assess calibration and subgroup performance, and seek clinical review before any real-world use.
