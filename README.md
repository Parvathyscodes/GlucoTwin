# GlucoTwin: A Digital Twin Proof of Concept for Short-Horizon Glucose Excursion Risk

**Status:** MVP scaffold and synthetic-data demonstration. Not clinically validated.

## Team and institution

- Team name: parvathy_b241057me
- Team leader: PARVATHY S
- College/incubator: National Institute of Technology, Calicut
- Team members: Individual entry — confirm eligibility and registration rules before submission.

> Replace all bracketed fields before publishing. Do not publish private phone numbers in the public README unless the challenge explicitly requires it. The submission portal may require those details separately.

## Problem statement

Can a patient-specific computational representation combine static medical history with a changing stream of physiological measurements to estimate the likelihood of a defined near-term glucose excursion?

GlucoTwin demonstrates an end-to-end software pattern: synthetic EHR-like profile + synthetic wearable/CGM-like time series → patient-state feature update → binary machine-learning estimate → clinician-facing dashboard.

## Healthcare use case

The prototype illustrates short-horizon prediction of whether the maximum glucose value in the next two hours reaches an illustrative threshold of 180 mg/dL. This threshold is used only to define a demonstrable engineering target in synthetic data; it is not presented as a clinical emergency threshold or treatment instruction.

## What is implemented

- Reproducible synthetic cohort generator with static and dynamic streams linked by synthetic patient IDs.
- Feature engineering from the past two hours of readings.
- Patient-level train/test split to reduce leakage across patients.
- Logistic regression classifier and a simple heuristic comparator.
- Evaluation metrics including ROC-AUC, precision, recall, F1 and Brier score where computable.
- Streamlit dashboard with patient selector, time-series replay slider, patient profile, trend charts, model estimate, and data/limitation disclosures.
- Tests for data generation, label range and patient-level split integrity.

## Important data disclosure

The default app generates **synthetic demonstration data only**. No real patient data is included. The generator is intentionally simple and is not a physiologically validated simulator. Metrics measured on synthetic data are software-pipeline checks only and must not be represented as evidence of clinical accuracy.

The challenge asks for static/historical data and dynamic sensor data. This prototype represents those two streams synthetically and links them by a synthetic patient ID. For a stronger competition submission, replace or augment this demonstration with a suitable, licensed open-source time-series dataset and clearly document how the static profile is sourced or simulated. Do not falsely claim that independently collected public records belong to the same person.

## Technical stack

- Python
- Pandas and NumPy
- scikit-learn
- Streamlit and Plotly
- pytest
- GitHub

## PRESENTATION AND ARCHITECTURE DIAGRAM ARE UPLOADED UNDER THE DOCS FILE AS: 

- GlucoTwin_Challenge_Presentation
- architecture_diagram


## Run locally

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows
pip install -r requirements.txt
python generate_demo_data.py
python train_model.py
pytest -q
streamlit run app.py
```

The Streamlit app also generates the synthetic data and trains its model on first launch, so `generate_demo_data.py` and `train_model.py` are optional for running the app.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) and [`docs/architecture_diagram.pdf`](docs/architecture_diagram.pdf).

## Model and evaluation

The current classifier is logistic regression with median imputation and feature scaling, without class weighting. An earlier class-weighted version was retained as a baseline for comparison.

Patients—not individual rows—are split between train and test sets. This is a more appropriate first check than randomly splitting overlapping time windows from the same patients.

The app reports test-set metrics on the synthetic cohort. These metrics can be unstable or undefined if a test split contains only one class. The simple heuristic comparator is included for transparency; it is not a clinical baseline. No performance numbers are asserted in this README because they should be generated and reviewed from the exact submitted run.

## Limitations and responsible use

- This is a research/educational proof of concept, not a medical device.
- It must not be used for diagnosis, treatment decisions, or personal health advice.
- Synthetic data does not represent the full diversity or physiology of real people.
- The target threshold and prediction horizon are engineering choices that require clinical review before real-world use.
- External validation, calibration analysis, subgroup evaluation, missing-data robustness, and clinician input are not yet established.
- Real-data use requires dataset license compliance, appropriate access approvals, provenance documentation, and correct attribution.
- No real-world clinical benefit is claimed.

## Reproducibility

The synthetic generator uses a fixed random seed. Record Python and package versions for the final submission, rerun tests, and attach the actual output of `python train_model.py` to the final results only after review.

## Open-source license

This repository is licensed under the MIT License. This software license does not change the license terms of any third-party datasets added later.
