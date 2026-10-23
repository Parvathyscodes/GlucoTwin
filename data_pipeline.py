"""Synthetic, reproducible data generation for the GlucoTwin proof of concept.

IMPORTANT: The bundled generator creates illustrative synthetic data only.
It is not real patient data and must not be represented as clinical evidence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RANDOM_SEED = 42

def make_synthetic_cohort(n_patients: int = 80, days: int = 5,
                          interval_minutes: int = 15,
                          seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (patient_profiles, sensor_stream) with stable synthetic patient IDs.

    The dynamic stream contains glucose, steps, heart rate, sleep proxy and meal
    carbohydrate intake. The static profile contains demographics and baseline
    clinical characteristics. Values are simulated for software demonstration.
    """
    rng = np.random.default_rng(seed)
    profiles = []
    streams = []
    points_per_day = int(24 * 60 / interval_minutes)
    for p in range(n_patients):
        pid = f"SYN{p+1:04d}"
        age = int(rng.integers(35, 76))
        bmi = float(np.clip(rng.normal(29, 4.5), 19, 43))
        hba1c = float(np.clip(rng.normal(7.4 + (bmi-29)*0.025, 1.0), 5.7, 11.5))
        years_since_dx = int(rng.integers(0, 19))
        medication_flag = int(rng.random() < 0.68)
        baseline = 112 + (hba1c - 6.0) * 16 + (bmi - 25) * 0.9
        profiles.append({
            "patient_id": pid, "age_years": age, "bmi": round(bmi, 1),
            "hba1c_percent": round(hba1c, 1),
            "years_since_diagnosis": years_since_dx,
            "medication_recorded": medication_flag,
            "data_origin": "synthetic_demo"
        })
        n = points_per_day * days
        timestamps = pd.date_range("2026-01-01", periods=n, freq=f"{interval_minutes}min")
        glucose = np.zeros(n)
        steps = np.zeros(n)
        heart_rate = np.zeros(n)
        sleep = np.zeros(n)
        carbs = np.zeros(n)
        latent = baseline + rng.normal(0, 6)
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60
            # Synthetic routine: meal events and activity are simulated.
            meal_prob = 0.025
            meal = int(rng.random() < meal_prob)
            carbs[i] = float(rng.choice([20, 35, 50, 70])) if meal else 0.0
            steps[i] = max(0, rng.normal(55 if 7 <= hour <= 21 else 4, 35))
            heart_rate[i] = max(48, min(125, rng.normal(75 + steps[i]*0.08, 7)))
            sleep[i] = 1 if (hour >= 23 or hour < 7) else 0
            circadian = 4 * np.sin(2*np.pi*(hour-6)/24)
            # Meal effect decays over several intervals; not a physiological model.
            recent_carbs = carbs[max(0, i-8):i+1]
            meal_effect = float(np.sum(recent_carbs) * 0.30)
            activity_effect = -0.025 * steps[i]
            noise = rng.normal(0, 4.5)
            latent = 0.80*latent + 0.20*(baseline + circadian + meal_effect + activity_effect) + noise
            glucose[i] = np.clip(latent, 55, 320)
        frame = pd.DataFrame({
            "patient_id": pid, "timestamp": timestamps,
            "glucose_mg_dl": np.round(glucose, 1),
            "steps_15min": np.round(steps).astype(int),
            "heart_rate_bpm": np.round(heart_rate).astype(int),
            "sleep_proxy": sleep.astype(int),
            "carbs_g_simulated": np.round(carbs, 1),
            "data_origin": "synthetic_demo"
        })
        streams.append(frame)
    return pd.DataFrame(profiles), pd.concat(streams, ignore_index=True)

def make_supervised_table(profiles: pd.DataFrame, stream: pd.DataFrame,
                          history_steps: int = 8, horizon_steps: int = 8,
                          excursion_threshold: float = 180.0) -> pd.DataFrame:
    """Create future-excursion labels from past-only features.

    With 15-minute samples, history_steps=8 represents 2 hours and
    horizon_steps=8 represents the following 2 hours. The threshold is an
    illustrative engineering label, not a diagnostic recommendation.
    """
    merged = stream.merge(profiles, on="patient_id", how="left", validate="many_to_one")
    rows = []
    for pid, group in merged.groupby("patient_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        g = group["glucose_mg_dl"].to_numpy()
        for i in range(history_steps-1, len(group)-horizon_steps):
            past = g[i-history_steps+1:i+1]
            future = g[i+1:i+1+horizon_steps]
            row = {
                "patient_id": pid,
                "timestamp": group.loc[i, "timestamp"],
                "glucose_now": float(g[i]),
                "glucose_mean_2h": float(np.mean(past)),
                "glucose_std_2h": float(np.std(past)),
                "glucose_slope_2h": float((past[-1]-past[0]) / max(history_steps-1,1)),
                "steps_now": float(group.loc[i, "steps_15min"]),
                "heart_rate_now": float(group.loc[i, "heart_rate_bpm"]),
                "sleep_proxy_now": float(group.loc[i, "sleep_proxy"]),
                "carbs_last_2h": float(group.loc[max(0,i-history_steps+1):i, "carbs_g_simulated"].sum()),
                "age_years": float(group.loc[i, "age_years"]),
                "bmi": float(group.loc[i, "bmi"]),
                "hba1c_percent": float(group.loc[i, "hba1c_percent"]),
                "years_since_diagnosis": float(group.loc[i, "years_since_diagnosis"]),
                "medication_recorded": float(group.loc[i, "medication_recorded"]),
                "future_max_glucose": float(np.max(future)),
                "future_excursion": int(np.max(future) >= excursion_threshold),
                "data_origin": "synthetic_demo"
            }
            rows.append(row)
    return pd.DataFrame(rows)
