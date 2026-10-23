"""Small state container for replaying a patient's dynamic stream."""
from __future__ import annotations
import pandas as pd
from .modeling import FEATURES

def latest_feature_row(profile: dict, stream: pd.DataFrame) -> dict:
    stream = stream.sort_values("timestamp").copy()
    if stream.empty:
        raise ValueError("The sensor stream is empty.")
    glucose = stream["glucose_mg_dl"].astype(float)
    recent = stream.tail(8)
    if len(recent) < 2:
        slope = 0.0
    else:
        slope = float((recent["glucose_mg_dl"].iloc[-1] - recent["glucose_mg_dl"].iloc[0]) / max(len(recent)-1,1))
    row = {
        "glucose_now": float(glucose.iloc[-1]),
        "glucose_mean_2h": float(recent["glucose_mg_dl"].mean()),
        "glucose_std_2h": float(recent["glucose_mg_dl"].std(ddof=0)),
        "glucose_slope_2h": slope,
        "steps_now": float(recent["steps_15min"].iloc[-1]),
        "heart_rate_now": float(recent["heart_rate_bpm"].iloc[-1]),
        "sleep_proxy_now": float(recent["sleep_proxy"].iloc[-1]),
        "carbs_last_2h": float(recent["carbs_g_simulated"].sum()),
        "age_years": float(profile["age_years"]),
        "bmi": float(profile["bmi"]),
        "hba1c_percent": float(profile["hba1c_percent"]),
        "years_since_diagnosis": float(profile["years_since_diagnosis"]),
        "medication_recorded": float(profile["medication_recorded"]),
    }
    return {k: row[k] for k in FEATURES}
