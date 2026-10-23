"""Train/evaluate a transparent baseline and ML classifier."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = [
    "glucose_now", "glucose_mean_2h", "glucose_std_2h", "glucose_slope_2h",
    "steps_now", "heart_rate_now", "sleep_proxy_now", "carbs_last_2h",
    "age_years", "bmi", "hba1c_percent", "years_since_diagnosis",
    "medication_recorded"
]
TARGET = "future_excursion"

def patient_split(table: pd.DataFrame, test_fraction: float = 0.25, seed: int = 42):
    ids = table["patient_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_ids = set(ids[:n_test])
    train = table[~table.patient_id.isin(test_ids)].copy()
    test = table[table.patient_id.isin(test_ids)].copy()
    return train, test


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1500, random_state=42))
    ])


def evaluate(y_true, probabilities, threshold=0.5):
    pred = (np.asarray(probabilities) >= threshold).astype(int)
    y = np.asarray(y_true).astype(int)
    metrics = {
        "n_test_windows": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else None,
        "accuracy_at_0_5": float(accuracy_score(y, pred)) if len(y) else None,
        "precision_at_0_5": float(precision_score(y, pred, zero_division=0)) if len(y) else None,
        "recall_at_0_5": float(recall_score(y, pred, zero_division=0)) if len(y) else None,
        "f1_at_0_5": float(f1_score(y, pred, zero_division=0)) if len(y) else None,
        "brier_score": float(brier_score_loss(y, probabilities)) if len(y) else None,
    }
    metrics["roc_auc"] = float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    metrics["average_precision"] = float(average_precision_score(y, probabilities)) if len(np.unique(y)) == 2 else None
    return metrics

def train_and_evaluate(table: pd.DataFrame):
    train, test = patient_split(table)
    model = build_model()
    model.fit(train[FEATURES], train[TARGET])
    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = evaluate(test[TARGET], probabilities)
    # Simple persistence-style heuristic: current glucose is not a learned model.
    # It is intentionally presented as a transparent comparator, not as clinical guidance.
    heuristic = np.clip((test["glucose_now"].to_numpy() - 140) / 70, 0.01, 0.99)
    heuristic_metrics = evaluate(test[TARGET], heuristic)
    predictions = test[["patient_id", "timestamp", "glucose_now", TARGET]].copy()
    predictions["risk_probability"] = probabilities
    predictions["heuristic_score"] = heuristic
    return model, metrics, heuristic_metrics, predictions, train, test
