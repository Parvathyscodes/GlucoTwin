from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from src.data_pipeline import make_synthetic_cohort, make_supervised_table
from src.modeling import FEATURES, train_and_evaluate
from src.twin import latest_feature_row

st.set_page_config(page_title="GlucoTwin | Digital Patient Twin", page_icon="🩺", layout="wide")
st.title("🩺 GlucoTwin")
st.caption("A research proof-of-concept for patient-state tracking and short-horizon glucose-excursion prediction")

st.warning("Research demo only. All default records are synthetic. This is not a medical device, diagnosis, or treatment recommendation.")

@st.cache_data
def load_data():
    profiles, stream = make_synthetic_cohort(n_patients=80, days=5, seed=42)
    windows = make_supervised_table(profiles, stream)
    return profiles, stream, windows

@st.cache_resource
def get_model():
    profiles, stream, windows = load_data()
    model, metrics, heuristic_metrics, predictions, train, test = train_and_evaluate(windows)
    return model, metrics, heuristic_metrics, predictions, train, test

profiles, stream, windows = load_data()
model, metrics, heuristic_metrics, predictions, train, test = get_model()

with st.sidebar:
    st.header("Virtual patient")
    patient_id = st.selectbox("Select synthetic patient", profiles.patient_id.tolist(), index=0)
    st.markdown("---")
    st.subheader("Replay controls")
    max_rows = len(stream[stream.patient_id == patient_id])
    replay_idx = st.slider("Sensor stream position", min_value=16, max_value=max_rows, value=max_rows, step=1, key="replay_idx")
    st.caption("Move the slider to replay the sensor stream. The twin state and risk estimate update from the selected history.")
    st.markdown("---")
    st.subheader("About this prototype")
    st.write("Static profile + dynamic sensor stream → patient-state features → ML risk estimate.")

profile = profiles[profiles.patient_id == patient_id].iloc[0]
patient_stream = stream[stream.patient_id == patient_id].sort_values("timestamp").iloc[:replay_idx].copy()
feature_row = latest_feature_row(profile.to_dict(), patient_stream)
x = pd.DataFrame([feature_row], columns=FEATURES)
risk = float(model.predict_proba(x)[0, 1])
latest = patient_stream.iloc[-1]
current_glucose = float(latest.glucose_mg_dl)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current glucose (synthetic)", f"{current_glucose:.0f} mg/dL")
c2.metric("Model-estimated event probability", f"{risk*100:.1f}%")
c3.metric("Readings in replay", f"{len(patient_stream):,}")
c4.metric("Patient profile", f"{int(profile.age_years)} years")

st.subheader("Virtual patient profile")
p1, p2, p3, p4 = st.columns(4)
p1.metric("BMI", f"{profile.bmi:.1f}")
p2.metric("HbA1c", f"{profile.hba1c_percent:.1f}%")
p3.metric("Years since diagnosis", f"{int(profile.years_since_diagnosis)}")
p4.metric("Medication recorded", "Yes" if profile.medication_recorded else "No")

st.subheader("Dynamic sensor timeline")
fig = px.line(patient_stream, x="timestamp", y="glucose_mg_dl", title="Simulated continuous glucose stream",
              labels={"timestamp":"Time", "glucose_mg_dl":"Glucose (mg/dL)"})
fig.add_hline(y=180, line_dash="dash", annotation_text="Illustrative excursion threshold (180 mg/dL)")
st.plotly_chart(fig, use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Supporting signals")
    sig = patient_stream.tail(96)
    fig2 = px.line(sig, x="timestamp", y="heart_rate_bpm", title="Heart rate (simulated)",
                   labels={"timestamp":"Time", "heart_rate_bpm":"Beats/min"})
    st.plotly_chart(fig2, use_container_width=True)
with right:
    st.subheader("Activity and simulated meal inputs")
    fig3 = px.bar(sig, x="timestamp", y="steps_15min", title="Steps per 15-minute interval",
                  labels={"timestamp":"Time", "steps_15min":"Steps"})
    st.plotly_chart(fig3, use_container_width=True)

st.subheader("Model evaluation (synthetic held-out patients)")
st.caption("These metrics check the code path only. They must not be interpreted as clinical accuracy or expected real-world performance.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("ROC-AUC", "N/A" if metrics["roc_auc"] is None else f'{metrics["roc_auc"]:.3f}')
m2.metric("Recall @ 0.5", "N/A" if metrics["recall_at_0_5"] is None else f'{metrics["recall_at_0_5"]:.3f}')
m3.metric("Precision @ 0.5", "N/A" if metrics["precision_at_0_5"] is None else f'{metrics["precision_at_0_5"]:.3f}')
m4.metric("Brier score", "N/A" if metrics["brier_score"] is None else f'{metrics["brier_score"]:.3f}')

with st.expander("What does the model use?"):
    st.write("Past 2-hour glucose summary and slope, latest activity and heart rate, a simulated meal proxy, plus static age, BMI, HbA1c, years since diagnosis, and medication-recorded flag.")
    st.write("Target: whether any future synthetic glucose reading reaches 180 mg/dL in the next two hours. This is an illustrative threshold, not an emergency definition.")
with st.expander("Data provenance and limitations"):
    st.write("The bundled demonstration cohort is generated in code with a fixed random seed. Static and dynamic streams share synthetic patient IDs by construction. No real patient records are included.")
    st.write("The generator is not a validated physiological simulator. Model performance on generated data does not establish clinical validity, fairness, or usefulness. Real open-source data integration and independent validation remain necessary before any clinical interpretation.")
with st.expander("Sample recent readings"):
    st.dataframe(patient_stream.tail(20), use_container_width=True, hide_index=True)
