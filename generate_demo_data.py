from pathlib import Path
from src.data_pipeline import make_synthetic_cohort, make_supervised_table

root = Path(__file__).parent
profiles, stream = make_synthetic_cohort()
table = make_supervised_table(profiles, stream)
profiles.to_csv(root / "data" / "synthetic_patient_profiles.csv", index=False)
stream.to_csv(root / "data" / "synthetic_sensor_stream.csv", index=False)
table.to_csv(root / "data" / "synthetic_supervised_windows.csv", index=False)
print(f"Wrote {len(profiles)} synthetic profiles, {len(stream)} readings, {len(table)} windows.")
print("Reminder: synthetic demonstration data only; not clinical evidence.")
