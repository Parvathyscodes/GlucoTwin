from pathlib import Path
import json
import joblib
from src.data_pipeline import make_synthetic_cohort, make_supervised_table
from src.modeling import train_and_evaluate

root = Path(__file__).parent
profiles, stream = make_synthetic_cohort()
table = make_supervised_table(profiles, stream)
model, metrics, heuristic_metrics, predictions, train, test = train_and_evaluate(table)
(root / "models").mkdir(exist_ok=True)
joblib.dump(model, root / "models" / "glucotwin_logistic_model.joblib")
(root / "reports").mkdir(exist_ok=True)
with open(root / "reports" / "evaluation_metrics.json", "w") as f:
    json.dump({
        "data_origin": "synthetic_demo",
        "target_definition": "future maximum glucose >= 180 mg/dL within the next 2 hours in simulated data",
        "patient_level_split": True,
        "model": metrics,
        "simple_heuristic_comparator": heuristic_metrics,
        "disclaimer": "Metrics are only a software-pipeline demonstration on synthetic data; they do not estimate clinical performance."
    }, f, indent=2)
predictions.to_csv(root / "reports" / "heldout_predictions.csv", index=False)
print(json.dumps(metrics, indent=2))
print("Saved model and reports. Interpret metrics only as synthetic pipeline checks.")
