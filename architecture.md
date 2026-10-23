# GlucoTwin Architecture

```mermaid
flowchart TD
    A["Static / historical profile<br/>synthetic EHR-like fields"] --> C["Patient ID join + validation"]
    B["Dynamic time-series<br/>synthetic CGM / wearable-like stream"] --> C
    C --> D["Time-aware feature engineering<br/>past 2-hour window"]
    D --> E["Patient-level train/test split"]
    E --> F["Logistic regression model<br/>plus simple comparator"]
    F --> G["Evaluation report<br/>ROC-AUC, precision, recall, Brier"]
    D --> H["Digital Twin state updater"]
    F --> H
    H --> I["Streamlit clinician-facing demo<br/>profile, trends, estimate, caveats"]
```

## Data flow

1. The static profile generator produces fictional demographics and baseline fields.
2. The sensor generator produces timestamped synthetic glucose, steps, heart rate, sleep proxy and simulated meal inputs.
3. Both streams are linked using a synthetic patient ID.
4. Features use only past readings at the prediction timestamp; labels are based on the following two hours.
5. The classifier is evaluated with a patient-level split.
6. The dashboard replays a selected patient's stream and updates features and the model estimate as the replay position changes.

## Non-goals

This prototype does not establish a clinical digital twin, validated physiological simulation, or safe clinical decision support.
