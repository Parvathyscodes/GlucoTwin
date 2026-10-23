from src.data_pipeline import make_synthetic_cohort, make_supervised_table
from src.modeling import patient_split, build_model, FEATURES, TARGET

def test_synthetic_data_shapes_and_labels():
    profiles, stream = make_synthetic_cohort(n_patients=8, days=2, seed=7)
    table = make_supervised_table(profiles, stream)
    assert profiles.patient_id.nunique() == 8
    assert stream.patient_id.nunique() == 8
    assert set(table[TARGET].unique()).issubset({0, 1})
    assert all(c in table.columns for c in FEATURES)

def test_patient_split_has_no_patient_overlap():
    profiles, stream = make_synthetic_cohort(n_patients=12, days=2, seed=8)
    table = make_supervised_table(profiles, stream)
    train, test = patient_split(table)
    assert set(train.patient_id).isdisjoint(set(test.patient_id))
    assert len(train) and len(test)
