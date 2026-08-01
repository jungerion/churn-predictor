import pandas as pd
from churn_predictor.data.load import load_and_clean, train_test_columns


def _write_sample_csv(tmp_path):
    df = pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Female", "Male", "Female"],
        "tenure": [1, 34, 2],
        "TotalCharges": ["29.85", " ", "108.15"],  # blank string like real Telco data
        "Churn": ["No", "No", "Yes"],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return path


def test_churn_target_mapped_to_binary(tmp_path):
    path = _write_sample_csv(tmp_path)
    df = load_and_clean(path, target_column="Churn")
    assert set(df["Churn"].unique()) <= {0, 1}
    assert df.loc[2, "Churn"] == 1