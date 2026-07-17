"""Demonstrate Isolation Forest prediction on one held-out row."""

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "isolation_forest_pipeline.joblib"
)

EVALUATION_DATA_PATH = (
    BASE_DIR
    / "data"
    / "evaluation_data.csv"
)


def load_artifacts():
    """Load the model and evaluation dataset."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Isolation Forest model not found. "
            "Run train.py first."
        )

    if not EVALUATION_DATA_PATH.exists():
        raise FileNotFoundError(
            "Evaluation data not found. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    evaluation_data = pd.read_csv(
        EVALUATION_DATA_PATH
    )

    return model, evaluation_data


def select_example(
    evaluation_data: pd.DataFrame,
) -> pd.DataFrame:
    """Select one held-out anomaly example when available."""

    anomaly_examples = evaluation_data[
        evaluation_data[
            "actual_is_anomaly"
        ] == 1
    ]

    if not anomaly_examples.empty:
        return anomaly_examples.iloc[
            [0]
        ].copy()

    return evaluation_data.iloc[
        [0]
    ].copy()


def main() -> None:
    """Predict anomaly status for one evaluation row."""

    model, evaluation_data = (
        load_artifacts()
    )

    example = select_example(
        evaluation_data
    )

    actual_label = int(
        example[
            "actual_is_anomaly"
        ].iloc[0]
    )

    observation_source = str(
        example[
            "observation_source"
        ].iloc[0]
    )

    features = example.drop(
        columns=[
            "actual_is_anomaly",
            "observation_source",
        ]
    )

    raw_prediction = int(
        model.predict(features)[0]
    )

    predicted_is_anomaly = (
        1
        if raw_prediction == -1
        else 0
    )

    decision_function = float(
        model.decision_function(
            features
        )[0]
    )

    score_sample = float(
        model.score_samples(
            features
        )[0]
    )

    anomaly_score = (
        -decision_function
    )

    actual_name = (
        "anomaly"
        if actual_label == 1
        else "normal"
    )

    predicted_name = (
        "anomaly"
        if predicted_is_anomaly == 1
        else "normal"
    )

    print(
        "\nIsolation Forest "
        "Held-Out Example"
    )

    print(
        f"Observation source: "
        f"{observation_source}"
    )

    print(
        f"Actual status: {actual_name}"
    )

    print(
        f"Predicted status: "
        f"{predicted_name}"
    )

    print(
        "Raw Isolation Forest prediction: "
        f"{raw_prediction}"
    )

    print(
        "Decision function: "
        f"{decision_function:.6f}"
    )

    print(
        "Anomaly score: "
        f"{anomaly_score:.6f}"
    )

    print(
        "Raw score_samples value: "
        f"{score_sample:.6f}"
    )

    print(
        "Prediction correct: "
        f"{actual_label == predicted_is_anomaly}"
    )

    print(
        "\nInterpretation:"
    )

    if predicted_is_anomaly == 1:
        print(
            "The observation falls below the "
            "learned normality threshold and "
            "was flagged as an anomaly."
        )
    else:
        print(
            "The observation falls within the "
            "model's learned normal region."
        )


if __name__ == "__main__":
    main()
