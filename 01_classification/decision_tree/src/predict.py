"""Generate predictions using the saved Decision Tree model."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "decision_tree.joblib"


def load_model():
    """Load trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run train.py first.")

    return joblib.load(MODEL_PATH)


def main() -> None:
    """Predict the class of one sample record."""

    model = load_model()
    dataset = load_breast_cancer(as_frame=True)

    sample = dataset.data.iloc[[0]].copy()

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    class_name = dataset.target_names[prediction]

    print(f"Predicted class: {class_name}")
    print(
        f"Malignant probability: {probability[0]:.4f}"
    )
    print(
        f"Benign probability: {probability[1]:.4f}"
    )


if __name__ == "__main__":
    main()