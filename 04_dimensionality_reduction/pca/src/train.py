"""Fit and save a PCA dimensionality-reduction pipeline."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
TEST_SIZE = 0.20
VARIANCE_TO_RETAIN = 0.95

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TEST_DATA_PATH = DATA_DIR / "test_data.csv"
MODEL_PATH = MODEL_DIR / "pca_pipeline.joblib"

TRAINING_SUMMARY_PATH = (
    METRICS_DIR / "training_summary.json"
)
EXPLAINED_VARIANCE_PATH = (
    METRICS_DIR / "explained_variance.csv"
)
COMPONENT_LOADINGS_PATH = (
    METRICS_DIR / "component_loadings.csv"
)


def load_dataset() -> pd.DataFrame:
    """Load numerical Wine dataset features."""

    dataset = load_wine(as_frame=True)
    return dataset.data


def split_dataset(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reproducible training and held-out test sets."""

    train_data, test_data = train_test_split(
        features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return (
        train_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def build_pipeline() -> Pipeline:
    """Create scaling and PCA pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=VARIANCE_TO_RETAIN,
                    svd_solver="full",
                ),
            ),
        ]
    )


def save_datasets(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save the exact train and test feature datasets."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(
        TRAIN_DATA_PATH,
        index=False,
    )

    test_data.to_csv(
        TEST_DATA_PATH,
        index=False,
    )


def save_training_summary(
    model: Pipeline,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> None:
    """Save PCA configuration and training information."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    summary = {
        "algorithm": "Principal Component Analysis",
        "dataset": "Scikit-learn Wine dataset",
        "training_records": len(train_data),
        "testing_records": len(test_data),
        "original_feature_count": train_data.shape[1],
        "retained_component_count": int(
            pca.n_components_
        ),
        "requested_variance_retention": (
            VARIANCE_TO_RETAIN
        ),
        "actual_cumulative_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "svd_solver": pca.svd_solver,
        "feature_scaling": "StandardScaler",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scaler_fitted_on_training_data_only": True,
        "pca_fitted_on_training_data_only": True,
        "test_data_used_during_fitting": False,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)


def save_explained_variance(
    model: Pipeline,
) -> None:
    """Save variance explained by each retained component."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    component_numbers = np.arange(
        1,
        pca.n_components_ + 1,
    )

    variance_table = pd.DataFrame(
        {
            "component": [
                f"PC{number}"
                for number in component_numbers
            ],
            "explained_variance": (
                pca.explained_variance_
            ),
            "explained_variance_ratio": (
                pca.explained_variance_ratio_
            ),
            "cumulative_explained_variance_ratio": (
                np.cumsum(
                    pca.explained_variance_ratio_
                )
            ),
            "singular_value": (
                pca.singular_values_
            ),
        }
    )

    variance_table.to_csv(
        EXPLAINED_VARIANCE_PATH,
        index=False,
    )


def save_component_loadings(
    model: Pipeline,
    feature_names: list[str],
) -> None:
    """Save PCA component weights for every feature."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    component_names = [
        f"PC{index + 1}"
        for index in range(pca.n_components_)
    ]

    loading_table = pd.DataFrame(
        pca.components_.T,
        index=feature_names,
        columns=component_names,
    )

    loading_table.index.name = "feature"

    loading_table.to_csv(
        COMPONENT_LOADINGS_PATH
    )


def main() -> None:
    """Run the complete PCA fitting workflow."""

    features = load_dataset()

    train_data, test_data = split_dataset(
        features
    )

    model = build_pipeline()

    # Only the training data are used here.
    model.fit(train_data)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        MODEL_PATH,
    )

    save_datasets(
        train_data=train_data,
        test_data=test_data,
    )

    save_training_summary(
        model=model,
        train_data=train_data,
        test_data=test_data,
    )

    save_explained_variance(model)

    save_component_loadings(
        model=model,
        feature_names=list(
            train_data.columns
        ),
    )

    pca = model.named_steps["pca"]

    print("\nPCA training completed.")
    print(
        f"Training records: {len(train_data)}"
    )
    print(
        f"Held-out test records: {len(test_data)}"
    )
    print(
        "Original feature count: "
        f"{train_data.shape[1]}"
    )
    print(
        "Retained component count: "
        f"{pca.n_components_}"
    )
    print(
        "Cumulative explained variance: "
        f"{np.sum(pca.explained_variance_ratio_):.4f}"
    )
    print(f"Pipeline saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
