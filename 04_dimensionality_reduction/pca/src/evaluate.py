"""Evaluate PCA transformation on training and held-out test data."""

from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "pca_pipeline.joblib"
TRAIN_DATA_PATH = BASE_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"

FIGURES_DIR = BASE_DIR / "outputs" / "figures"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
TRANSFORMED_DIR = (
    BASE_DIR / "outputs" / "transformed_data"
)

METRICS_PATH = METRICS_DIR / "metrics.json"
RECONSTRUCTION_FEATURE_PATH = (
    METRICS_DIR
    / "test_reconstruction_error_by_feature.csv"
)


def load_artifacts():
    """Load saved PCA pipeline and exact train/test sets."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "PCA pipeline not found. Run train.py first."
        )

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. Run train.py first."
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Test data not found. Run train.py first."
        )

    model = joblib.load(MODEL_PATH)
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

    return model, train_data, test_data


def component_names(
    component_count: int,
) -> list[str]:
    """Return standard principal-component column names."""

    return [
        f"PC{index + 1}"
        for index in range(component_count)
    ]


def transform_datasets(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform training and held-out test features."""

    transformed_train = model.transform(
        train_data
    )

    transformed_test = model.transform(
        test_data
    )

    return transformed_train, transformed_test


def reconstruct_original_data(
    model,
    transformed_data: np.ndarray,
) -> np.ndarray:
    """Reconstruct original-unit features from PCA components."""

    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]

    reconstructed_scaled = pca.inverse_transform(
        transformed_data
    )

    reconstructed_original = scaler.inverse_transform(
        reconstructed_scaled
    )

    return reconstructed_original


def calculate_reconstruction_metrics(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    reconstructed_train: np.ndarray,
    reconstructed_test: np.ndarray,
) -> dict[str, float]:
    """Calculate reconstruction errors on train and test sets."""

    train_mse = mean_squared_error(
        train_data,
        reconstructed_train,
    )

    test_mse = mean_squared_error(
        test_data,
        reconstructed_test,
    )

    train_rmse = float(np.sqrt(train_mse))
    test_rmse = float(np.sqrt(test_mse))

    return {
        "train_reconstruction_mse_original_units": float(
            train_mse
        ),
        "train_reconstruction_rmse_original_units": (
            train_rmse
        ),
        "test_reconstruction_mse_original_units": float(
            test_mse
        ),
        "test_reconstruction_rmse_original_units": (
            test_rmse
        ),
        "test_to_train_rmse_ratio": float(
            test_rmse / train_rmse
            if train_rmse != 0
            else np.nan
        ),
    }


def calculate_standardized_reconstruction_metrics(
    model,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
) -> dict[str, float]:
    """Measure reconstruction in standardized feature space."""

    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]

    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    reconstructed_scaled_train = (
        pca.inverse_transform(
            transformed_train
        )
    )

    reconstructed_scaled_test = (
        pca.inverse_transform(
            transformed_test
        )
    )

    train_mse = mean_squared_error(
        scaled_train,
        reconstructed_scaled_train,
    )

    test_mse = mean_squared_error(
        scaled_test,
        reconstructed_scaled_test,
    )

    return {
        "train_reconstruction_mse_standardized": float(
            train_mse
        ),
        "train_reconstruction_rmse_standardized": float(
            np.sqrt(train_mse)
        ),
        "test_reconstruction_mse_standardized": float(
            test_mse
        ),
        "test_reconstruction_rmse_standardized": float(
            np.sqrt(test_mse)
        ),
    }


def save_metrics(
    model,
    reconstruction_metrics: dict[str, float],
    standardized_metrics: dict[str, float],
) -> None:
    """Save complete PCA evaluation metrics."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    metrics = {
        "original_feature_count": int(
            pca.n_features_in_
        ),
        "retained_component_count": int(
            pca.n_components_
        ),
        "dimensionality_reduction_percentage": float(
            (
                1
                - pca.n_components_
                / pca.n_features_in_
            )
            * 100
        ),
        "cumulative_explained_variance_ratio": float(
            np.sum(
                pca.explained_variance_ratio_
            )
        ),
        **reconstruction_metrics,
        **standardized_metrics,
    }

    with METRICS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)


def save_transformed_data(
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
) -> None:
    """Save component representations for train and test sets."""

    TRANSFORMED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    names = component_names(
        transformed_train.shape[1]
    )

    train_table = pd.DataFrame(
        transformed_train,
        columns=names,
    )

    train_table.insert(
        0,
        "dataset_split",
        "train",
    )

    train_table.insert(
        1,
        "row_id",
        np.arange(len(train_table)),
    )

    test_table = pd.DataFrame(
        transformed_test,
        columns=names,
    )

    test_table.insert(
        0,
        "dataset_split",
        "test",
    )

    test_table.insert(
        1,
        "row_id",
        np.arange(len(test_table)),
    )

    train_table.to_csv(
        TRANSFORMED_DIR / "train_transformed.csv",
        index=False,
    )

    test_table.to_csv(
        TRANSFORMED_DIR / "test_transformed.csv",
        index=False,
    )


def save_reconstructed_test_data(
    test_data: pd.DataFrame,
    reconstructed_test: np.ndarray,
) -> None:
    """Save held-out test data and its reconstruction."""

    TRANSFORMED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reconstructed_table = pd.DataFrame(
        reconstructed_test,
        columns=test_data.columns,
    )

    reconstructed_table.to_csv(
        TRANSFORMED_DIR
        / "test_reconstructed.csv",
        index=False,
    )

    error_table = (
        test_data.reset_index(drop=True)
        - reconstructed_table
    )

    error_table.to_csv(
        TRANSFORMED_DIR
        / "test_reconstruction_errors.csv",
        index=False,
    )


def save_feature_reconstruction_errors(
    test_data: pd.DataFrame,
    reconstructed_test: np.ndarray,
) -> None:
    """Save test reconstruction errors for each feature."""

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    reconstructed_table = pd.DataFrame(
        reconstructed_test,
        columns=test_data.columns,
    )

    errors = (
        test_data.reset_index(drop=True)
        - reconstructed_table
    )

    feature_metrics = pd.DataFrame(
        {
            "feature": test_data.columns,
            "mean_absolute_reconstruction_error": (
                errors.abs().mean().to_numpy()
            ),
            "root_mean_squared_reconstruction_error": (
                np.sqrt(
                    errors.pow(2).mean()
                ).to_numpy()
            ),
        }
    ).sort_values(
        by="root_mean_squared_reconstruction_error",
        ascending=False,
    )

    feature_metrics.to_csv(
        RECONSTRUCTION_FEATURE_PATH,
        index=False,
    )


def save_explained_variance_plot(model) -> None:
    """Plot individual explained variance ratios."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    labels = component_names(
        pca.n_components_
    )

    plt.figure(figsize=(9, 6))

    plt.bar(
        labels,
        pca.explained_variance_ratio_,
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title(
        "Explained Variance by Principal Component"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "explained_variance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_cumulative_variance_plot(model) -> None:
    """Plot cumulative explained variance."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    component_numbers = np.arange(
        1,
        pca.n_components_ + 1,
    )

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(9, 6))

    plt.plot(
        component_numbers,
        cumulative_variance,
        marker="o",
    )

    plt.axhline(
        0.95,
        linestyle="--",
        label="95% target",
    )

    plt.xlabel("Number of Components")
    plt.ylabel(
        "Cumulative Explained Variance Ratio"
    )
    plt.title(
        "PCA Cumulative Explained Variance"
    )
    plt.xticks(component_numbers)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "cumulative_explained_variance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_two_component_projection(
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
) -> None:
    """Compare train and test projections using PC1 and PC2."""

    if transformed_train.shape[1] < 2:
        print(
            "Two-component plot skipped: "
            "fewer than two components retained."
        )
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))

    plt.scatter(
        transformed_train[:, 0],
        transformed_train[:, 1],
        alpha=0.65,
        label="Training data",
    )

    plt.scatter(
        transformed_test[:, 0],
        transformed_test[:, 1],
        marker="x",
        label="Held-out test data",
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(
        "Training and Test Data in PCA Space"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "train_test_pca_projection.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def save_loading_heatmap(model, feature_names) -> None:
    """Visualize component weights for original features."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pca = model.named_steps["pca"]

    loading_matrix = pca.components_.T

    plt.figure(
        figsize=(
            max(9, pca.n_components_ * 1.1),
            max(7, len(feature_names) * 0.45),
        )
    )

    image = plt.imshow(
        loading_matrix,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Component Weight",
    )

    plt.xticks(
        range(pca.n_components_),
        component_names(pca.n_components_),
    )

    plt.yticks(
        range(len(feature_names)),
        feature_names,
    )

    plt.xlabel("Principal Component")
    plt.ylabel("Original Feature")
    plt.title("PCA Component Loadings")
    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR / "component_loadings.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Run full PCA evaluation on held-out data."""

    model, train_data, test_data = load_artifacts()

    transformed_train, transformed_test = (
        transform_datasets(
            model=model,
            train_data=train_data,
            test_data=test_data,
        )
    )

    reconstructed_train = reconstruct_original_data(
        model,
        transformed_train,
    )

    reconstructed_test = reconstruct_original_data(
        model,
        transformed_test,
    )

    reconstruction_metrics = (
        calculate_reconstruction_metrics(
            train_data=train_data,
            test_data=test_data,
            reconstructed_train=reconstructed_train,
            reconstructed_test=reconstructed_test,
        )
    )

    standardized_metrics = (
        calculate_standardized_reconstruction_metrics(
            model=model,
            train_data=train_data,
            test_data=test_data,
            transformed_train=transformed_train,
            transformed_test=transformed_test,
        )
    )

    save_metrics(
        model=model,
        reconstruction_metrics=reconstruction_metrics,
        standardized_metrics=standardized_metrics,
    )

    save_transformed_data(
        transformed_train=transformed_train,
        transformed_test=transformed_test,
    )

    save_reconstructed_test_data(
        test_data=test_data,
        reconstructed_test=reconstructed_test,
    )

    save_feature_reconstruction_errors(
        test_data=test_data,
        reconstructed_test=reconstructed_test,
    )

    save_explained_variance_plot(model)
    save_cumulative_variance_plot(model)

    save_two_component_projection(
        transformed_train=transformed_train,
        transformed_test=transformed_test,
    )

    save_loading_heatmap(
        model=model,
        feature_names=list(train_data.columns),
    )

    pca = model.named_steps["pca"]

    print("\nPCA Evaluation")
    print(
        "Retained components: "
        f"{pca.n_components_}"
    )
    print(
        "Cumulative explained variance: "
        f"{np.sum(pca.explained_variance_ratio_):.4f}"
    )

    print("\nReconstruction metrics:")

    for name, value in {
        **reconstruction_metrics,
        **standardized_metrics,
    }.items():
        print(f"{name}: {value:.6f}")

    print(
        "\nHeld-out test transformation completed."
    )


if __name__ == "__main__":
    main()
