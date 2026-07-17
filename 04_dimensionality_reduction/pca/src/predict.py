"""Transform held-out or external observations using saved PCA."""

from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "pca_pipeline.joblib"
TEST_DATA_PATH = BASE_DIR / "data" / "test_data.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "predictions"


def parse_arguments() -> argparse.Namespace:
    """Read optional external CSV path."""

    parser = argparse.ArgumentParser(
        description=(
            "Transform numerical records using "
            "the trained PCA pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional path to an external CSV file. "
            "Without this argument, one held-out "
            "test row is used."
        ),
    )

    return parser.parse_args()


def load_model():
    """Load the trained PCA pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "PCA pipeline not found. Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def expected_feature_names(model) -> list[str]:
    """Return expected original feature names."""

    if not hasattr(model, "feature_names_in_"):
        raise AttributeError(
            "Saved pipeline does not contain "
            "feature-name metadata."
        )

    return list(model.feature_names_in_)


def load_input_data(
    input_path: Path | None,
) -> tuple[pd.DataFrame, str]:
    """Load external CSV or one held-out test row."""

    if input_path is not None:
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input CSV not found: {input_path}"
            )

        data = pd.read_csv(input_path)

        if data.empty:
            raise ValueError(
                "The external input CSV contains no rows."
            )

        return data, "external_csv"

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Held-out test data not found. "
            "Run train.py first."
        )

    test_data = pd.read_csv(TEST_DATA_PATH)

    if test_data.empty:
        raise ValueError(
            "The held-out test dataset is empty."
        )

    # This row was not used when fitting PCA.
    return (
        test_data.iloc[[0]].copy(),
        "held_out_test_demo",
    )


def validate_input(
    data: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:
    """Validate and reorder input columns."""

    missing_features = (
        set(expected_features)
        - set(data.columns)
    )

    unexpected_features = (
        set(data.columns)
        - set(expected_features)
    )

    if missing_features:
        raise ValueError(
            "Missing features: "
            f"{sorted(missing_features)}"
        )

    if unexpected_features:
        raise ValueError(
            "Unexpected features: "
            f"{sorted(unexpected_features)}"
        )

    ordered_data = data.loc[
        :,
        expected_features,
    ].copy()

    non_numeric_columns = (
        ordered_data
        .select_dtypes(exclude="number")
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise ValueError(
            "Non-numeric columns found: "
            f"{non_numeric_columns}"
        )

    if ordered_data.isnull().any().any():
        missing_locations = (
            ordered_data
            .isnull()
            .sum()
        )

        missing_locations = (
            missing_locations[
                missing_locations > 0
            ]
            .to_dict()
        )

        raise ValueError(
            "Input contains missing values: "
            f"{missing_locations}"
        )

    return ordered_data


def transform_and_reconstruct(
    model,
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform data and reconstruct original feature values."""

    transformed = model.transform(data)

    scaler = model.named_steps["scaler"]
    pca = model.named_steps["pca"]

    reconstructed_scaled = pca.inverse_transform(
        transformed
    )

    reconstructed_original = scaler.inverse_transform(
        reconstructed_scaled
    )

    return transformed, reconstructed_original


def save_outputs(
    original_data: pd.DataFrame,
    transformed: np.ndarray,
    reconstructed: np.ndarray,
    input_source: str,
) -> None:
    """Save transformed components and reconstructions."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    component_columns = [
        f"PC{index + 1}"
        for index in range(
            transformed.shape[1]
        )
    ]

    transformed_table = pd.DataFrame(
        transformed,
        columns=component_columns,
    )

    transformed_table.insert(
        0,
        "input_source",
        input_source,
    )

    transformed_table.insert(
        1,
        "row_id",
        np.arange(len(transformed_table)),
    )

    reconstructed_table = pd.DataFrame(
        reconstructed,
        columns=original_data.columns,
    )

    reconstruction_error = (
        original_data.reset_index(drop=True)
        - reconstructed_table
    )

    transformed_table.to_csv(
        OUTPUT_DIR
        / "pca_transformed_predictions.csv",
        index=False,
    )

    reconstructed_table.to_csv(
        OUTPUT_DIR
        / "pca_reconstructed_predictions.csv",
        index=False,
    )

    reconstruction_error.to_csv(
        OUTPUT_DIR
        / "pca_prediction_reconstruction_errors.csv",
        index=False,
    )


def main() -> None:
    """Transform held-out or external observations."""

    arguments = parse_arguments()
    model = load_model()

    input_data, input_source = load_input_data(
        arguments.input
    )

    expected_features = expected_feature_names(
        model
    )

    validated_data = validate_input(
        data=input_data,
        expected_features=expected_features,
    )

    transformed, reconstructed = (
        transform_and_reconstruct(
            model=model,
            data=validated_data,
        )
    )

    save_outputs(
        original_data=validated_data,
        transformed=transformed,
        reconstructed=reconstructed,
        input_source=input_source,
    )

    component_columns = [
        f"PC{index + 1}"
        for index in range(
            transformed.shape[1]
        )
    ]

    transformed_table = pd.DataFrame(
        transformed,
        columns=component_columns,
    )

    row_rmse = np.sqrt(
        np.mean(
            (
                validated_data.to_numpy()
                - reconstructed
            )
            ** 2,
            axis=1,
        )
    )

    print("\nPCA Transformation")
    print(f"Input source: {input_source}")
    print(
        f"Records transformed: "
        f"{len(validated_data)}"
    )
    print(
        f"Original feature count: "
        f"{validated_data.shape[1]}"
    )
    print(
        f"Retained component count: "
        f"{transformed.shape[1]}"
    )

    print("\nFirst transformed row:")

    print(
        transformed_table
        .iloc[[0]]
        .to_string(index=False)
    )

    print(
        "\nFirst-row reconstruction RMSE "
        "in original feature units: "
        f"{row_rmse[0]:.6f}"
    )

    print(
        "\nOutputs saved to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
