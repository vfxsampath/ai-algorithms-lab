"""Train and save a TF-IDF content-based recommendation system."""

from __future__ import annotations

from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


RANDOM_STATE = 42
MIN_INTERACTIONS_PER_USER = 4

INTERACTION_WEIGHTS = {
    "viewed": 1.0,
    "liked": 2.0,
    "completed": 3.0,
}

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

ITEMS_PATH = DATA_DIR / "items.csv"
ALL_INTERACTIONS_PATH = DATA_DIR / "all_interactions.csv"
TRAIN_INTERACTIONS_PATH = DATA_DIR / "train_interactions.csv"
TEST_INTERACTIONS_PATH = DATA_DIR / "test_interactions.csv"

MODEL_PATH = MODEL_DIR / "content_recommender.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TRAINING_SUMMARY_PATH = METRICS_DIR / "training_summary.json"
VOCABULARY_PATH = METRICS_DIR / "tfidf_vocabulary.csv"
USER_PROFILES_PATH = METRICS_DIR / "user_profile_summary.csv"


def create_item_catalogue() -> pd.DataFrame:
    """Create a reproducible educational course catalogue."""

    items = [
        {
            "item_id": 1,
            "title": "Python Foundations",
            "category": "Programming",
            "difficulty": "Beginner",
            "description": (
                "Learn Python syntax, variables, loops, functions, "
                "data structures, and practical programming."
            ),
            "skills": "python|programming|functions|data structures",
        },
        {
            "item_id": 2,
            "title": "Python for Data Analysis",
            "category": "Data Science",
            "difficulty": "Beginner",
            "description": (
                "Analyse structured datasets using Python, pandas, "
                "NumPy, data cleaning, and exploratory analysis."
            ),
            "skills": "python|pandas|numpy|data analysis",
        },
        {
            "item_id": 3,
            "title": "Machine Learning Foundations",
            "category": "Machine Learning",
            "difficulty": "Beginner",
            "description": (
                "Understand supervised learning, unsupervised learning, "
                "training, testing, features, models, and evaluation."
            ),
            "skills": "machine learning|training|evaluation|scikit learn",
        },
        {
            "item_id": 4,
            "title": "Classification Algorithms",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Build classification models using logistic regression, "
                "decision trees, random forests, SVM, KNN, and Naive Bayes."
            ),
            "skills": (
                "classification|logistic regression|random forest|svm|knn"
            ),
        },
        {
            "item_id": 5,
            "title": "Regression Algorithms",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Predict continuous outcomes using linear regression, "
                "regularization, tree regression, and ensemble methods."
            ),
            "skills": (
                "regression|linear regression|ridge|lasso|random forest"
            ),
        },
        {
            "item_id": 6,
            "title": "Clustering and Segmentation",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Discover groups using K-Means, DBSCAN, hierarchical "
                "clustering, and customer segmentation techniques."
            ),
            "skills": "clustering|kmeans|dbscan|segmentation",
        },
        {
            "item_id": 7,
            "title": "Anomaly Detection",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Detect unusual records using Isolation Forest, local "
                "outlier methods, anomaly scores, and threshold analysis."
            ),
            "skills": "anomaly detection|isolation forest|outliers|risk",
        },
        {
            "item_id": 8,
            "title": "Principal Component Analysis",
            "category": "Data Science",
            "difficulty": "Intermediate",
            "description": (
                "Reduce feature dimensions using PCA, explained variance, "
                "component loadings, and reconstruction analysis."
            ),
            "skills": "pca|dimensionality reduction|variance|visualization",
        },
        {
            "item_id": 9,
            "title": "Time-Series Forecasting",
            "category": "Forecasting",
            "difficulty": "Intermediate",
            "description": (
                "Forecast chronological data using ARIMA, SARIMA, "
                "exponential smoothing, Prophet, and temporal validation."
            ),
            "skills": "time series|arima|sarima|prophet|forecasting",
        },
        {
            "item_id": 10,
            "title": "XGBoost for Structured Data",
            "category": "Machine Learning",
            "difficulty": "Advanced",
            "description": (
                "Develop high-performance gradient-boosted tree models "
                "using feature engineering, tuning, and interpretation."
            ),
            "skills": "xgboost|boosting|feature engineering|tabular data",
        },
        {
            "item_id": 11,
            "title": "Natural Language Processing",
            "category": "Artificial Intelligence",
            "difficulty": "Intermediate",
            "description": (
                "Process human language using tokenization, text cleaning, "
                "TF-IDF, classification, and document analysis."
            ),
            "skills": "nlp|text processing|tfidf|tokenization",
        },
        {
            "item_id": 12,
            "title": "Semantic Search and Embeddings",
            "category": "Artificial Intelligence",
            "difficulty": "Advanced",
            "description": (
                "Build semantic retrieval systems using vector embeddings, "
                "cosine similarity, indexing, and document search."
            ),
            "skills": "embeddings|semantic search|vectors|cosine similarity",
        },
        {
            "item_id": 13,
            "title": "Large Language Model Applications",
            "category": "Artificial Intelligence",
            "difficulty": "Advanced",
            "description": (
                "Build applications with large language models, prompts, "
                "structured outputs, tool use, and API integration."
            ),
            "skills": "llm|prompt engineering|openai api|structured output",
        },
        {
            "item_id": 14,
            "title": "Retrieval-Augmented Generation",
            "category": "Artificial Intelligence",
            "difficulty": "Advanced",
            "description": (
                "Create RAG pipelines combining document ingestion, "
                "embeddings, vector retrieval, and grounded generation."
            ),
            "skills": "rag|embeddings|retrieval|llm|document intelligence",
        },
        {
            "item_id": 15,
            "title": "Agentic AI Development",
            "category": "Agentic AI",
            "difficulty": "Advanced",
            "description": (
                "Design autonomous AI agents using goals, tools, memory, "
                "planning, reasoning, and workflow execution."
            ),
            "skills": "agentic ai|agents|tools|memory|planning",
        },
        {
            "item_id": 16,
            "title": "Multi-Agent Systems",
            "category": "Agentic AI",
            "difficulty": "Advanced",
            "description": (
                "Build multi-agent architectures with specialized roles, "
                "coordination, communication, and task orchestration."
            ),
            "skills": "multi agent|orchestration|agents|coordination",
        },
        {
            "item_id": 17,
            "title": "FastAPI for AI Applications",
            "category": "Application Development",
            "difficulty": "Intermediate",
            "description": (
                "Deploy machine-learning and AI functionality through "
                "FastAPI endpoints, validation, and REST services."
            ),
            "skills": "fastapi|python|rest api|deployment",
        },
        {
            "item_id": 18,
            "title": "Database Systems with PostgreSQL",
            "category": "Application Development",
            "difficulty": "Intermediate",
            "description": (
                "Design relational schemas, SQL queries, indexes, "
                "transactions, and PostgreSQL-backed applications."
            ),
            "skills": "postgresql|sql|database|schema design",
        },
        {
            "item_id": 19,
            "title": "Supabase Application Development",
            "category": "Application Development",
            "difficulty": "Intermediate",
            "description": (
                "Build applications using Supabase PostgreSQL, "
                "authentication, storage, row-level security, and APIs."
            ),
            "skills": "supabase|postgresql|authentication|rls|storage",
        },
        {
            "item_id": 20,
            "title": "Docker for AI Developers",
            "category": "DevOps",
            "difficulty": "Intermediate",
            "description": (
                "Package AI applications using containers, Dockerfiles, "
                "images, environment variables, and deployment workflows."
            ),
            "skills": "docker|containers|deployment|devops",
        },
        {
            "item_id": 21,
            "title": "Process Mining",
            "category": "Business Intelligence",
            "difficulty": "Advanced",
            "description": (
                "Analyse event logs, reconstruct workflows, discover "
                "variants, and identify process bottlenecks."
            ),
            "skills": "process mining|event logs|pm4py|workflow",
        },
        {
            "item_id": 22,
            "title": "Business Process Optimization",
            "category": "Business Intelligence",
            "difficulty": "Intermediate",
            "description": (
                "Improve enterprise workflows through root-cause analysis, "
                "metrics, process redesign, and digital transformation."
            ),
            "skills": (
                "process optimization|root cause|digital transformation|kpi"
            ),
        },
        {
            "item_id": 23,
            "title": "Data Visualization with Python",
            "category": "Data Science",
            "difficulty": "Beginner",
            "description": (
                "Communicate insights through Python charts, dashboards, "
                "statistical graphics, and visual storytelling."
            ),
            "skills": "visualization|matplotlib|plotly|dashboard",
        },
        {
            "item_id": 24,
            "title": "Association-Rule Mining",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Discover product relationships using Apriori, support, "
                "confidence, lift, and market-basket analysis."
            ),
            "skills": "apriori|association rules|support|confidence|lift",
        },
        {
            "item_id": 25,
            "title": "Recommendation Systems",
            "category": "Artificial Intelligence",
            "difficulty": "Advanced",
            "description": (
                "Build content-based, collaborative, matrix-factorization, "
                "and hybrid recommendation systems."
            ),
            "skills": (
                "recommendation|content based|collaborative filtering|ranking"
            ),
        },
        {
            "item_id": 26,
            "title": "Model Evaluation and Validation",
            "category": "Machine Learning",
            "difficulty": "Intermediate",
            "description": (
                "Evaluate models using train-test separation, cross-validation, "
                "classification metrics, regression metrics, and ranking metrics."
            ),
            "skills": "evaluation|cross validation|metrics|model selection",
        },
        {
            "item_id": 27,
            "title": "MLOps Foundations",
            "category": "DevOps",
            "difficulty": "Advanced",
            "description": (
                "Manage machine-learning lifecycles through versioning, "
                "model tracking, testing, deployment, and monitoring."
            ),
            "skills": "mlops|monitoring|versioning|deployment|drift",
        },
        {
            "item_id": 28,
            "title": "AI Product Development",
            "category": "Business and AI",
            "difficulty": "Advanced",
            "description": (
                "Translate business problems into AI products through "
                "discovery, architecture, prototyping, evaluation, and deployment."
            ),
            "skills": "ai product|product strategy|architecture|prototype",
        },
        {
            "item_id": 29,
            "title": "Digital Transformation Strategy",
            "category": "Business and AI",
            "difficulty": "Intermediate",
            "description": (
                "Plan digital transformation using process analysis, "
                "technology roadmaps, governance, and change management."
            ),
            "skills": (
                "digital transformation|strategy|governance|change management"
            ),
        },
        {
            "item_id": 30,
            "title": "Responsible and Explainable AI",
            "category": "Business and AI",
            "difficulty": "Advanced",
            "description": (
                "Study model explainability, fairness, privacy, governance, "
                "risk management, and responsible AI deployment."
            ),
            "skills": "responsible ai|explainability|fairness|governance",
        },
    ]

    return pd.DataFrame(items)


def create_user_preferences() -> dict[int, list[str]]:
    """Define broad interests used to generate interactions."""

    return {
        101: ["Artificial Intelligence", "Agentic AI"],
        102: ["Machine Learning", "Data Science"],
        103: ["Application Development", "DevOps"],
        104: ["Forecasting", "Data Science"],
        105: ["Business Intelligence", "Business and AI"],
        106: ["Machine Learning", "Artificial Intelligence"],
        107: ["Agentic AI", "Application Development"],
        108: ["Data Science", "Business Intelligence"],
        109: ["DevOps", "Application Development"],
        110: ["Business and AI", "Artificial Intelligence"],
        111: ["Machine Learning", "Forecasting"],
        112: ["Artificial Intelligence", "Data Science"],
    }


def generate_interactions(
    items: pd.DataFrame,
) -> pd.DataFrame:
    """Generate chronological user-item interactions."""

    random_generator = random.Random(RANDOM_STATE)

    preferences = create_user_preferences()

    interaction_rows: list[dict[str, object]] = []

    start_date = pd.Timestamp("2025-01-01")

    for user_id, preferred_categories in preferences.items():
        preferred_items = items[
            items["category"].isin(
                preferred_categories
            )
        ]["item_id"].tolist()

        other_items = items[
            ~items["category"].isin(
                preferred_categories
            )
        ]["item_id"].tolist()

        interaction_count = random_generator.randint(
            6,
            9,
        )

        preferred_count = min(
            len(preferred_items),
            max(4, interaction_count - 2),
        )

        selected_preferred = random_generator.sample(
            preferred_items,
            k=preferred_count,
        )

        remaining_count = (
            interaction_count
            - preferred_count
        )

        selected_other = random_generator.sample(
            other_items,
            k=remaining_count,
        )

        selected_items = (
            selected_preferred
            + selected_other
        )

        random_generator.shuffle(selected_items)

        for sequence_number, item_id in enumerate(
            selected_items,
            start=1,
        ):
            item_category = str(
                items.loc[
                    items["item_id"] == item_id,
                    "category",
                ].iloc[0]
            )

            if item_category in preferred_categories:
                interaction_type = random_generator.choices(
                    population=[
                        "viewed",
                        "liked",
                        "completed",
                    ],
                    weights=[
                        0.15,
                        0.35,
                        0.50,
                    ],
                    k=1,
                )[0]
            else:
                interaction_type = random_generator.choices(
                    population=[
                        "viewed",
                        "liked",
                        "completed",
                    ],
                    weights=[
                        0.65,
                        0.25,
                        0.10,
                    ],
                    k=1,
                )[0]

            interaction_date = (
                start_date
                + pd.Timedelta(
                    days=(
                        sequence_number * 12
                        + random_generator.randint(
                            0,
                            7,
                        )
                        + (user_id - 100)
                    )
                )
            )

            interaction_rows.append(
                {
                    "user_id": user_id,
                    "item_id": item_id,
                    "interaction_type": interaction_type,
                    "interaction_weight": (
                        INTERACTION_WEIGHTS[
                            interaction_type
                        ]
                    ),
                    "interaction_date": interaction_date,
                }
            )

    interactions = pd.DataFrame(
        interaction_rows
    )

    return interactions.sort_values(
        by=[
            "user_id",
            "interaction_date",
        ]
    ).reset_index(drop=True)


def chronological_user_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out each user's latest positive interaction.

    User profiles are built only from earlier interactions.
    """

    train_groups = []
    test_groups = []

    for user_id, user_interactions in interactions.groupby(
        "user_id"
    ):
        ordered = user_interactions.sort_values(
            "interaction_date"
        ).reset_index(drop=True)

        if len(ordered) < MIN_INTERACTIONS_PER_USER:
            raise ValueError(
                f"User {user_id} does not have enough interactions."
            )

        positive_candidates = ordered[
            ordered["interaction_type"].isin(
                [
                    "liked",
                    "completed",
                ]
            )
        ]

        if positive_candidates.empty:
            test_index = ordered.index[-1]
        else:
            test_index = positive_candidates.index[-1]

        test_row = ordered.loc[[test_index]].copy()

        train_rows = ordered.drop(
            index=test_index
        ).copy()

        train_groups.append(train_rows)
        test_groups.append(test_row)

    train_interactions = pd.concat(
        train_groups,
        ignore_index=True,
    )

    test_interactions = pd.concat(
        test_groups,
        ignore_index=True,
    )

    return train_interactions, test_interactions


def create_content_text(
    items: pd.DataFrame,
) -> pd.Series:
    """Combine item metadata into one text representation."""

    return (
        items["title"].fillna("")
        + " "
        + items["category"].fillna("")
        + " "
        + items["difficulty"].fillna("")
        + " "
        + items["description"].fillna("")
        + " "
        + items["skills"]
        .fillna("")
        .str.replace(
            "|",
            " ",
            regex=False,
        )
    )


def fit_content_model(
    items: pd.DataFrame,
) -> tuple[
    TfidfVectorizer,
    csr_matrix,
]:
    """Fit TF-IDF and create item vectors."""

    content_text = create_content_text(items)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    item_matrix = vectorizer.fit_transform(
        content_text
    )

    return (
        vectorizer,
        csr_matrix(item_matrix),
    )


def build_user_profiles(
    items: pd.DataFrame,
    item_matrix: csr_matrix,
    train_interactions: pd.DataFrame,
) -> dict[int, csr_matrix]:
    """Create weighted TF-IDF profiles from training interactions."""

    item_position = {
        int(item_id): position
        for position, item_id in enumerate(
            items["item_id"]
        )
    }

    profiles: dict[int, csr_matrix] = {}

    for user_id, user_data in train_interactions.groupby(
        "user_id"
    ):
        matrix_rows = []
        weights = []

        for interaction in user_data.itertuples():
            position = item_position[
                int(interaction.item_id)
            ]

            matrix_rows.append(
                item_matrix[position]
            )

            weights.append(
                float(
                    interaction.interaction_weight
                )
            )

        stacked_rows = csr_matrix(
            np.vstack(
                [
                    row.toarray()
                    for row in matrix_rows
                ]
            )
        )

        weight_array = np.asarray(
            weights,
            dtype=float,
        )

        weighted_profile = (
            stacked_rows.multiply(
                weight_array[:, np.newaxis]
            )
            .sum(axis=0)
            / weight_array.sum()
        )

        profile = csr_matrix(
            weighted_profile
        )

        profile_norm = np.sqrt(
            profile.multiply(profile).sum()
        )

        if profile_norm > 0:
            profile = profile / profile_norm

        profiles[int(user_id)] = csr_matrix(
            profile
        )

    return profiles


def save_data(
    items: pd.DataFrame,
    all_interactions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
) -> None:
    """Save catalogue and chronological interaction splits."""

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    items.to_csv(
        ITEMS_PATH,
        index=False,
    )

    all_interactions.to_csv(
        ALL_INTERACTIONS_PATH,
        index=False,
    )

    train_interactions.to_csv(
        TRAIN_INTERACTIONS_PATH,
        index=False,
    )

    test_interactions.to_csv(
        TEST_INTERACTIONS_PATH,
        index=False,
    )


def save_model(
    vectorizer: TfidfVectorizer,
    item_matrix: csr_matrix,
    user_profiles: dict[int, csr_matrix],
    items: pd.DataFrame,
) -> None:
    """Save all reusable recommendation artifacts."""

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifacts = {
        "vectorizer": vectorizer,
        "item_matrix": item_matrix,
        "user_profiles": user_profiles,
        "item_ids": items[
            "item_id"
        ].astype(int).tolist(),
    }

    joblib.dump(
        artifacts,
        MODEL_PATH,
    )

    metadata = {
        "algorithm": "Content-Based Filtering",
        "text_representation": "TF-IDF",
        "similarity_measure": "Cosine similarity",
        "interaction_weights": (
            INTERACTION_WEIGHTS
        ),
        "item_count": len(items),
        "user_count": len(user_profiles),
        "feature_count": int(
            item_matrix.shape[1]
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )


def save_vocabulary(
    vectorizer: TfidfVectorizer,
) -> None:
    """Save learned TF-IDF terms."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    vocabulary = pd.DataFrame(
        {
            "term": vectorizer.get_feature_names_out(),
        }
    )

    vocabulary.to_csv(
        VOCABULARY_PATH,
        index=False,
    )


def save_user_profile_summary(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
) -> None:
    """Save interaction counts for each evaluated user."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_summary = (
        train_interactions
        .groupby("user_id")
        .agg(
            training_interactions=(
                "item_id",
                "count",
            ),
            total_training_weight=(
                "interaction_weight",
                "sum",
            ),
        )
        .reset_index()
    )

    held_out_summary = test_interactions[
        [
            "user_id",
            "item_id",
            "interaction_type",
        ]
    ].rename(
        columns={
            "item_id": "held_out_item_id",
            "interaction_type": (
                "held_out_interaction_type"
            ),
        }
    )

    summary = train_summary.merge(
        held_out_summary,
        on="user_id",
        how="left",
    )

    summary.to_csv(
        USER_PROFILES_PATH,
        index=False,
    )


def save_training_summary(
    items: pd.DataFrame,
    all_interactions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    item_matrix: csr_matrix,
) -> None:
    """Save configuration and split information."""

    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "algorithm": "Content-Based Filtering",
        "catalogue": "Synthetic AI course catalogue",
        "item_count": len(items),
        "user_count": int(
            all_interactions[
                "user_id"
            ].nunique()
        ),
        "total_interaction_count": len(
            all_interactions
        ),
        "training_interaction_count": len(
            train_interactions
        ),
        "held_out_interaction_count": len(
            test_interactions
        ),
        "tfidf_feature_count": int(
            item_matrix.shape[1]
        ),
        "ngram_range": [
            1,
            2,
        ],
        "interaction_weights": (
            INTERACTION_WEIGHTS
        ),
        "split_method": (
            "Latest positive interaction "
            "held out per user"
        ),
        "user_profiles_built_from_training_only": True,
        "held_out_items_used_in_user_profiles": False,
        "random_state": RANDOM_STATE,
    }

    with TRAINING_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


def main() -> None:
    """Run the complete content-recommender training workflow."""

    items = create_item_catalogue()

    all_interactions = generate_interactions(
        items
    )

    (
        train_interactions,
        test_interactions,
    ) = chronological_user_split(
        all_interactions
    )

    (
        vectorizer,
        item_matrix,
    ) = fit_content_model(items)

    user_profiles = build_user_profiles(
        items=items,
        item_matrix=item_matrix,
        train_interactions=train_interactions,
    )

    save_data(
        items=items,
        all_interactions=all_interactions,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
    )

    save_model(
        vectorizer=vectorizer,
        item_matrix=item_matrix,
        user_profiles=user_profiles,
        items=items,
    )

    save_vocabulary(vectorizer)

    save_user_profile_summary(
        train_interactions=train_interactions,
        test_interactions=test_interactions,
    )

    save_training_summary(
        items=items,
        all_interactions=all_interactions,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        item_matrix=item_matrix,
    )

    print(
        "\nContent-based recommendation "
        "training completed."
    )

    print(
        f"Catalogue items: {len(items)}"
    )

    print(
        "Users: "
        f"{all_interactions['user_id'].nunique()}"
    )

    print(
        "Training interactions: "
        f"{len(train_interactions)}"
    )

    print(
        "Held-out interactions: "
        f"{len(test_interactions)}"
    )

    print(
        "TF-IDF features: "
        f"{item_matrix.shape[1]}"
    )

    print(
        f"Model saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()
