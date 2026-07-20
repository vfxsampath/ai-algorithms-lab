# AI Algorithm Selection Guide

This guide identifies a suitable starting category. Final selection should be based on experiments, validation, interpretability, and business requirements.

```mermaid
flowchart TD
    A["What outcome do you need?"]

    A --> B{"Do you have a target variable?"}
    A --> C{"Are observations ordered by time?"}
    A --> D{"Do you need recommendations?"}

    B -->|Yes| E{"Target type?"}
    B -->|No| F{"Pattern-discovery objective?"}

    C -->|Yes| G["Time-Series Forecasting"]
    D -->|Yes| H["Recommendation Systems"]

    E -->|Category| I["Classification"]
    E -->|Continuous number| J["Regression"]

    F -->|Discover groups| K["Clustering"]
    F -->|Reduce dimensions| L["Dimensionality Reduction"]
    F -->|Find unusual observations| M["Anomaly Detection"]
    F -->|Find co-occurring items| N["Association Rules"]
```

## Classification Selection

```mermaid
flowchart TD
    A["Classification Requirement"]

    A --> B{"Primary priority?"}

    B -->|Simple baseline| C["Logistic Regression"]
    B -->|Explainable rules| D["Decision Tree"]
    B -->|Strong general performance| E["Random Forest"]
    B -->|Similarity-based| F["KNN"]
    B -->|High-dimensional nonlinear| G["SVM"]
    B -->|Fast probabilistic baseline| H["Naive Bayes"]
    B -->|High-performance tabular data| I["XGBoost"]
```

## Regression Selection

```mermaid
flowchart TD
    A["Regression Requirement"]

    A --> B{"Primary priority?"}

    B -->|Simple and interpretable| C["Linear Regression"]
    B -->|Control coefficient size| D["Ridge Regression"]
    B -->|Feature selection| E["Lasso Regression"]
    B -->|Explainable nonlinear rules| F["Decision Tree Regression"]
    B -->|Strong general performance| G["Random Forest Regression"]
    B -->|High-performance tabular data| H["XGBoost Regression"]
```

## Unsupervised Selection

```mermaid
flowchart TD
    A["Unsupervised Requirement"]

    A --> B{"Desired result?"}

    B -->|Compact groups| C["K-Means"]
    B -->|Irregular groups and noise| D["DBSCAN"]
    B -->|Hierarchy| E["Hierarchical Clustering"]
    B -->|Linear feature compression| F["PCA"]
    B -->|Two-dimensional visualization| G["t-SNE or UMAP"]
    B -->|General anomaly detection| H["Isolation Forest"]
    B -->|Local anomalies| I["Local Outlier Factor"]
    B -->|Items purchased together| J["Apriori or FP-Growth"]
```

## Selection Checklist

Before choosing a final algorithm, examine:

1. Is the target labelled?
2. Is the target categorical, numerical, or temporal?
3. How many observations and features are available?
4. Are classes imbalanced?
5. Is explainability required?
6. How costly are false positives and false negatives?
7. Is prediction latency important?
8. Does the model need probability estimates?
9. Can the model be monitored after deployment?
10. Does a simpler baseline perform nearly as well?

## Professional Model-Selection Process

```text
Define the business problem
        ↓
Prepare train and test data
        ↓
Build a simple baseline
        ↓
Test several suitable algorithms
        ↓
Compare relevant evaluation metrics
        ↓
Consider interpretability and cost
        ↓
Select and validate the final solution
```