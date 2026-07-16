# K-Nearest Neighbors Classification — Result Interpretation

## 1. Purpose

This document explains how to interpret the K-Nearest Neighbors classification results.

The target labels are:

- `0` — Malignant
- `1` — Benign

The model is intended to demonstrate similarity-based machine learning and should not be treated as a clinically validated diagnostic system.

---

## 2. How KNN Produces a Prediction

KNN does not estimate a conventional mathematical model during training.

Instead, it stores the training examples.

For a new observation, it:

1. scales the input features;
2. calculates distances to training observations;
3. identifies the seven nearest observations;
4. examines their class labels;
5. applies distance-weighted voting;
6. returns the final class and probability.

Closer observations receive more influence because the model uses:

```text
weights = "distance"

3. Importance of Scaling

KNN is distance-based.

Without scaling, features with larger numerical values could dominate the distance calculation.

The project uses StandardScaler, which centers each feature around zero and scales it using the training-set standard deviation.

The scaler is included inside the saved pipeline, ensuring identical preprocessing during training and prediction.

4. Meaning of K

The model uses:

n_neighbors = 7

This means each prediction considers seven nearby training examples.

A smaller value of k:

creates more local decision boundaries;
may capture detailed patterns;
can be sensitive to noise;
can overfit.

A larger value of k:

produces smoother predictions;
reduces sensitivity to individual observations;
may underfit;
may favor the majority class.

The best value should be selected using cross-validation rather than assumption.

5. Distance Weighting

The model uses distance-based voting.

Closer neighbors contribute more strongly than distant neighbors.

This can improve predictions when the closest observations are more relevant than the full set of selected neighbors.

However, predictions may become sensitive to extremely close observations or noisy data points.

6. Accuracy

Accuracy measures the proportion of all test observations predicted correctly.

Accuracy = Correct Predictions / Total Predictions

Accuracy provides a useful overall summary but should not be interpreted alone.

Class-specific recall is especially important when one type of error has greater consequences.

7. Balanced Accuracy

Balanced accuracy calculates the average recall across the two classes.

It provides a better view than ordinary accuracy when class sizes differ.

A high balanced accuracy means the model performs well for both malignant and benign observations rather than mainly succeeding on the larger class.

8. Precision

Precision indicates how often predictions for a class are correct.

For malignant cases:

Precision malignant =
Correct malignant predictions /
All malignant predictions

For benign cases:

Precision benign =
Correct benign predictions /
All benign predictions

High malignant precision means that observations predicted as malignant are usually malignant.

9. Recall

Recall indicates how many actual members of a class were detected.

For malignant cases:

Recall malignant =
Correct malignant predictions /
All actual malignant observations

Malignant recall is important because a malignant observation predicted as benign may represent the more serious error.

10. F1-Score

The F1-score balances precision and recall.

F1 =
2 × Precision × Recall /
(Precision + Recall)

A high F1-score indicates that the model maintains a useful balance between missed cases and incorrect alerts.

11. ROC-AUC

ROC-AUC measures the model's ability to separate the classes across classification thresholds.

General interpretation:

Score	Meaning
0.50	Random-level separation
0.60–0.70	Weak
0.70–0.80	Acceptable
0.80–0.90	Strong
Above 0.90	Excellent

A high ROC-AUC indicates strong class-separation ability.

It does not automatically identify the safest prediction threshold.

12. Confusion Matrix

The class order is:

0 — Malignant
1 — Benign
Actual	Predicted	Interpretation
Malignant	Malignant	Correct malignant prediction
Malignant	Benign	Malignant observation missed
Benign	Malignant	Benign observation incorrectly flagged
Benign	Benign	Correct benign prediction

A malignant observation predicted as benign may be the most important error to minimize.

13. Probability Interpretation

KNN probabilities are based on the class composition and weighting of nearby observations.

Example:

Malignant probability: 0.82
Benign probability: 0.18

This means the weighted neighborhood strongly favors the malignant class.

It should not automatically be interpreted as a calibrated clinical probability.

14. Neighbor Distances

The file:

outputs/predictions/example_neighbors.csv

contains the seven nearest training observations for one example.

A smaller distance means the training observation is more similar to the query observation after scaling.

Neighbor distances help explain why KNN produced a particular prediction.

However, similarity does not prove causal or medical equivalence.

15. High-Dimensional Data

The dataset contains many features.

In high-dimensional spaces, distances can become less informative because observations may appear similarly far apart.

This issue is commonly called the curse of dimensionality.

Possible improvements include:

feature selection;
Principal Component Analysis;
removal of irrelevant features;
metric learning;
dimensionality reduction before KNN.
16. Comparison With Other Models
Aspect	KNN	Logistic Regression	Decision Tree	Random Forest
Main principle	Neighbor similarity	Linear probability model	Rule-based splits	Tree ensemble
Training cost	Low	Low	Moderate	Higher
Prediction cost	Higher	Low	Low	Moderate
Scaling required	Yes	Usually	No	No
Nonlinear patterns	Yes	Limited	Yes	Yes
Interpretability	Neighbor-based	Coefficients	Very high	Moderate
Large datasets	Less suitable	Suitable	Suitable	Suitable
High dimensions	Often weaker	Often suitable	Suitable	Suitable
17. Recommended Improvements

Future improvements should include:

Cross-validation for selecting k.
Comparison of uniform and distance weighting.
Comparison of Euclidean and Manhattan distance.
GridSearchCV for hyperparameter tuning.
PCA before KNN.
Feature-selection experiments.
Calibration analysis.
Threshold analysis.
Learning curves.
Runtime comparison.
FastAPI prediction endpoint.
Automated unit tests.
18. Final Conclusion

K-Nearest Neighbors provides an intuitive similarity-based classification approach.

Its main strengths are simplicity, nonlinear decision boundaries, and neighborhood-level explanations.

Its main weaknesses are prediction cost, sensitivity to scaling, sensitivity to irrelevant features, and reduced effectiveness in high-dimensional datasets.

In this repository, KNN demonstrates:

instance-based learning;
feature scaling;
distance-based classification;
weighted voting;
probability prediction;
nearest-neighbor explanation;
comparison with parametric, tree-based, and ensemble models.

The results are educational and must not be interpreted as clinical diagnostic evidence.