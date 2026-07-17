# Support Vector Machine Classification — Result Interpretation

## 1. Purpose

This document explains how to interpret the Support Vector Machine classification results.

The model classifies observations as:

- `0` — Malignant
- `1` — Benign

The project demonstrates nonlinear classification, feature scaling, margin-based learning, and probability calibration.

It is not a clinically validated diagnostic system.

---

## 2. How SVM Makes Decisions

Support Vector Machine attempts to identify a decision boundary that separates the two classes.

Rather than only finding any possible boundary, it seeks a boundary with a strong margin between the classes.

The observations closest to the boundary are the most influential.

These influential observations are called support vectors.

The model uses an RBF kernel, allowing the classification boundary to be nonlinear.

---

## 3. RBF Kernel Interpretation

The Radial Basis Function kernel measures similarity between observations.

Observations that are close together in the scaled feature space receive higher similarity values.

This allows the model to form curved or complex boundaries rather than relying on one straight linear separator.

The RBF kernel is useful when class relationships are not linearly separable.

---

## 4. Feature Scaling

The SVM is highly sensitive to feature scale.

The pipeline applies `StandardScaler` before training.

This ensures that one feature does not dominate merely because its numerical values are larger.

Standardization approximately transforms each feature to:

- mean = 0;
- standard deviation = 1.

Scaling parameters are learned only from training data through the pipeline.

---

## 5. Interpretation of C

The model uses:

```text
C = 2.0
C controls the penalty applied to classification errors.

A smaller value of C:

allows more training errors;
encourages a wider margin;
applies stronger regularization;
may reduce overfitting;
may increase underfitting.

A larger value of C:

penalizes training errors more strongly;
attempts to classify more training observations correctly;
may create a narrower or more complex margin;
may increase overfitting risk.

The best value should be selected through cross-validation.

6. Interpretation of Gamma

The model uses:

gamma = "scale"

Gamma controls how far the influence of a training observation extends.

A small gamma:

creates broad influence;
produces smoother decision boundaries;
may underfit.

A large gamma:

creates highly localized influence;
produces more complex boundaries;
may overfit.

The interaction between C and gamma is important.

Both should normally be tuned together.

7. Probability Calibration

The SVM is wrapped with:

CalibratedClassifierCV

Sigmoid calibration converts SVM outputs into probability estimates.

The calibration process uses five internal folds.

This allows the model to produce outputs such as:

Malignant probability: 0.92
Benign probability: 0.08

These probabilities are useful for ranking and threshold decisions.

They should not automatically be interpreted as real clinical risk without external validation.

8. Accuracy

Accuracy measures the proportion of all test predictions that were correct.

Accuracy = Correct predictions / Total predictions

Example:

Accuracy of 0.97 means approximately 97% of test observations were classified correctly.

Accuracy alone may hide weaker performance for one class.

It should be considered together with balanced accuracy, class-specific recall, precision, F1-score, and the confusion matrix.

9. Balanced Accuracy

Balanced accuracy calculates the average recall across both classes.

It gives equal importance to malignant and benign observations.

This is helpful when the two classes contain different numbers of samples.

A strong balanced accuracy means the model is performing well across both classes rather than mainly performing well for the larger class.

10. Precision

Precision measures how many predictions for a class were correct.

For malignant observations:

Precision malignant =
Correct malignant predictions /
All malignant predictions

High malignant precision means that observations predicted as malignant are usually truly malignant.

For benign observations, precision measures how reliable benign predictions are.

11. Recall

Recall measures how many actual observations belonging to a class were detected.

For malignant cases:

Recall malignant =
Correct malignant predictions /
All actual malignant observations

Malignant recall is particularly important because a malignant case predicted as benign may be a serious error.

For benign cases, recall measures how many truly benign observations were identified correctly.

12. F1-Score

The F1-score balances precision and recall.

F1 =
2 × Precision × Recall /
(Precision + Recall)

A high F1-score indicates that the model maintains a good balance between missing cases and creating incorrect alerts.

Separate F1-scores should be reviewed for malignant and benign classes.

13. ROC-AUC

ROC-AUC measures the model’s ability to separate classes across probability thresholds.

ROC-AUC	Interpretation
0.50	Random-level separation
0.60–0.70	Weak
0.70–0.80	Acceptable
0.80–0.90	Strong
Above 0.90	Excellent

A strong ROC-AUC indicates good ranking performance.

It does not automatically identify the safest threshold.

14. Precision–Recall Curve

The precision-recall curve shows the trade-off between:

detecting more positive observations;
maintaining reliable positive predictions.

Changing the classification threshold changes precision and recall.

This curve is especially useful when:

classes are imbalanced;
one class is operationally important;
false positives and false negatives have different costs.
15. Calibration Curve

The calibration curve compares predicted probabilities with observed frequencies.

A well-calibrated model should produce points near the diagonal line.

For example:

Among observations assigned approximately 80% benign probability, about 80% should actually be benign.

A model can classify accurately while still producing poorly calibrated probabilities.

Therefore, probability calibration and classification discrimination are different concepts.

16. Confusion Matrix

The target classes are:

0 — Malignant
1 — Benign
Actual	Predicted	Interpretation
Malignant	Malignant	Correct malignant prediction
Malignant	Benign	Malignant observation missed
Benign	Malignant	Benign observation incorrectly flagged
Benign	Benign	Correct benign prediction

A malignant observation predicted as benign may be the highest-risk error.

This should be reviewed through malignant recall and confusion-matrix values.

17. Prediction Confidence

Prediction confidence is calculated as the larger of the two calibrated class probabilities.

Example:

Malignant probability: 0.87
Benign probability: 0.13
Prediction confidence: 0.87

High confidence does not guarantee that the prediction is correct.

Confidence must be checked against calibration results and external validation.

18. Comparison With KNN
Aspect	SVM	KNN
Learning style	Margin-based	Neighbor-based
Prediction speed	Usually faster after training	Can be slower
Training cost	Higher	Low
Scaling required	Yes	Yes
Nonlinear modelling	Through kernels	Through local neighborhoods
High-dimensional data	Often effective	Can struggle
Interpretability	Moderate to low	Neighbor-based
Main hyperparameters	C and gamma	Number of neighbors and distance

SVM often performs well in higher-dimensional spaces.

KNN is more intuitive but can be affected by the curse of dimensionality.

19. Comparison With Logistic Regression
Aspect	SVM RBF	Logistic Regression
Boundary	Nonlinear	Mainly linear
Feature scaling	Required	Recommended
Interpretability	Lower	Higher
Coefficients	Not directly available	Available
Complex relationships	Strong	Limited
Probability output	Requires calibration	Native probability model
Training cost	Higher	Lower

Logistic Regression provides clearer coefficient-based interpretation.

RBF SVM can capture more complex relationships.

20. Comparison With Tree Models
Aspect	SVM	Decision Tree	Random Forest
Scaling required	Yes	No	No
Nonlinear relationships	Yes	Yes	Yes
Feature importance	Not directly	Available	Available
Interpretability	Lower	Very high	Moderate
Stability	Usually strong	Lower	Strong
Large datasets	Can become slow	Suitable	Suitable
Main mechanism	Maximum margin	Rules and splits	Tree ensemble
21. Strengths

The SVM model provides:

strong nonlinear classification;
high-dimensional effectiveness;
regularization through C;
flexible kernels;
class weighting;
calibrated probability outputs;
strong separation performance.
22. Limitations

The main limitations are:

sensitivity to scaling;
sensitivity to C and gamma;
training cost on large datasets;
limited direct interpretability;
no simple RBF feature coefficients;
additional cost for probability calibration;
possible overfitting with aggressive hyperparameters.
23. Recommended Improvements

Future improvements should include:

Cross-validation.
Grid search over C.
Grid search over gamma.
Comparison of linear, polynomial, and RBF kernels.
Threshold optimization.
Probability-calibration comparison.
Learning curves.
Permutation feature importance.
SHAP-compatible surrogate explanations.
Runtime comparison.
FastAPI deployment.
Automated unit tests.
Input schema validation.
Model-comparison dashboard.
24. Final Conclusion

Support Vector Machine provides a powerful margin-based approach for binary classification.

The RBF kernel enables the model to represent nonlinear relationships.

Its main advantages are:

strong class-separation ability;
effectiveness in high-dimensional spaces;
flexible nonlinear boundaries;
explicit regularization controls.

Its main disadvantages are:

limited interpretability;
scaling dependence;
sensitivity to hyperparameters;
increased computational cost.

In this repository, SVM demonstrates:

maximum-margin learning;
kernel methods;
feature scaling;
probability calibration;
nonlinear classification;
comparison with distance-based, linear, tree-based, and ensemble models.

The results are educational and must not be treated as clinical diagnostic evidence.