# Gaussian Naive Bayes Classification — Result Interpretation

## 1. Purpose

This document explains how to interpret the Gaussian Naive Bayes classification results.

The target labels are:

- `0` — Malignant
- `1` — Benign

The project demonstrates probabilistic classification using Bayes' theorem and Gaussian feature-distribution assumptions.

It is not a clinically validated diagnostic system.

---

## 2. How the Model Produces Predictions

Gaussian Naive Bayes calculates the probability that an observation belongs to each class.

It combines:

- the prior probability of each class;
- the likelihood of each observed feature value;
- the assumption that features are conditionally independent.

The predicted class is the class with the highest posterior probability.

---

## 3. Class Prior Probabilities

Class priors represent how common each class is in the training data.

Example:

```text
Malignant prior: 0.37
Benign prior: 0.63

Before considering feature values, the model begins with these estimated class probabilities.

A more common class receives a higher initial prior probability.

The prior is then updated using the observed feature likelihoods.

4. Conditional Independence Assumption

Naive Bayes assumes that features are independent after the class is known.

For example, within the malignant class, the model treats radius and perimeter as conditionally independent.

In reality, many medical measurements are correlated.

The assumption is therefore unlikely to be completely true.

Despite this, Naive Bayes can still perform effectively because exact independence is not always required for useful classification.

5. Gaussian Distribution Assumption

The model assumes every numerical feature follows a Gaussian distribution within each class.

For each class and feature, the algorithm estimates:

a mean;
a variance.

The file:

outputs/metrics/class_parameters.csv

contains those values.

A feature value close to a class mean receives a relatively high likelihood under that class.

A value far from the class mean generally receives a lower likelihood.

6. Variance Smoothing

The model uses:

var_smoothing = 1e-9

Variance smoothing adds a small numerical adjustment to the estimated variances.

Its purpose is to prevent numerical instability when a feature has very small variance.

The value is not a conventional regularization parameter like C in SVM.

7. Accuracy

Accuracy measures the proportion of all test predictions that were correct.

Accuracy =
Correct predictions /
Total predictions

Accuracy is useful as an overall summary.

It should not be interpreted alone because one class may be easier to predict or more common than the other.

8. Balanced Accuracy

Balanced accuracy calculates average recall across both classes.

It gives equal importance to malignant and benign cases.

This is helpful when the class frequencies differ.

A high balanced accuracy indicates that performance is not limited to the majority class.

9. Precision

Precision measures how reliable predictions for a particular class are.

For malignant cases:

Precision malignant =
Correct malignant predictions /
All malignant predictions

High malignant precision means that observations predicted as malignant are usually truly malignant.

10. Recall

Recall measures how many actual cases belonging to a class were detected.

For malignant cases:

Recall malignant =
Correct malignant predictions /
All actual malignant observations

Malignant recall is important because a malignant observation predicted as benign may be the more serious error.

11. F1-Score

The F1-score combines precision and recall.

F1 =
2 × Precision × Recall /
(Precision + Recall)

Separate F1-scores are calculated for malignant and benign classes.

A high F1-score indicates a useful balance between missed cases and false alerts.

12. ROC-AUC

ROC-AUC evaluates the model’s ability to rank the two classes across probability thresholds.

Score	Interpretation
0.50	Random-level separation
0.60–0.70	Weak
0.70–0.80	Acceptable
0.80–0.90	Strong
Above 0.90	Excellent

A high ROC-AUC indicates good separation.

It does not prove that probability estimates are well calibrated.

13. Log Loss

Log loss evaluates the quality of predicted probabilities.

It penalizes confident incorrect predictions more heavily than uncertain incorrect predictions.

Lower values are better.

A model can have good accuracy but poor log loss when it produces overconfident probabilities.

This is particularly relevant for Naive Bayes because its independence assumptions can create extreme probability values.

14. Calibration Curve

The calibration curve compares predicted probabilities with observed outcome frequencies.

A well-calibrated model should follow the diagonal line.

For example:

Among observations assigned approximately 80% benign probability, approximately 80% should actually be benign.

Naive Bayes may classify accurately while producing overconfident probabilities.

Calibration should therefore be inspected separately from accuracy and ROC-AUC.

15. Probability Distribution

The probability-distribution chart shows predicted benign probabilities for:

actual malignant observations;
actual benign observations.

Strong class separation should create:

malignant observations concentrated near low benign probabilities;
benign observations concentrated near high benign probabilities.

Overlap indicates uncertainty or difficult observations.

Extreme probabilities may indicate model confidence, but they may also reflect the independence assumption.

16. Confusion Matrix

The class meanings are:

0 — Malignant
1 — Benign
Actual	Predicted	Meaning
Malignant	Malignant	Correct malignant classification
Malignant	Benign	Malignant case missed
Benign	Malignant	Benign case incorrectly flagged
Benign	Benign	Correct benign classification

A malignant observation predicted as benign may be the highest-risk error in this demonstration.

17. Prediction Confidence

Prediction confidence is the larger class probability.

Example:

Malignant probability: 0.98
Benign probability: 0.02
Prediction confidence: 0.98

High confidence does not guarantee correctness.

Confidence should be evaluated together with:

calibration;
log loss;
external validation;
error analysis.
18. Comparison With Logistic Regression
Aspect	Gaussian Naive Bayes	Logistic Regression
Main principle	Bayes probabilities	Linear decision boundary
Distribution assumption	Gaussian features	No Gaussian feature requirement
Independence assumption	Yes	No
Feature scaling	Usually unnecessary	Often recommended
Training speed	Very fast	Fast
Feature interactions	Not directly	Limited unless added
Probability output	Native	Native
Coefficients	No simple global coefficients	Available
Correlated features	Can be problematic	More tolerant
19. Comparison With SVM
Aspect	Gaussian Naive Bayes	RBF SVM
Training speed	Very fast	Higher cost
Scaling	Usually unnecessary	Required
Nonlinear boundaries	Through distributions	Through kernel
Probability output	Native	Requires calibration
Interpretability	Probabilistic assumptions	Limited
Hyperparameter tuning	Limited	Important
Large datasets	Often suitable	Can become expensive
20. Comparison With Tree Models
Aspect	Gaussian Naive Bayes	Decision Tree	Random Forest
Main principle	Probability distributions	Rule-based splits	Tree ensemble
Nonlinear interactions	Limited	Strong	Strong
Scaling required	No	No	No
Interpretability	Assumption-based	Very high	Moderate
Feature importance	Not conventional	Available	Available
Training speed	Very fast	Fast	Moderate
Correlated features	Can reduce quality	Usually manageable	Usually manageable
21. Strengths

Gaussian Naive Bayes offers:

simple probabilistic reasoning;
very fast training;
fast prediction;
native probability estimates;
low memory requirements;
strong baseline performance;
support for incremental learning;
usefulness with limited training data.
22. Limitations

Its major limitations are:

conditional independence assumption;
Gaussian feature-distribution assumption;
limited modelling of interactions;
sensitivity to correlated features;
potentially overconfident probabilities;
lower flexibility than ensemble and kernel methods.
23. Recommended Improvements

Future improvements could include:

Cross-validation.
Testing different var_smoothing values.
Probability calibration.
Correlation analysis.
Feature-selection experiments.
Removal of highly correlated features.
Comparison with Quadratic Discriminant Analysis.
Comparison with Multinomial Naive Bayes on text data.
Threshold tuning.
Incremental-learning demonstration using partial_fit.
FastAPI deployment.
Automated unit tests.
Model-comparison dashboard.
24. Final Conclusion

Gaussian Naive Bayes provides a fast and interpretable probabilistic baseline.

Its main advantages are:

speed;
simplicity;
probability output;
small computational requirements;
effectiveness with limited data.

Its main disadvantages are:

strong distribution assumptions;
independence assumptions;
limited interaction modelling;
potentially overconfident probabilities.

In this repository, Gaussian Naive Bayes demonstrates:

Bayes' theorem;
class priors;
Gaussian likelihoods;
posterior probabilities;
probability-based classification;
calibration analysis;
comparison with linear, distance-based, kernel, and tree-based models.

The results are educational and must not be treated as clinical diagnostic evidence.