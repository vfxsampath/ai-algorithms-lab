# Isolation Forest — Result Interpretation

## 1. Purpose

This document explains the Isolation Forest outputs and their practical meaning.

The model is trained without anomaly labels.

Controlled synthetic anomalies are used only for held-out educational evaluation.

---

## 2. How Isolation Forest Works

Isolation Forest repeatedly:

1. selects a feature randomly;
2. selects a split value randomly;
3. partitions observations;
4. continues until observations are isolated.

Unusual observations tend to be isolated with shorter tree paths.

Common observations generally require more partitions.

The final anomaly score combines information across many isolation trees.

---

## 3. Training Data

The model is fitted only on the saved training dataset.

The training data are assumed to represent the operating distribution sufficiently well.

They may still contain naturally unusual records because the source dataset does not provide verified anomaly labels.

---

## 4. Held-Out Normal Observations

Held-out normal observations come from the original dataset but were not used during fitting.

They provide an indication of how the learned model treats unseen records from the same source distribution.

They are treated as normal for controlled evaluation, although some may be naturally unusual.

---

## 5. Synthetic Anomalies

Synthetic anomalies were created by strongly perturbing several features of held-out observations.

They provide known anomaly labels for demonstrating evaluation metrics.

They do not represent:

- confirmed real fraud;
- real equipment failures;
- real operational incidents;
- independent external validation.

Performance on synthetic anomalies may be easier than performance on subtle real-world anomalies.

---

## 6. Contamination

The model uses:

```text
contamination = 0.05
Contamination helps determine the score threshold used to separate inliers and outliers.

It represents an expected anomaly proportion for threshold setting.

It is not a discovered fact about the dataset.

A contamination value that is too high may create excessive false alarms.

A value that is too low may cause the model to miss meaningful anomalies.

7. Raw Prediction

Isolation Forest returns:

1 = inlier
-1 = outlier

The project converts these values to:

0 = normal
1 = anomaly

This conversion makes the anomaly class consistent with standard binary-classification metrics.

8. Decision Function

The decision function indicates which side of the learned threshold an observation falls on.

Typical interpretation:

positive value: inlier side;
negative value: outlier side;
near zero: close to the decision boundary.

Values close to zero indicate uncertain or borderline records.

9. Project Anomaly Score

This project calculates:

anomaly score = -decision function

Under this convention:

larger values indicate greater anomaly evidence;
smaller values indicate greater normality;
values around zero lie near the threshold.

The score is useful for ranking records even when a binary alert is not required.

10. Score Samples

score_samples() returns the model's raw sample scores.

Lower values are associated with more abnormal observations.

The exact values should be interpreted comparatively rather than as probabilities.

An Isolation Forest score is not a percentage chance of fraud or failure.

11. Accuracy

Accuracy reports the percentage of all controlled evaluation records classified correctly.

Accuracy can be misleading when anomalies are rare.

A model that predicts every record as normal may achieve high accuracy while detecting no anomalies.

Therefore, anomaly recall, precision, F1-score, ROC-AUC, and average precision are more informative.

12. Balanced Accuracy

Balanced accuracy averages recall across:

normal observations;
anomaly observations.

It prevents the larger class from dominating the reported performance.

A high balanced accuracy indicates that both classes are being recognized reasonably well.

13. Anomaly Precision

Anomaly precision answers:

Of all records flagged as anomalies, how many were labelled anomalies in the controlled evaluation set?

High precision means fewer false alarms.

Low precision means many ordinary held-out records were flagged.

In practical systems, low precision can create investigation overload.

14. Anomaly Recall

Anomaly recall answers:

Of all labelled anomalies, how many did the model detect?

High recall means fewer anomalies were missed.

Low recall means many anomalies passed through as normal.

The preferred balance depends on the cost of missed incidents versus false alerts.

15. Anomaly F1-Score

The F1-score balances anomaly precision and recall.

It is useful when:

anomaly labels are rare;
both false alarms and missed anomalies matter;
one combined threshold-based metric is needed.

It does not evaluate ranking performance across all thresholds.

16. ROC-AUC

ROC-AUC measures how well the anomaly score ranks labelled anomalies above normal observations across thresholds.

A high ROC-AUC indicates good ranking separation.

It does not guarantee useful precision when true anomalies are extremely rare.

17. Average Precision

Average precision summarizes the precision-recall relationship across score thresholds.

It is particularly useful for rare-event detection.

Higher values indicate that true anomalies tend to appear near the top of the ranked anomaly list.

Average precision should be interpreted relative to the anomaly prevalence in the evaluation dataset.

18. Confusion Matrix

The matrix contains:

Actual	Predicted	Meaning
Normal	Normal	Correctly accepted normal record
Normal	Anomaly	False alarm
Anomaly	Normal	Missed anomaly
Anomaly	Anomaly	Correctly detected anomaly

False alarms increase operational workload.

Missed anomalies increase operational risk.

19. Anomaly-Score Distribution

The distribution chart compares scores for:

held-out normal observations;
synthetic anomalies.

A useful model should generally assign higher anomaly scores to synthetic anomalies.

Overlap indicates difficult or borderline observations.

Perfect separation may indicate that synthetic anomalies are much easier to detect than realistic anomalies.

20. PCA Visualization

The PCA chart projects training and evaluation data into two dimensions.

It helps visualize:

broad training-data structure;
held-out normal observations;
synthetic anomalies;
records flagged by the model.

The Isolation Forest uses all transformed features.

PCA is used only for visualization.

Overlap in the PCA chart does not necessarily mean the model cannot separate records in the complete feature space.

21. Why Scaling Is Included

Isolation Forest does not rely on Euclidean distance in the same way as KNN or K-Means.

Scaling is included to:

create a consistent pipeline;
simplify downstream visualization;
keep preprocessing reusable;
standardize feature handling across the project.

The scaler is fitted only on training data.

22. Training Records Flagged as Outliers

Even though the training data are treated as the reference distribution, the contamination threshold causes some training observations to be flagged as outliers.

This is expected.

It reflects the threshold assumption, not verified anomaly truth.

Training outliers should be reviewed rather than automatically deleted.

23. Held-Out Example Prediction

predict.py selects one labelled evaluation record.

It reports:

observation source;
actual controlled label;
predicted status;
raw prediction;
decision function;
anomaly score;
score-samples value.

The example demonstrates inference using a saved model.

It does not represent independent external validation.

24. Business Interpretation

Isolation Forest can help prioritize records for review.

A practical workflow could be:

score new records;
rank by anomaly score;
investigate the highest-ranked cases;
collect analyst feedback;
refine features and thresholds;
monitor alert volumes and drift.

The model should support investigation, not automatically determine guilt, fraud, failure, or cause.

25. Threshold Selection

The binary prediction threshold should reflect business costs.

A stricter anomaly threshold may:

reduce false alarms;
lower recall.

A more permissive threshold may:

detect more anomalies;
increase investigation workload.

Thresholds should be selected using validation data and operational requirements.

26. Strengths

Isolation Forest provides:

unsupervised training;
efficient tree-based scoring;
anomaly ranking;
support for high-dimensional data;
low dependence on distribution assumptions;
scalable screening.
27. Limitations

Its main limitations are:

contamination sensitivity;
limited direct explanation;
possible difficulty with subtle anomalies;
no built-in cause analysis;
sensitivity to training-distribution quality;
possible performance decline under data drift;
synthetic evaluation limitations.
28. Recommended Improvements

Future improvements should include:

Compare contamination settings.
Add cross-validation-style stability analysis.
Compare Isolation Forest with Local Outlier Factor.
Compare with One-Class SVM.
Add feature-contribution explanations.
Add real labelled anomaly data.
Evaluate threshold-cost trade-offs.
Add drift monitoring.
Add time-based evaluation.
Add alert-ranking analysis.
Add unit tests.
Add a model card.
29. Final Conclusion

Isolation Forest detects unusual observations by measuring how quickly random trees isolate them.

The project uses:

training-only model fitting;
held-out normal observations;
synthetic anomalies for controlled evaluation;
anomaly-ranking scores;
threshold-based predictions;
PCA visualization.

Its principal strength is unsupervised anomaly scoring.

Its main weakness is that an unusual pattern does not automatically reveal its meaning, cause, or business risk.

The synthetic-anomaly evaluation demonstrates the workflow but does not establish real-world anomaly-detection performance.