import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class Metric:
	name: str
	threshold: float
	fn: Callable
	ge: bool  # is metric a ge metric or a le metric
	
	def test(self, value):
		if self.ge:
			return value >= self.threshold
		return value <= self.threshold
	
	def __repr__(self):
		op = ">=" if self.ge else "<="
		return f"{self.name} {op} {self.threshold}"

def accuracy(y_pred, y_true):
	return (y_pred == y_true).mean()

def precision(y_pred, y_true):
	if len(y_pred) == 0:
		return 0.0
	tp = ((y_pred == 1) & (y_true == 1)).sum()
	fp = ((y_pred == 1) & (y_true == 0)).sum()
	if tp + fp == 0:
		return 0.0
	return tp / (tp + fp)

def recall(y_pred, y_true):
	if len(y_pred) == 0:
		return 0.0
	tp = ((y_pred == 1) & (y_true == 1)).sum()
	fn = ((y_pred == 0) & (y_true == 1)).sum()
	if tp + fn == 0:
		return 0.0
	return tp / (tp + fn)

def f1_score(y_pred, y_true):
	p = precision(y_pred, y_true)
	r = recall(y_pred, y_true)
	if p + r == 0:
		return 0.0
	return 2 * (p * r) / (p + r)

def specificity(y_pred, y_true):
	if len(y_pred) == 0:
		return 0.0
	tn = ((y_pred == 0) & (y_true == 0)).sum()
	fp = ((y_pred == 1) & (y_true == 0)).sum()
	if tn + fp == 0:
		return 0.0
	return tn / (tn + fp)

def balanced_accuracy(y_pred, y_true):
	return (recall(y_pred, y_true) + specificity(y_pred, y_true)) / 2

def positive_rate(y_pred, y_true=None):
	if len(y_pred) == 0:
		return 0.0
	return (y_pred == 1).mean()

def negative_rate(y_pred, y_true=None):
	if len(y_pred) == 0:
		return 0.0
	return (y_pred == 0).mean()

def divergence_from_mean(X, mean):
	if mean == 0:
		return np.abs(X)
	return np.abs(X / mean - 1)

def max_divergence_from_mean(values):
	if len(values) == 0:
		return 0.0
	mean = np.mean(values)
	if mean == 0:
		return 0.0
	return np.max(np.abs(values / mean - 1))

def mean_divergence(prediction, expectation):
	divergence = np.abs(prediction - expectation)
	return divergence.mean()

def disparate_impact_ratio(y_pred_group1, y_pred_group2):
	rate1 = positive_rate(y_pred_group1)
	rate2 = positive_rate(y_pred_group2)
	if rate1 == 0 or rate2 == 0:
		return 0.0
	return min(rate1 / rate2, rate2 / rate1)

def change_rate(y_new, y_old):
	return (y_new != y_old).mean()

def stability_score(y_new, y_old):
	return (y_new == y_old).mean()

def prediction_variance(predictions_list):
	if len(predictions_list) == 0:
		return 0.0
	stacked = np.stack(predictions_list, axis=0)
	return np.mean(np.var(stacked, axis=0))

def prediction_distribution_distance(y_pred1, y_pred2):
	dist1 = np.bincount(y_pred1, minlength=2) / len(y_pred1)
	dist2 = np.bincount(y_pred2, minlength=2) / len(y_pred2)
	return 0.5 * np.sum(np.abs(dist1 - dist2))

def calibration_error(y_pred, y_true):
	pred_rate = positive_rate(y_pred)
	true_rate = positive_rate(y_true)
	return abs(pred_rate - true_rate)

def prediction_flip_rate(y_pred_modified, y_pred_original, y_true):
	correct_original = (y_pred_original == y_true)
	correct_modified = (y_pred_modified == y_true)
	flipped = (correct_original != correct_modified)
	return flipped.mean()

def consistency_score(predictions_list):
	if len(predictions_list) <= 1:
		return 1.0
	stacked = np.stack(predictions_list, axis=0)

	unanimous = np.all(stacked == stacked[0], axis=0)
	return unanimous.mean()

ACCURACY_METRIC = Metric(
	name="Accuracy",
	threshold=0.80,
	fn=accuracy,
	ge=True
)

ACCURACY_STRICT = Metric(
	name="Accuracy",
	threshold=0.85,
	fn=accuracy,
	ge=True
)

PRECISION_METRIC = Metric(
	name="Precision",
	threshold=0.70,
	fn=precision,
	ge=True
)

RECALL_METRIC = Metric(
	name="Recall",
	threshold=0.65,
	fn=recall,
	ge=True
)

F1_METRIC = Metric(
	name="F1 Score",
	threshold=0.70,
	fn=f1_score,
	ge=True
)

BALANCED_ACC_METRIC = Metric(
	name="Balanced Acc",
	threshold=0.75,
	fn=balanced_accuracy,
	ge=True
)

DIVERGENCE_FROM_MEAN = Metric(
	name="Mean Div",
	threshold=0.10,
	fn=divergence_from_mean,
	ge=False
)

DIVERGENCE_STRICT = Metric(
	name="Mean Div",
	threshold=0.05,
	fn=divergence_from_mean,
	ge=False
)

DISPARATE_IMPACT = Metric(
	name="Disp Impact",
	threshold=0.80,
	fn=disparate_impact_ratio,
	ge=True
)

CHANGE_RATE = Metric(
	name="Change Rate",
	threshold=0.05,
	fn=change_rate,
	ge=False
)

CHANGE_RATE_STRICT = Metric(
	name="Change Rate",
	threshold=0.03,
	fn=change_rate,
	ge=False
)

STABILITY_METRIC = Metric(
	name="Stability",
	threshold=0.95,
	fn=stability_score,
	ge=True
)

FLIP_RATE_METRIC = Metric(
	name="Flip Rate",
	threshold=0.05,
	fn=prediction_flip_rate,
	ge=False
)

DIST_SHIFT_METRIC = Metric(
	name="Dist Shift",
	threshold=0.10,
	fn=prediction_distribution_distance,
	ge=False
)

CALIBRATION_METRIC = Metric(
	name="Calibration",
	threshold=0.10,
	fn=calibration_error,
	ge=False
)

STANDARD_PERFORMANCE_METRICS = [
	ACCURACY_METRIC,
	PRECISION_METRIC,
	RECALL_METRIC,
	F1_METRIC
]

FAIRNESS_METRICS = [
	DIVERGENCE_FROM_MEAN,
	DISPARATE_IMPACT,
    CALIBRATION_METRIC
]

ROBUSTNESS_METRICS = [
	CHANGE_RATE,
    DIST_SHIFT_METRIC,
    CALIBRATION_METRIC
]

