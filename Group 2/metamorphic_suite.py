import numpy as np
import pandas as pd
from utils import *
from term_styling import style, fg, bg
from metrics import Metric


class MetamorphicSuite:
	def __init__(self, metamorphic_fn, test_type, tries,
				 problem_columns, classical_metrics,
				 test_metrics, verbosity=0, **transform_kwargs):
		self.metamorphic_fn = metamorphic_fn
		self.tries = tries
		self.classical_metrics = classical_metrics if isinstance(classical_metrics, list) else [classical_metrics]
		self.test_metrics = test_metrics if isinstance(test_metrics, list) else [test_metrics]
		self.verbosity = verbosity
		self.test_type = test_type
		self.problem_columns = problem_columns
		self.transform_kwargs = transform_kwargs
	
	def metamorphic_test(self, model, X, y, columns, title):
		if self.verbosity == 2:
			self._print_header(title)
		
		n_classical = len(self.classical_metrics)
		n_test = len(self.test_metrics)
		
		cm_measurements = np.zeros((self.tries, n_classical))
		tm_measurements = np.zeros((self.tries, n_test))
		cm_passes = np.zeros((self.tries, n_classical))
		tm_passes = np.zeros((self.tries, n_test))
		
		y_pred_orig = model.predict(X)
		
		for trial_idx in range(self.tries):
			X_alt = self.metamorphic_fn(X, columns, **self.transform_kwargs)
			y_pred = model.predict(X_alt)
			
			for m_idx, metric in enumerate(self.classical_metrics):
				measurement = metric.fn(y_pred, y)
				cm_measurements[trial_idx, m_idx] = measurement
				cm_passes[trial_idx, m_idx] = 1 if metric.test(measurement) else 0
			
			for m_idx, metric in enumerate(self.test_metrics):
				measurement = 0
				if 'flip' in metric.name.lower():
					measurement = metric.fn(y_pred, y_pred_orig, y)
				else:
					measurement = metric.fn(y_pred, y_pred_orig)
				tm_measurements[trial_idx, m_idx] = measurement
				tm_passes[trial_idx, m_idx] = 1 if metric.test(measurement) else 0
		
		if self.verbosity == 2:
			self._print_detailed_results(cm_measurements, tm_measurements, cm_passes, tm_passes)
		
		if self.verbosity >= 1:
			self._print_summary(title, cm_passes, tm_passes)
		
		return {
			'classical_passes': cm_passes.sum(axis=0).astype(int),
			'test_passes': tm_passes.sum(axis=0).astype(int),
			'tests': self.tries,
			'classical_measurements': cm_measurements,
			'test_measurements': tm_measurements
		}
	
	def _print_header(self, title):
		title_string = f"=== {self.test_type} Testing {title} |"
		for metric in self.classical_metrics:
			title_string += f" {metric.name} {style.bold}{metric.threshold}{style.reset} |"
		for metric in self.test_metrics:
			title_string += f" {metric.name} {style.bold}{metric.threshold}{style.reset} |"
		title_string += "="
		
		dashes = max(80, len(title_string) - 38)
		print("=" * dashes)
		print(title_string)
		print("=" * dashes)
	
	def _print_detailed_results(self, cm_measurements, tm_measurements, cm_passes, tm_passes):
		for trial_idx in range(self.tries):
			line = f"Trial {trial_idx}\t|"
			
			# Classical metrics
			for m_idx, metric in enumerate(self.classical_metrics):
				val = cm_measurements[trial_idx, m_idx]
				passed = cm_passes[trial_idx, m_idx]
				status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
				line += f" {metric.name}: {val:.4f} ({status}) |"
			
			# Test metrics
			for m_idx, metric in enumerate(self.test_metrics):
				val = tm_measurements[trial_idx, m_idx]
				passed = tm_passes[trial_idx, m_idx]
				status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
				line += f" {metric.name}: {val:.4f} ({status}) |"
			
			print(line)
	
	def _print_summary(self, title, cm_passes, tm_passes):
		if self.verbosity == 2:
			title_line = "Total Passes "
		else:
			title_line = f"{fg.cyan}{self.test_type}{fg.reset} Testing Results {title}"
		
		line = title_line
		line += " " * max(1, 80 - len(title_line)) + "|"
		
		for m_idx, metric in enumerate(self.classical_metrics):
			passes = int(cm_passes[:, m_idx].sum())
			line += f" {metric.name}: {passes}/{self.tries} |"
		
		for m_idx, metric in enumerate(self.test_metrics):
			passes = int(tm_passes[:, m_idx].sum())
			line += f" {metric.name}: {passes}/{self.tries} |"
		
		print(line)
		if self.verbosity == 2:
			print()
	
	def run(self, models, titles, features, target):
		total_results = []
		
		for idx, model in enumerate(models):
			result = {
				'title': titles[idx],
				'classical_passes': np.zeros(len(self.classical_metrics)),
				'test_passes': np.zeros(len(self.test_metrics)),
				'tests': 0
			}
			
			for problem_type, problem in self.problem_columns.items():
				if problem_type == 'full':
					continue
				
				set_title = fg.gray + problem_type + fg.reset
				test_result = self.metamorphic_test(
					model=model,
					X=features,
					y=target,
					columns=problem['names'],
					title=f"{titles[idx]} {set_title}"
				)
				
				result['classical_passes'] += test_result['classical_passes']
				result['test_passes'] += test_result['test_passes']
				result['tests'] += test_result['tests']
			
			total_results.append(result)
		
		self._print_aggregate(total_results)
		
		return total_results
	
	def _print_aggregate(self, total_results):
		if len(total_results) < 2:
			return
		
		# Header
		header = "= Test Target | # of Tests"
		for result in total_results:
			header += f" | {result['title']}"
		header += " ="
		
		dashes = max(80, len(header) - 20)
		print("=" * dashes)
		print(header)
		print("=" * dashes)
		
		for m_idx, metric in enumerate(self.classical_metrics):
			line = f"{metric.name:13s} | {total_results[0]['tests']:<10d}"
			for result in total_results:
				passes = int(result['classical_passes'][m_idx])
				line += f" | {passes:<{len(result['title'])}d}"
			print(line)
		
		for m_idx, metric in enumerate(self.test_metrics):
			line = f"{metric.name:13s} | {total_results[0]['tests']:<10d}"
			for result in total_results:
				passes = int(result['test_passes'][m_idx])
				line += f" | {passes:<{len(result['title'])}d}"
			print(line)
		
		print()


def shuffle_columns(data, column_set):
	data = data.copy()
	shuffled = data[column_set].sample(frac=1).reset_index(drop=True)
	data[column_set] = shuffled
	return data


def flip_columns(data, column_set):
	data = data.copy()
	for col in column_set:
		uniq = data[col].unique()
		subset_mean = uniq.mean()
		subset = 2 * subset_mean - (data[col])
		data[col] = subset
	return data


def add_noise_to_columns(data, column_set, noise_mean=0.0, noise_scale=2.0):
	data = data.copy()
	for col in column_set:
		noise = np.random.normal(
			loc=noise_mean,
			scale=data[col].std() * noise_scale,
			size=data[col].shape[0]
		)
		data[col] = data[col] + noise
	return data


def scale_columns(data, column_set, scale_factor=1.5):
	data = data.copy()
	for col in column_set:
		data[col] = data[col] * scale_factor
	return data


def shift_columns(data, column_set, shift_amount=None):
	data = data.copy()
	for col in column_set:
		if shift_amount is None:
			shift = data[col].std() * 0.5
		else:
			shift = shift_amount
		data[col] = data[col] + shift
	return data


def quantize_columns(data, column_set, n_bins=10):
	data = data.copy()
	for col in column_set:
		data[col] = pd.qcut(data[col], q=n_bins, labels=False, duplicates='drop')
	return data


def permute_within_quantiles(data, column_set, n_quantiles=4):
	data = data.copy()
	for col in column_set:
		quantile_labels = pd.qcut(data[col], q=n_quantiles, labels=False, duplicates='drop')
		
		new_values = data[col].copy()
		for q in range(n_quantiles):
			mask = quantile_labels == q
			values_in_quantile = data.loc[mask, col].values
			np.random.shuffle(values_in_quantile)
			new_values[mask] = values_in_quantile
		
		data[col] = new_values
	
	return data
