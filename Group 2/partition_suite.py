import numpy as np
import pandas as pd
from utils import *
from term_styling import style, fg, bg
from metrics import Metric

class PartitionSuite:
	
	def __init__(self, problem_cols, classical_metrics, 
				 test_metrics, verbosity=0):
		self.problem_cols = problem_cols
		self.classical_metrics = classical_metrics if isinstance(classical_metrics, list) else [classical_metrics]
		self.test_metrics = test_metrics if isinstance(test_metrics, list) else [test_metrics]
		self.verbosity = verbosity
	
	def partition_test(self, model, X, y, partitions, title):
		if self.verbosity == 2:
			self._print_header(title)
		
		n_partitions = len(partitions)
		n_classical = len(self.classical_metrics)
		n_test = len(self.test_metrics)
		
		cm_measurements = np.zeros((n_partitions, n_classical))
		tm_measurements = np.zeros((n_partitions, n_test))
		cm_passes = np.zeros((n_partitions, n_classical))
		tm_passes = np.zeros((n_partitions, n_test))
		
		pred_rates = np.zeros(n_partitions)
		true_rates = np.zeros(n_partitions)
		
		y_preds_per_partition = []
		
		for idx, partition in enumerate(partitions):
			X_part = X.iloc[partition[0]]
			y_part = y.iloc[partition[0]]
			y_pred = model.predict(X_part)
			
			y_preds_per_partition.append(y_pred)
			
			for m_idx, metric in enumerate(self.classical_metrics):
				measurement = metric.fn(y_pred, y_part)
				cm_measurements[idx, m_idx] = measurement
				cm_passes[idx, m_idx] = 1 if metric.test(measurement) else 0

			
			pred_rates[idx] = (y_pred == 1).mean()
			true_rates[idx] = (y_part == 1).mean()
		
		mean_pred_rate = pred_rates.mean()
		
		for idx in range(n_partitions):
			for m_idx, metric in enumerate(self.test_metrics):
				measurement = 0
				if 'parity' in metric.name.lower():
					measurement = abs(pred_rates[idx] - mean_pred_rate)
				elif 'calibration' in metric.name.lower():
					measurement = metric.fn(
						y_preds_per_partition[idx], 
						y.iloc[partitions[idx][0]]
					)
				else:
					measurement = metric.fn(pred_rates[idx], mean_pred_rate)
				tm_measurements[idx, m_idx] = measurement
				tm_passes[idx, m_idx] = 1 if metric.test(measurement) else 0
		
		if self.verbosity == 2:
			self._print_detailed_results(
				partitions, cm_measurements, tm_measurements, 
				cm_passes, tm_passes, pred_rates, true_rates
			)
		
		if self.verbosity >= 1:
			self._print_summary(title, cm_passes, tm_passes, n_partitions)
		
		return {
			'classical_passes': cm_passes.sum(axis=0).astype(int),
			'test_passes': tm_passes.sum(axis=0).astype(int),
			'tests': n_partitions,
			'classical_measurements': cm_measurements,
			'test_measurements': tm_measurements
		}
	
	def _print_header(self, title):
		title_string = f"=== Partition Testing {title} |"
		for metric in self.classical_metrics:
			title_string += f" {metric.name} {style.bold}{metric.threshold}{style.reset} |"
		for metric in self.test_metrics:
			title_string += f" {metric.name} {style.bold}{metric.threshold}{style.reset} |"
		title_string += "="
		
		dashes = max(80, len(title_string) - 38)
		print("=" * dashes)
		print(title_string)
		print("=" * dashes)
	
	def _print_detailed_results(self, partitions, cm_measurements, tm_measurements,
							   cm_passes, tm_passes, pred_rates, true_rates):
		for idx in range(len(partitions)):
			line = f"Partition {idx}\t|"
			
			# Classical metrics
			for m_idx, metric in enumerate(self.classical_metrics):
				val = cm_measurements[idx, m_idx]
				passed = cm_passes[idx, m_idx]
				status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
				line += f" {metric.name}: {val:.4f} ({status}) |"
			
			# Test metrics
			for m_idx, metric in enumerate(self.test_metrics):
				val = tm_measurements[idx, m_idx]
				passed = tm_passes[idx, m_idx]
				status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
				line += f" {metric.name}: {val:.4f} ({status}) |"
			
			# Prediction vs true rates
			line += f" Pred: {pred_rates[idx]:.3f} True: {true_rates[idx]:.3f}"
			
			print(line)
	
	def _print_summary(self, title, cm_passes, tm_passes, n_partitions):
		if self.verbosity == 2:
			title_line = "Total Passes "
		else:
			title_line = f"{fg.cyan}Partition{fg.reset} Testing Results {title}"
		
		line = title_line
		line += " " * max(1, 80 - len(title_line)) + "|"
		
		for m_idx, metric in enumerate(self.classical_metrics):
			passes = int(cm_passes[:, m_idx].sum())
			line += f" {metric.name}: {passes}/{n_partitions} |"
		
		for m_idx, metric in enumerate(self.test_metrics):
			passes = int(tm_passes[:, m_idx].sum())
			line += f" {metric.name}: {passes}/{n_partitions} |"
		
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
			
			for problem_name, problem in self.problem_cols.items():
				if problem_name == 'full':
					continue
				
				set_title = fg.gray + problem_name + fg.reset
				test_result = self.partition_test(
					model=model,
					X=features,
					y=target,
					partitions=problem['partitions'],
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
