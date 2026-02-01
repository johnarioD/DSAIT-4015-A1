import numpy as np
import pandas as pd
import json
from utils import *
from term_styling import style, fg, bg
from metrics import Metric, consistency_score

class BoundarySuite:
	def __init__(self, features_to_test, classical_metrics,
				 percentile_low = 0.05, percentile_high = 0.95,
				 verbosity = 0):
		self.features_to_test = features_to_test
		self.classical_metrics = classical_metrics if isinstance(classical_metrics, list) else [classical_metrics]
		self.percentile_low = percentile_low
		self.percentile_high = percentile_high
		self.verbosity = verbosity
	
	def boundary_test(self, model, X, y, title):
		if self.verbosity == 2:
			self._print_header(title)
		
		results_per_feature = {}
		
		for feature in self.features_to_test:
			low_threshold = X[feature].quantile(self.percentile_low)
			high_threshold = X[feature].quantile(self.percentile_high)
			
			mask_low = X[feature] <= low_threshold
			mask_high = X[feature] >= high_threshold
			
			X_low = X[mask_low]
			y_low = y[mask_low]
			X_high = X[mask_high]
			y_high = y[mask_high]
			
			if len(X_low) < 10 or len(X_high) < 10:
				continue
			
			y_pred_low = model.predict(X_low)
			y_pred_high = model.predict(X_high)
			
			feature_results = {
				'low_measurements': [],
				'high_measurements': [],
				'low_passes': [],
				'high_passes': []
			}
			
			for metric in self.classical_metrics:
				val_low = metric.fn(y_pred_low, y_low)
				val_high = metric.fn(y_pred_high, y_high)
				
				feature_results['low_measurements'].append(val_low)
				feature_results['high_measurements'].append(val_high)
				feature_results['low_passes'].append(metric.test(val_low))
				feature_results['high_passes'].append(metric.test(val_high))
			
			results_per_feature[feature] = feature_results
			
			if self.verbosity == 2:
				self._print_feature_results(feature, feature_results, low_threshold, high_threshold)
		
		if self.verbosity >= 1:
			self._print_summary(title, results_per_feature)
		
		return results_per_feature
	
	def _print_header(self, title):
		title_string = f"=== Boundary Testing {title} |"
		for metric in self.classical_metrics:
			title_string += f" {metric.name} {style.bold}{metric.threshold}{style.reset} |"
		title_string += "="
		
		dashes = len(title_string) - 38
		print("=" * dashes)
		print(title_string)
		print("=" * dashes)
	
	def _print_feature_results(self, feature, results, low_val, high_val):
		print(f"\n{style.bold}{feature}{style.reset} (Low <= {low_val:.3f}, High >= {high_val:.3f})")
		
		for m_idx, metric in enumerate(self.classical_metrics):
			val_low = results['low_measurements'][m_idx]
			val_high = results['high_measurements'][m_idx]
			pass_low = results['low_passes'][m_idx]
			pass_high = results['high_passes'][m_idx]
			
			status_low = f"{fg.green}PASS{fg.reset}" if pass_low else f"{fg.red}FAIL{fg.reset}"
			status_high = f"{fg.green}PASS{fg.reset}" if pass_high else f"{fg.red}FAIL{fg.reset}"
			
			print(f"  {metric.name}: Low={val_low:.4f} ({status_low}), High={val_high:.4f} ({status_high})")
	
	def _print_summary(self, title, results_per_feature):
		if self.verbosity == 2:
			return
		
		total_tests = len(results_per_feature) * 2
		total_passes = sum(
			sum(r['low_passes']) + sum(r['high_passes'])
			for r in results_per_feature.values()
		)
		
		print(f"{fg.cyan}Boundary{fg.reset} Testing {title}: {total_passes}/{total_tests * len(self.classical_metrics)} passed")
		print()
	
	def run(self, models, titles, features, target):
		all_results = []
		
		for idx, model in enumerate(models):
			results = self.boundary_test(model, features, target, titles[idx])
			all_results.append({'title': titles[idx], 'results': results})
		
		return all_results
	
	def save_json(self, results, filename):
		json_results = []
		for result_set in results:
			features_data = {}
			for feature, data in result_set['results'].items():
				features_data[feature] = {
					'low_measurements': [float(x) for x in data['low_measurements']],
					'high_measurements': [float(x) for x in data['high_measurements']],
					'low_passes': [bool(x) for x in data['low_passes']],
					'high_passes': [bool(x) for x in data['high_passes']]
				}
			json_results.append({
				'title': result_set['title'],
				'features': features_data
			})
		with open(filename, 'w') as f:
			json.dump(json_results, f, indent=2)


