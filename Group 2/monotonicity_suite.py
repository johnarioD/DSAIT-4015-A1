import numpy as np
import pandas as pd
import json
from utils import *
from term_styling import style, fg, bg
from metrics import Metric, consistency_score

class MonotonicitySuite:
	def __init__(self, monotonicity_specs,
				 violation_threshold = 0.10, n_samples = 100,
				 verbosity = 0):
		self.monotonicity_specs = monotonicity_specs
		self.violation_threshold = violation_threshold
		self.n_samples = n_samples
		self.verbosity = verbosity
	
	def monotonicity_test(self, model, X, y, title):
		if self.verbosity == 2:
			print("=" * 80)
			print(f"=== Monotonicity Testing {title} | Violation Threshold: {style.bold}{self.violation_threshold}{style.reset} ===")
			print("=" * 80)
		
		results_per_feature = {}
		
		for feature, expected_direction in self.monotonicity_specs.items():
			sample_size = min(self.n_samples, len(X))
			X_sample = X.sample(n=sample_size, random_state=42)
			
			violations = 0
			total_comparisons = 0
			
			for idx in X_sample.index:
				x_base = X.loc[[idx]].copy()
				y_pred_base = model.predict(x_base)[0]
				
				x_increased = x_base.copy()
				current_val = x_increased[feature].values[0]
				x_increased[feature] = current_val + abs(current_val) * 0.5 + 0.5
				y_pred_increased = model.predict(x_increased)[0]
				
				total_comparisons += 1
				
				if expected_direction == 'increasing':
					if y_pred_increased < y_pred_base:
						violations += 1
				elif expected_direction == 'decreasing':
					if y_pred_increased > y_pred_base:
						violations += 1
				elif expected_direction == 'none':
					if y_pred_increased != y_pred_base:
						violations += 1
			
			violation_rate = violations / total_comparisons if total_comparisons > 0 else 0
			passed = violation_rate <= self.violation_threshold
			
			results_per_feature[feature] = {
				'expected_direction': expected_direction,
				'violations': violations,
				'total': total_comparisons,
				'violation_rate': violation_rate,
				'passed': passed
			}
			
			if self.verbosity == 2:
				status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
				print(f"{feature:40s} ({expected_direction:10s}): {violations}/{total_comparisons} violations ({violation_rate:.4f}) ({status})")
		
		if self.verbosity >= 1 and self.verbosity != 2:
			total_passed = sum(1 for r in results_per_feature.values() if r['passed'])
			total_features = len(results_per_feature)
			print(f"{fg.cyan}Monotonicity{fg.reset} Testing {title}: {total_passed}/{total_features} features passed")
			print()
		elif self.verbosity == 2:
			print()
		
		return results_per_feature
	
	def run(self, models, titles, features, target):
		all_results = []
		
		for idx, model in enumerate(models):
			results = self.monotonicity_test(model, features, target, titles[idx])
			all_results.append({'title': titles[idx], 'results': results})
		
		if len(all_results) >= 2 and self.verbosity >= 1:
			self._print_comparison(all_results)
		
		return all_results
	
	def save_json(self, results, filename):
		json_results = []
		for result_set in results:
			features_data = {}
			for feature, data in result_set['results'].items():
				features_data[feature] = {
					'expected_direction': data['expected_direction'],
					'violations': int(data['violations']),
					'total': int(data['total']),
					'violation_rate': float(data['violation_rate']),
					'passed': bool(data['passed'])
				}
			json_results.append({
				'title': result_set['title'],
				'features': features_data
			})
		with open(filename, 'w') as f:
			json.dump(json_results, f, indent=2)
	
	def _print_comparison(self, all_results):
		header = "= Monotonicity | " + " | ".join([r['title'] for r in all_results]) + " ="
		dashes = len(header) - 20
		print("=" * dashes)
		print(header)
		print("=" * dashes)
		
		all_features = set()
		for result_set in all_results:
			all_features.update(result_set['results'].keys())
		
		for feature in sorted(all_features):
			line = f"{feature[:20]:20s}"
			for result_set in all_results:
				if feature in result_set['results']:
					r = result_set['results'][feature]
					status = f"{fg.green}PASS{fg.reset}" if r['passed'] else f"{fg.red}FAIL{fg.reset}"
					line += f" | {r['violation_rate']:.4f} {status}"
				else:
					line += " | N/A"
			print(line)
		
		print()
