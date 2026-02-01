import numpy as np
import pandas as pd
import json
from utils import *
from term_styling import style, fg, bg
from metrics import Metric, consistency_score

class ConsistencySuite:
	def __init__(self, n_trials = 10, sample_size = 100,
				 consistency_threshold = 1.0, random_state=42, verbosity = 0):
		self.n_trials = n_trials
		self.sample_size = sample_size
		self.consistency_threshold = consistency_threshold
		self.verbosity = verbosity
		self.random_state = random_state
	
	def consistency_test(self, model, X, y, title):
		if self.verbosity == 2:
			print("=" * 80)
			print(f"=== Consistency Testing {title} | Threshold: {style.bold}{self.consistency_threshold}{style.reset} ===")
			print("=" * 80)
		
		sample_size = min(self.sample_size, len(X))
		X_sample = X.sample(n=sample_size, random_state=self.random_state)
		y_sample = y.loc[X_sample.index]
		
		predictions = []
		for trial in range(self.n_trials):
			y_pred = model.predict(X_sample)
			predictions.append(y_pred)
		
		predictions = np.array(predictions)
		
		per_sample_consistency = np.array([
			np.all(predictions[:, i] == predictions[0, i])
			for i in range(sample_size)
		])
		
		consistency_rate = per_sample_consistency.mean()
		passed = consistency_rate >= self.consistency_threshold
		
		pred_variance = np.var(predictions, axis=0).mean()
		
		if self.verbosity >= 1:
			status = f"{fg.green}PASS{fg.reset}" if passed else f"{fg.red}FAIL{fg.reset}"
			print(f"Consistency Test {title}:")
			print(f"  Consistency Rate: {consistency_rate:.4f} ({status})")
			print(f"  Prediction Variance: {pred_variance:.6f}")
			print(f"  Unanimous Predictions: {per_sample_consistency.sum()}/{sample_size}")
			if self.verbosity == 2:
				print()
		
		return {
			'consistency_rate': consistency_rate,
			'passed': passed,
			'pred_variance': pred_variance,
			'unanimous_count': per_sample_consistency.sum()
		}
	
	def run(self, models, titles, features, target):
		results = []
		
		for idx, model in enumerate(models):
			result = self.consistency_test(model, features, target, titles[idx])
			result['title'] = titles[idx]
			results.append(result)
		
		if len(results) >= 2 and self.verbosity >= 1:
			self._print_comparison(results)
		
		return results
	
	def save_json(self, results, filename):
		json_results = []
		for result in results:
			json_results.append({
				'title': result['title'],
				'consistency_rate': float(result['consistency_rate']),
				'passed': bool(result['passed']),
				'pred_variance': float(result['pred_variance']),
				'unanimous_count': int(result['unanimous_count'])
			})
		with open(filename, 'w') as f:
			json.dump(json_results, f, indent=2)
	
	def _print_comparison(self, results):
		header = "= Consistency | " + " | ".join([r['title'] for r in results]) + " ="
		dashes = len(header) - 20
		print("=" * dashes)
		print(header)
		print("=" * dashes)
		
		line = "Rate		 "
		for r in results:
			rate_str = f"{r['consistency_rate']:.4f}"
			status = f"({fg.green}PASS{fg.reset})" if r['passed'] else f"({fg.red}FAIL{fg.reset})"
			line += f" | {rate_str} {status}"
		print(line)
		
		line = "Variance	 "
		for r in results:
			line += f" | {r['pred_variance']:.6f}"
		print(line)
		
		print()


