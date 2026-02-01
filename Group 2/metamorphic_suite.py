import numpy as np
import pandas as pd
from utils import *
from term_styling import style, fg, bg

class MetamorphicSuite():
	def __init__(self, metamorphic_fn, test_type, tries, problem_columns, classical_metric, test_metric, verbosity=0 ):
		self.metamorphic_fn = metamorphic_fn
		self.tries = tries
		self.classical_metric = classical_metric
		self.test_metric = test_metric
		self.verbosity = verbosity
		self.test_type = test_type
		self.problem_columns = problem_columns
	
	def metamorphic_test(self, model, X, y, columns, title ):
		if self.verbosity == 2:
			print_title_card( suite=self.test_type+" Testing", title=title, classical_metric=self.classical_metric, test_metric=self.test_metric )

		tm_passes, cm_passes = 0, 0
		checked_per_try = np.empty( self.tries )
		y_pred_orig = model.predict( X )

		for idx in range(self.tries):
			X_alt = self.metamorphic_fn( X, columns )
			y_pred = model.predict( X_alt )

			cm_val = self.classical_metric.fn(y_pred, y)
			tm_val = self.test_metric.fn(y_pred,y_pred_orig)
			
			cm_pass, tm_pass = 0, 0
			if cm_val >= self.classical_metric.threshold:
				cm_pass = 1
			if tm_val <= self.test_metric.threshold:
				tm_pass = 1

			cm_passes += cm_pass
			tm_passes += tm_pass
			if self.verbosity ==2:
				print_iter( iter_name=f"Iteration {idx}", vals=[cm_val,tm_val], passes=[cm_pass,tm_pass], metrics=[self.classical_metric,self.test_metric])

		if self.verbosity >= 1:
			title = f"{fg.cyan}{self.test_type}{fg.reset} Testing Results {title}"
			if self.verbosity == 2:
				title = "Total Passes "
			print_totals( title=title,
						passes=[int(cm_passes),int(tm_passes)],
						metrics=[self.classical_metric,self.test_metric],
						trials=self.tries, add_newline=self.verbosity==2 )
		
		return {
			'test_passes': tm_passes,
			'classical_passes': cm_passes,
			'tests': self.tries
		}
	
	def run( self, models, titles, features, target ):
		total_results = []
		for idx, model in enumerate(models):
			total_results.append({ 'title': titles[idx], 'test_passes': 0, 'classical_passes': 0, 'tests': 0 })
			for problem_type, problem in self.problem_columns.items():
				if problem_type == 'full':
					continue
				set_title = fg.gray + problem_type + fg.reset
				results = self.metamorphic_test( model=model, X=features, y=target, columns=problem['names'], title=f"{titles[idx]} {set_title}")
				total_results[idx]['test_passes'] += results['test_passes']
				total_results[idx]['classical_passes'] += results['classical_passes']
				total_results[idx]['tests'] += results['tests']
		
		aggregate_results( total_results, [self.classical_metric,self.test_metric] )

def shuffle_columns( data, column_set ):
	data = data.copy()
	shuffled = data[column_set].sample(frac=1).reset_index(drop=True)
	data[column_set] = shuffled
	return data

def flip_columns( data, column_set ):
	data = data.copy()
	for col in column_set:
		uniq = data[col].unique()
		subset_mean = uniq.mean()
		subset = 2*subset_mean - ( data[col] )
		data[col] = subset
	return data

def add_noise_to_columns( data, column_set, noise_mean=0.0, noise_scale=2.0 ):
	data = data.copy()
	for col in column_set:
		noise = np.random.normal( loc=noise_mean, scale=data[col].std()*noise_scale, size=data[col].shape[0] )
		data[col] = data[col] + noise
	return data
