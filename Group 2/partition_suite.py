import numpy as np
import pandas as pd
from utils import *
from term_styling import style, fg, bg

class PartitionSuite():
	def __init__( self, problem_cols, classical_metric, test_metric, verbosity=0 ):
		self.problem_cols = problem_cols
		self.classical_metric = classical_metric
		self.test_metric = test_metric
		self.verbosity = verbosity
	
	def partition_test(self, model, X, y, partitions, title):
		if self.verbosity == 2:
			print_title_card( suite="Partition Testing", title=title, classical_metric=self.classical_metric, test_metric=self.test_metric )

		cm_measurements, tm_measurements = np.zeros( len(partitions) ), np.zeros( len(partitions) )
		checked_per_partition = np.empty( len(partitions) )

		for idx, partition in enumerate( partitions ):
			X_part = X.iloc[partition[0]]
			y_part = y.iloc[partition[0]]

			y_pred = model.predict(X_part)

			cm_measurements[idx] = self.classical_metric.fn(y_pred, y_part)
			checked_per_partition[idx] = ( y_pred == 1 ).mean()

		tm_measurements = self.test_metric.fn( checked_per_partition )
	
		cm_passes, tm_passes = np.zeros( len(partitions) ), np.zeros( len(partitions) )
		for idx in range(len(partitions)):
			cm_passes[idx] = 1 if cm_measurements[idx] >= self.classical_metric.threshold else 0
			tm_passes[idx] = 1 if tm_measurements[idx] <= self.test_metric.threshold else 0
	
		if self.verbosity == 2:
			for idx, partition in enumerate( partitions ):
				measurements = [cm_measurements[idx],tm_measurements[idx]]
				passes = [cm_passes[idx],tm_passes[idx]]
				print_iter( iter_name=f"Partition {idx}", vals=measurements, passes=passes, metrics=[self.classical_metric,self.test_metric])

		if self.verbosity >= 1:
			title = f"{fg.cyan}Partition{fg.reset} Testing Results {title}"
			if self.verbosity == 2:
				title = "Total Passes "
			print_totals( title=title,
						passes=[int(cm_passes.sum()),int(tm_passes.sum())],
						metrics=[self.classical_metric,self.test_metric],
						trials=len(partitions), add_newline=self.verbosity==2 )
	
		return {
			'test_passes': int(tm_passes.sum()),
			'classical_passes': int(cm_passes.sum()),
			'tests': len(partitions)
		}

	def run( self, models, titles, features, target ):
		total_results = []
		for idx, model in enumerate(models):
			total_results.append({ 'title': titles[idx], 'test_passes': 0, 'classical_passes': 0, 'tests': 0 })
			for problem_name, problem in self.problem_cols.items():
				if problem_name == 'full':
					continue
				set_title = fg.gray + problem_name + fg.reset
				results = self.partition_test( model=model, X=features, y=target, partitions=problem['partitions'], title=f"{titles[idx]} {set_title}" )
				total_results[idx]['test_passes'] += results['test_passes']
				total_results[idx]['classical_passes'] += results['classical_passes']
				total_results[idx]['tests'] += results['tests']
		
		aggregate_results( total_results, [self.classical_metric,self.test_metric] )
