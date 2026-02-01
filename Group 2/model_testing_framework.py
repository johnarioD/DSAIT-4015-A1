import numpy as np
import pandas as pd

from term_styling import style, fg, bg

class TestSuite:
	def __init__( self, problem_cols, partitions, classical_thresh=0.9, metric_thresh=0.05 ):
		self.partitions = partitions
		self.problem_cols = problem_cols
def partition_test( model, X, y, )
