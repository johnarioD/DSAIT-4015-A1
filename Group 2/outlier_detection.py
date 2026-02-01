from term_styling import style, fg, bg
import numpy as np
import pandas as pd

def get_color( val, n_samples ):
	if val >= 0.15 * n_samples:
		return fg.red
	elif val >= 0.05 * n_samples:
		return fg.orange
	elif val >= 0.01 * n_samples:
		return fg.yellow
	return fg.green

def identify_outliers( x, columns, thres=2 ):
	spaces = 120
	print(f"{style.bold}Dataset Outlier Test{style.reset} - {thres} Sigma")
	print("-"*(spaces+40))
	
	n_samples = x.shape[0]
	for column in columns:
		x_min, x_max = x[column].min(), x[column].max()
		mean, std = x[column].mean(), x[column].std()
		
		#print( f"Min: {x_min}\tMax: {x_max}" )
		#print( f"Mean: {mean}\tSTD: {std}" )

		lows = x[column] <= mean - thres*std
		highs = x[column] >= mean + thres*std
		lows, highs = x[lows].shape[0], x[highs].shape[0]

		if lows > 0 or highs > 0:
			to_print = f"{column.capitalize()}:" + " "*(spaces-len(column)) + "Low:\t"
			to_print += get_color( lows, n_samples ) + f"{lows}\t" + style.reset + "High:\t"
			to_print += get_color( highs, n_samples ) + f"{highs}\t" + style.reset
				
			print( to_print )
