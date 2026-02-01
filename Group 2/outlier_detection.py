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

def identify_outliers( x, columns, thres=2, verbosity=0 ):
	spaces = 120
	print(f"{style.bold}Dataset Outlier Test{style.reset} - {thres} Sigma")
	print("-"*(spaces+40))
	
	n_samples, n_features = x.shape
	features_with_many_outliers = 0
	for column in columns:
		x_min, x_max = x[column].min(), x[column].max()
		mean, std = x[column].mean(), x[column].std()
		
		#print( f"Min: {x_min}\tMax: {x_max}" )
		#print( f"Mean: {mean}\tSTD: {std}" )

		lows = x[column] <= mean - thres*std
		highs = x[column] >= mean + thres*std
		lows, highs = x[lows].shape[0], x[highs].shape[0]

		if verbosity > 1 and ( lows > 0 or highs > 0 ):
			to_print = f"{column.capitalize()}:" + " "*(spaces-len(column)) + "Low:\t"
			to_print += get_color( lows, n_samples ) + f"{lows}\t" + style.reset + "High:\t"
			to_print += get_color( highs, n_samples ) + f"{highs}\t" + style.reset
			
			print( to_print )

		if lows >= 0.05 * n_samples or highs >= 0.05 * n_samples:
			features_with_many_outliers += 1

	if verbosity > 0:
		final_color = get_color( features_with_many_outliers, n_features )
		print( f"{style.bold}Final Result{style.reset} {final_color}{features_with_many_outliers}{fg.reset}/"
			f"{fg.green}{n_features}{fg.reset} features are found to have "
			f"{fg.red}significant amounts of{fg.purple}{style.bold} outliers{style.reset}." )
