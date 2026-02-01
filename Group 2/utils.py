from term_styling import style, fg, bg

def print_title_card( suite, title, classical_metric, test_metric):
	title_string = f"=== {suite} {title} |"
	title_string += f" {classical_metric.name} {style.bold}{classical_metric.threshold}{style.reset} |"
	title_string += f" {test_metric.name} {style.bold}{test_metric.threshold}{style.reset} ==="
	dashes = len(title_string) - 38
	print( "="*dashes )
	print( title_string )
	print( "="*dashes )

def print_iter( iter_name, vals, passes, metrics ):
	to_print = f"{iter_name}\t|"
	for idx, is_pass in enumerate(passes):
		pass_str = fg.green+'PASS'+fg.reset if is_pass else fg.red+'FAIL'+fg.reset
		to_print += f" {metrics[idx].name}: {vals[idx]:.4f} ({pass_str}) |"
	to_print = to_print[:-2]
	print(to_print)

def print_totals( title, passes, metrics, trials, add_newline=False ):
	empty_space = 20 if add_newline else 80
	to_print = f"{title}"
	to_print += ' '*(empty_space-len(to_print)) + '|'
	for idx, val in enumerate(passes):
		to_print += f" {metrics[idx].name}: {val}/{trials} |"
	to_print = to_print[:-2]
	print(to_print)
	if add_newline:
		print()

def aggregate_results( results, metrics ):
	resultsA, resultsB = results
	to_print = f"= Test Target | # of Tests | {resultsA['title']} | {resultsB['title']} ="
	dashes = len(to_print) - 20
	print( "="*dashes )
	print( to_print )
	print( "="*dashes )
	empty_space = 20
	extra_spaces = len(metrics[0].name) - len(metrics[1].name)
	if extra_spaces >= 0:
		print( f"{metrics[0].name}  | {resultsA['tests']}\t   | {resultsA['classical_passes']}\t\t| {resultsB['classical_passes']}")
		print( f"{metrics[1].name}  " + " "*extra_spaces +f"| {resultsA['tests']}\t   | {resultsA['test_passes']}\t\t| {resultsB['test_passes']}\n")
	else:
		extra_spaces *= -1
		print( f"{metrics[0].name}  " + " "*extra_spaces +f"| {resultsA['tests']}\t   | {resultsA['classical_passes']}\t\t| {resultsB['classical_passes']}")
		print( f"{metrics[1].name}  | {resultsA['tests']}\t   | {resultsA['test_passes']}\t\t| {resultsB['test_passes']}\n")
