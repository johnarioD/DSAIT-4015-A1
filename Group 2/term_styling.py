class style:
	reset = '\033[0m'
	bold = '\033[01m'
	normal = '\033[22m'
	light = '\033[02m'
	italics = '\033[03m'
	underline = '\033[04m'
	strikethrough = '\033[09m'

class fg:
	black = '\033[30m'
	red = '\033[31m'
	green = '\033[32m'
	orange = '\033[33m'
	yellow = '\033[93m'
	blue = '\033[34m'
	purple = '\033[35m'
	cyan = '\033[36m'
	gray = '\033[37m'
	reset = '\033[39m'
	def advanced( num ):
		return '\033[38;5;' + str(num) + 'm'
	lime = advanced(10)
	lred = advanced(9)
	lblue = advanced(14)
	white = advanced(15)

class bg:
	black = '\033[40m'
	red = '\033[41m'
	green = '\033[42m'
	orange = '\033[43m'
	blue = '\033[44m'
	purple = '\033[45m'
	cyan = '\033[46m'
	gray = '\033[47m'
	reset = '\033[49m'
	def advanced( num ):
		return '\033[48;5;' + str(num) + 'm'
	lime = advanced(10)
	lred = advanced(9)
	lblue = advanced(14)
	white = advanced(15)

