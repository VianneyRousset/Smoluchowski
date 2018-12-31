#!/usr/bin/env python

from sys import stdout

class ProgressBar:

	BAR_CHAR = ' ▏▎▍▌▋▊▉█'

	def __init__(self, width=80, print_percent=True):
		self.width = width
		self.print_percent = print_percent
		self.started = False

	def set_ratio(self, r=0.0):
		from math import ceil 
		r = max(r, 0)
		r = min(r, 1)

		stdout.write('\b'*(self.width))
		self.started = True

		percent = ''
		if self.print_percent:
			percent = ' {:6.2f}% '.format(round(r*100, 2))
			width = self.width - len(percent)

		r = r*(width-2)
		nBlocks = ceil(r) - 1
		last = int(round((r - nBlocks) * (len(self.BAR_CHAR) - 1)))
		last = self.BAR_CHAR[last]
		Nspaces = width - nBlocks - 1 - len(percent)
		stdout.write(percent + '|' + self.BAR_CHAR[-1]*nBlocks + last + ' '*Nspaces + '|')
		stdout.flush()

	def end(self):
		self.set_ratio(1.0)
		print()
