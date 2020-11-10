from __future__ import print_function
from __future__ import division
from . import C_

from tqdm import tqdm
import sys

###################################################################################################################################################

class LevelBar():
	def __init__(self, count_dict,
		prefix_str='',
		ncols=10,
		):
		self.f = open('/dev/null', 'w')
		self.bar_kwargs = {
			'ncols':ncols,
			'bar_format':'|{bar}|',
			#'bar_format':'{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
			#'bar_format':'{l_bar}{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}]',
			'postfix':'',
			'file':self.f,
			'leave':True,
		}
		self.count_dict = count_dict.copy()
		self.total_count = sum([self.count_dict[key] for key in count_dict.keys()])
		self.prefix_str = prefix_str

	def __repr__(self):
		total = 100
		txt = ''
		for k,key in enumerate(self.count_dict.keys()):
			bar = tqdm(total=total, **self.bar_kwargs)
			count = self.count_dict[key]
			percent = count/self.total_count*100
			bar.update(percent)
			txt += f'{self.prefix_str}{bar.__repr__()} {key} - {count:,}/{self.total_count:,} ({percent:.2f}%)\n'

		return txt