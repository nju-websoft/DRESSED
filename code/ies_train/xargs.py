'''
define args
@file: xargs.py
@author: qxliu
@time: 2019/3/1 20:46
'''
import numpy as np
from util.constants import *


def boolean_string(s):
	if s not in {'False', 'True'}:
		raise ValueError('Not a valid boolean string')
	return s == 'True'


def add_data_options(parser):
	# #===== necessary args (positional args)
	parser.add_argument('ds_name', type=lambda ds_name: DS_Names[ds_name], choices=list(DS_Names))

	parser.add_argument('-dataSplit', type=str, default='e5fold', choices=['e5fold'])
	parser.add_argument('-rlName', type=str, default='ds', choices=['ds'])

	# ===== optional args
	parser.add_argument('-foldId', default=-1, type=int)
	parser.add_argument('-deepset_style', default='curr_disH_cand', type=str)
	# ========2. fixed
	parser.add_argument('-byUser', default=True, type=boolean_string)


def add_train_options(parser):
	# ===== for log
	parser.add_argument('-run_name', default='try', type=str, help='name for file prefix (without ds_name)')  # ====
	parser.add_argument('-gamma', default=0.6, type=float)
	parser.add_argument('-gamma_out', default=0.6, type=float)

	# ===== optional args
	# ========1. to tune
	parser.add_argument('-batch_size', default=16, type=int)

	# ========2. almost fixed
	parser.add_argument('-batch_num', default=np.inf, type=int,
						help='default: compute accroding to batch_size; otherwise, final_batch_num=min(comptute, setted)')
	parser.add_argument('-epoch_num', default=50, type=int, help='')

	parser.add_argument('-learning_rate', default=0.01, type=float)

	parser.add_argument('-num_repeats', default=1, type=int)
	parser.add_argument('-num_to_save', default=2500, type=int,
						help='max num of model to save for each train, default=3, to set to 20')
	parser.add_argument('-dev_step', default=1, type=int,
						help='the steps to compute dev and test, if=1, will dev for each epoch, be slow')

	# ======= for control
	parser.add_argument('-do_rev_folds', default=False, type=boolean_string,
						help="default False, if =True, will run each folds in reversed order, i.e. run the last fold first, using: for fold_id, fold_data in reversed(list(enumerate(dataList)))")

def add_test_options(parser):
	parser.add_argument('-restore_path', default=None, type=str,
						help='full path for the trained model to be used for test')
