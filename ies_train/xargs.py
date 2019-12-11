'''
define args
@file: xargs.py
@author: qxliu
@time: 2019/3/1 20:46
'''
import argparse
from data.constants import *
import tensorflow as tf

DICT_ACTIVATION = {
	'relu': tf.nn.relu
	, 'leaky_relu': tf.nn.leaky_relu
	, 'sigmoid': tf.nn.sigmoid
}


def boolean_string(s):
	if s not in {'False', 'True'}:
		raise ValueError('Not a valid boolean string')
	return s == 'True'


def add_data_options(parser):
	# #===== necessary args (positional args)
	parser.add_argument('dsName', type=lambda dsName: DS_Names[dsName], choices=list(DS_Names))
	parser.add_argument('dataSplit', type=str, choices=['e5fold'])  # ===
	parser.add_argument('rlName', type=str, choices=['ds'])

	# ===== optional args
	# ========1. to tune
	# parser.add_argument('-initSumm', default=Init_Summ.facese_rm, type=lambda initSumm: Init_Summ[initSumm], choices=list(Init_Summ), help='init summ for train')
	parser.add_argument('-initSumm', default=default_InitSumm, type=lambda initSumm: Init_Summ[initSumm],
						choices=list(Init_Summ))
	parser.add_argument('-initSummForTest', default=default_InitSumm, type=lambda initSumm: Init_Summ[initSumm],
						choices=list(Init_Summ))
	#======= for simple state encoder
	parser.add_argument('-encodeStyle', default='', type=str) # correspond to Env().init()
	parser.add_argument('-ftypes', default='embed', type=str)
	parser.add_argument('-foldId', default=-1, type=int)

	# ======= for deepsets encoder
	parser.add_argument('-deepsetStyle', default='triple_curr_disH_cand', type=str)
	parser.add_argument('-deepsetHidden', default=300, type=int)
	parser.add_argument('-deepsetOut', default=150, type=int)

	# ========2. fixed
	parser.add_argument('-topK', default=TOPK.top5, type=lambda topK: TOPK[topK], choices=list(TOPK))
	parser.add_argument('-cleaned', default=True, type=boolean_string)
	parser.add_argument('-byUser', default=True, type=boolean_string)
	parser.add_argument('-edistinct', default=True, type=boolean_string)
	parser.add_argument('-reward_strategy', default='rdcg', type=str, choices=['rdcg','rdislike','rrouge'])


def add_train_options(parser):
	# ===== for log
	parser.add_argument('run_name', default='try', type=str, help='name for file prefix (without dsName)')  # ====
	parser.add_argument('-gamma', default=0.6, type=float)
	parser.add_argument('-gamma_out', default=0.6, type=float)
	parser.add_argument('-useTotalReward', default=False, type=boolean_string, help='should set a non-zero gamma')

	# ===== optional args
	# ========1. to tune
	parser.add_argument('-batch_size', default=1, type=int)

	# ========2. almost fixed
	parser.add_argument('-batch_num', default=np.inf, type=int,
						help='default: compute accroding to batch_size; otherwise, final_batch_num=min(comptute, setted)')
	parser.add_argument('-epoch_num', default=100, type=int, help='')

	parser.add_argument('-num_h1_units', default=64, type=int)
	parser.add_argument('-type_activation', default=DICT_ACTIVATION['leaky_relu'], type=str,
						choices=DICT_ACTIVATION.keys())
	parser.add_argument('-type_activation_out', default=DICT_ACTIVATION['leaky_relu'], type=str,
						choices=DICT_ACTIVATION.keys())
	parser.add_argument('-learning_rate', default=0.01, type=float)
	parser.add_argument('-dropout_rate', default=0.5, type=float, help='set dropout_rate_for_train')

	parser.add_argument('-num_repeats', default=1, type=int)
	parser.add_argument('-num_to_save', default=2500, type=int,
						help='max num of model to save for each train, default=3, to set to 20')
	parser.add_argument('-dev_step', default=1, type=int,
						help='the steps to compute dev and test, if=1, will dev for each epoch, be slow')

	# ======= for control
	parser.add_argument('-do_debug', default=False, type=boolean_string,
						help="default False, if =True, will limit to only run 3 epoch for each fold and only 2 bach for each epoch")
	parser.add_argument('-do_rev_folds', default=False, type=boolean_string,
						help="default False, if =True, will run each folds in reversed order, i.e. run the last fold first, using: for fold_id, fold_data in reversed(list(enumerate(dataList)))")

	parser.add_argument('-use_neg', default=True, type=boolean_string,
						help="default True, if =True, whether use negative loss")
	parser.add_argument('-is_try', default=False, type=boolean_string,
						help="whether to do extra print")


def add_test_options(parser):
	parser.add_argument('-restore_path', default=None, type=str,
						help='full path for the trained model to be used for test')
