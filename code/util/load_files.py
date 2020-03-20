'''
@file: load_files.py
@author: qxLiu
@time: 2020/3/14 23:58
'''

from collections import deque
import os
import numpy as np
CODE_DIR = os.path.dirname(os.path.dirname(__file__))#os.path.dirname(os.getcwd())
ROOT_DIR = os.path.dirname(CODE_DIR)

IN_ORIGIN = os.path.join(ROOT_DIR, 'data', 'in_origin')
IN_PARSED = os.path.join(ROOT_DIR, 'data', 'in_parsed')

OUT_LOG_DIR = os.path.join(ROOT_DIR,'data','out_logs/')

if not os.path.exists(OUT_LOG_DIR):
	os.makedirs(OUT_LOG_DIR)

def load_desc(ds_name_str):
	in_file = os.path.join(IN_PARSED, 'desc_cleaned_{}.txt'.format(ds_name_str))
	eid_desc_dict = dict()
	max_desc_len = 0
	with open(in_file,'r',encoding='utf-8') as fin:
		for line in fin:
			items = line.strip().split('\t')
			eid = int(items[0])
			desc = eval(items[1])
			assert type(desc)==list
			eid_desc_dict[eid]=desc
			desc_size = len(desc)
			if desc_size>max_desc_len:
				max_desc_len = desc_size
	return eid_desc_dict, desc_size

def load_gold(ds_name_str, topk_str):
	in_file = os.path.join(IN_PARSED, 'eugold_cleaned_{}_{}.txt'.format(ds_name_str, topk_str))
	eu_gold_dict = dict()
	with open(in_file, 'r', encoding='utf-8') as fin:
		for line in fin:
			items = line.strip().split('\t')
			eid = int(items[0])
			uid = int(items[1])
			gold = eval(items[2])
			assert type(gold) == list
			eu_gold_dict[(eid,uid)] = gold
	return eu_gold_dict

def load_init_summ(ds_name_str):
	in_file = os.path.join(IN_PARSED, 'init_summ_{}.txt'.format(ds_name_str))
	eid_inisumm_dict = dict()
	with open(in_file, 'r', encoding='utf-8') as fin:
		for line in fin:
			items = line.strip().split('\t')
			eid = int(items[0])
			inisum = eval(items[1])
			assert type(inisum) == list
			eid_inisumm_dict[eid] = inisum
	return eid_inisumm_dict


def load_e5fold_train_valid_test(ds_name_str):
	in_file = os.path.join(IN_PARSED, 'split_cleaned_{}.txt'.format(ds_name_str))
	splits_list = [[]]*5
	line_count = 0
	with open(in_file, 'r', encoding='utf-8') as fin:
		for id, line in enumerate(fin):
			line_count += 1
			items = line.strip().split('\t')
			eid = int(items[0])
			uid = int(items[1])
			split_id = int(items[2])
			splits_list[split_id] = splits_list[split_id] + [(eid,uid)]
			# not use append(), splits_list[split_id].append((eid,uid)) will wrongly add to all splits;
	assert len(splits_list[0])<line_count/4
	# print('qid_list:', len(splits_list[0]))
	train_list, valid_list, test_list = [], [], []
	qid_list = deque(splits_list)  # my: deque: preferred over list when need quicker append and pop operations
	for fold_id in range(5):
		rotate = fold_id
		map(qid_list.rotate(rotate), qid_list)
		train, valid, test = qid_list[0] + qid_list[1] + qid_list[2], \
						   qid_list[3], qid_list[4]
		# print(fold_id,'train-valid-test:',len(train),len(valid),len(test))
		train_list.append(train)
		valid_list.append(valid)
		test_list.append(test)
	return train_list, valid_list, test_list


def load_tembed(ds_name_str):
	in_file = os.path.join(IN_PARSED, 'tembed_{}.npz'.format(ds_name_str))
	content = np.load(in_file)
	tid_embed_dict = eval(str(content["tembedding_ftaw"]))
	# print(type(tid_embed_dict))
	# # check content
	# for tid,embed in tid_embed_dict.items():
	# 	print(tid, len(embed))
	return tid_embed_dict



if __name__ == '__main__':
	# load_tembed('dsfaces')
	# load_e5fold_train_valid_test('dbpedia')
	load_e5fold_train_valid_test('dsfaces')