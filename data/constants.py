'''
@file: Constants.py
@author: qxliu
@time: 2019/2/24 15:35
'''
#==== common imports
import numpy as np
import copy
import math
import time as time
from itertools import count
from collections import OrderedDict

#==== necessory imports
import os
from os.path import dirname, abspath
from enum import Enum


#================== remote server (for v2 implement)




# ====== 1. common enum (moved from util.StaticFeature)
class DS_Names(Enum):
	all = 0;
	dbpedia = 1;  # lef-side is name (dsName), right-side is value (first eid)
	lmdb = 101;
	dsfaces = 10000;


class TOPK(Enum):
	top5 = 5;  # lef-side is name, right-side is value
	top10 = 10;


class Init_Summ(Enum):
	cd = 'cd_o_rm' # 'cd_rm'

default_InitSumm = Init_Summ.cd


# ====== 2. global settings
# DEFAULT_OUT_DIR = os.path.dirname(__file__)+'/inout/'
DEFAULT_OUT_DIR = dirname(dirname(abspath(__file__)))+'/inout/'
default_LOG_DIR = DEFAULT_OUT_DIR+'logs/'
print('log path: ', default_LOG_DIR)
if not os.path.exists(default_LOG_DIR):
	os.makedirs(default_LOG_DIR)

#===== for features
DEFAULT_EMBED_VERSION = 'newaw'
FASTTEXT_EMBED_DIM = 300  # dim of pVec or vVec from fast text

#==== coppy from umodel.FSParams
default_topK = TOPK.top5
default_Cleaned=True #

def convertStr2IntList(listStr):
	tmp = listStr.split(', ')  # has empty item
	tmp2 = [int(x) for x in tmp if x!=''] # remove empty items and convert to int
	# print('tmp', tmp, tmp2)
	return tmp2


# if __name__ == '__main__':
# 	print('dir:',default_LOG_DIR)
# 	with open(default_LOG_DIR+'testlog.txt','w') as f:
# 		f.write('first line')
