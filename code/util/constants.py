'''
@file: Constants.py
@author: qxliu
@time: 2019/2/24 15:35
'''
from enum import Enum


# ====== 1. common enum (moved from util.StaticFeature)
class DS_Names(Enum):
	dbpedia = 1;  # lef-side is name (ds_name), right-side is value (first eid)
	lmdb = 101;
	dsfaces = 1;

class TOPK(Enum):
	top5 = 5;  # lef-side is name, right-side is value
	top10 = 10;
