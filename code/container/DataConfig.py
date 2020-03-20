'''
@file: DataConfig.py
@author: qxliu
@time: 2019/2/24 15:30
'''

# from code import util as dbtl
import util.load_files as fld
from util.constants import *


class DataConfig:
	def __init__(self
				 #=== enum items
				 , ds_name
				 # , topK
	             , dataSplit=None, foldId=-1
				 #===== fixed values
				 , byUser = True
	             , forUreal = False
				 , gamma = 0.6
				 , gamma_out = 0.6
				 ):
		self.ds_name = ds_name
		self.dataSplit = dataSplit # values: diff, 10fold
		self.foldId = foldId  # only useful when dataSplit='10fold'
		#=== container split
		self.byUser = byUser
		# self.edistinct = edistinct
		self.forUreal = forUreal
		self.GAMMA = gamma
		self.GAMMA_OUT = gamma_out
		#==== constant
		self.NUM_FEATURES = 600 # dim of tembed
		self.reward_strategy = 'rdcg'
		self.ftypes = 'embed'
		self.cleaned = True
		self.topK = TOPK.top5
		self.bad_reward = 0.0

		self._load_data()

		#=== by compute
		self.MAX_CAND_LENGTH = self.MAX_DESC_LENGTH - self.topK.value


	def _load_data(self):
		print('loading data...')
		self.eid_desc_dict, self.MAX_DESC_LENGTH = fld.load_desc(self.ds_name.name)
		self.eu_gold_dict = fld.load_gold(self.ds_name.name, self.topK.name)
		self.eid_inisumm_dict = fld.load_init_summ(self.ds_name.name)
		self.tid_embed_dict = fld.load_tembed(self.ds_name.name)
		self.train_list, self.valid_list, self.test_list = fld.load_e5fold_train_valid_test(self.ds_name.name)
		print('finished loading!')

	def __str__(self):
		attrs = vars(self)

		attrValStr = ', '.join(str((k,v)) for k,v in attrs.items() if not k.endswith('_dict') and not k.endswith('_list'))
		return attrValStr

	def get_train_valid_test(self, fold_id=None):
		if fold_id == None:
			return self.train_list, self.valid_list, self.test_list
		else:
			return self.train_list[fold_id], self.valid_list[fold_id], self.test_list[fold_id]

	def get_desc_by_eid(self, eid, cleaned=True):
		return self.eid_desc_dict.get(eid)
	def get_gold_by_eupair(self, eupair, topk=TOPK.top5, cleaned=True):
		return self.eu_gold_dict.get(eupair)
	def get_inisumm_by_eid(self, eid, topk=TOPK.top5, cleaned=True):
		return self.eid_inisumm_dict.get(eid)
	def get_tembed_by_tid(self, tid):
		return self.tid_embed_dict.get(tid)
