'''
@file: DataConfig.py
@author: qxliu
@time: 2019/2/24 15:30
'''

from data.constants import *
import util.dbUtil as dbtl
# import util.dbUtil_WSDM as dbtlw
# import util.StaticFeature as sf
class DataConfig:
	def __init__(self
				 #=== enum items
				 , dsName, topK
	             , dataSplit=None, foldId=-1
				 , ftypes=None, initSumm=None
	             , randGoldNum=None #=== only used when initSumm=InitSumm.ctlrand
				 , encodeStyle=''
				 #===== fixed values
				 , cleaned = True
				 , byUser = True
				 , edistinct = True
	             , forUreal = False
				 , gamma = 0.0
				 , gamma_out = 1.0
				 , useTotalReward = False
				 , is_try = False
				 , reward_strategy='rdcg'
				 ):
		self.dsName = dsName
		self.topK = topK
		self.dataSplit = dataSplit # values: diff, 10fold
		self.foldId = foldId  # only useful when dataSplit='10fold'
		self.ftypes = ftypes  # only for one-triple
		self.initSumm = initSumm # for train
		self.initSummForTest = initSumm # for test
		self.randGoldNum = randGoldNum #=== only used when initSumm=InitSumm.ctlrand; always set according to batches

		self.encodeStyle = encodeStyle # for un-train encoder:
		# '', 'cosine_avg_curr_dislike_history',
		# 'mean_curr_dislike_history', 'mean_curr_dishis'
		# old: 'pdiv_avg_curr_dislike_history' (un-implemented)
		# encode methods: cosine, mean
		# encode items: curr, dislike, history, dishis, cand
		#=== data split
		self.byUser = byUser
		self.cleaned = cleaned
		self.edistinct = edistinct
		self.forUreal = forUreal
		self.GAMMA = gamma
		self.GAMMA_OUT = gamma_out
		self.useTotalReward = useTotalReward
		# #==== for State
		self.is_try = is_try
		self.reward_strategy = reward_strategy

	def loadTables(self):
		print('loading tables...')
		if not self.forUreal:
			dbtl._initTripleCleanMap()#(rmLMDBBnode=self.rmLMDBBnode)
		if 'embed' in self.ftypes:
			dbtl._initEmbeddingForTid_newaw()
		dbtl._initDescByEid(self.cleaned)
		print('finished loading')

	def __str__(self):
		attrs = vars(self)
		attrValStr = ', '.join(str(item) for item in attrs.items())
		return attrValStr

	def getDataE5Fold_TrainDevTest(self, foldId=None):
		if(foldId==None):
			return dbtl.getE5FoldTrainDevTest(self.dsName, self.cleaned, topK=self.topK)
		else:
			assert foldId<5
			trainList, devList, testList = dbtl.getE5FoldTrainDevTest(self.dsName, self.cleaned, topK=self.topK)
			return trainList[foldId], devList[foldId], testList[foldId]

	def getDataAll(self):
		return dbtl.getEUIdList(self.dsName, topK=self.topK, cleaned=True)