'''
@file: TrainConfig.py
@author: qxliu
@time: 2019/2/24 15:34
'''
import os
import time
import numpy as np
import tensorflow as tf

class TrainConfig:
	def __init__(self
	             , num_repeats=1
	             , batch_size = 16
	             , batch_num = np.inf #==== should infer from dataConfig size
	             , epoch_num = 50 #==== num of rounds
	             , dev_step = 1
	             , learning_rate = 0.01
				 #==== for deepset
				 , deepset_style = None
	             #==== for log
	             , name='try'
	             , signature=str(time.time())
				 , do_rev_folds = False
	             ):
		#===== environment
		self.num_repeats = num_repeats
		self.do_rev_folds = do_rev_folds

		#===== tunnable
		self.BATCH_SIZE = batch_size
		self.BATCH_NUM = batch_num
		self.EPOCH_NUM = epoch_num
		self.DEV_STEP = dev_step
		self.LEARNING_RATE = learning_rate
		#====== struct
		self.deepset_style = deepset_style
		#==== fixed
		self.name = name
		self.signature = signature
		self.MODEL_FILE_PATH_FULL = None



	def __str__(self):
		attrs = vars(self)
		attrValStr = ', '.join(str(item) for item in attrs.items())
		return attrValStr

	def setSignature(self, signature):
		self.signature = str(signature)
	def setModelFileName(self, model_file_name):
		# self.MODEL_FILE_PATH_FULL = self.MODEL_FILE_DIR + self.name+'_'+model_file_name+'_'+self.signature
		self.MODEL_FILE_PATH_FULL = model_file_name
	def setBatchNum(self, batch_num):
		self.BATCH_NUM = batch_num
	def setName(self, name):
		self.name = name
	def getLogFileName(self, model_name):
		# return self.MODEL_FILE_DIR + self.name+'_'+model_name+'_'+self.signature+'.txt'
		# return os.path.join(self.MODEL_FILE_DIR, '{}_{}_{}.txt'.format(self.name, model_name, self.signature))
		return os.path.join(os.path.dirname(self.MODEL_FILE_PATH_FULL), '{}_{}_{}.txt'.format(self.name, model_name, self.signature))












