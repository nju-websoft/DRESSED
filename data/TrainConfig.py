'''
@file: TrainConfig.py
@author: qxliu
@time: 2019/2/24 15:34
'''
import tensorflow as tf
from data.constants import *


class TrainConfig:
	def __init__(self
	             , num_repeats=1
	             , num_features=-1
	             , is_train=True
	             , batch_size = 1
	             , batch_num = np.inf #==== should infer from dataConfig size
	             , epoch_num = 50 #==== num of rounds
	             , dev_step = 1
	             , learning_rate = 0.01
	             , type_activation = tf.nn.leaky_relu
				 , type_activation_out = tf.nn.leaky_relu
	             , num_h1_units = 64
	             , dropout_rate_for_train = 0.5
				 #==== for deepset
				 , deepset_style = 'triple_curr_cand_disH'
	             , deepset_encode_hidden_units = 300
	             , deepset_encode_out_units = 150
	             #==== for log
	             , name='try'
	             , signature=str(time.time())
				 , do_debug = False
				 , do_rev_folds = False
				 , use_neg = True
	             ):
		#===== environment
		self.num_repeats = num_repeats
		self.IS_TRAIN = is_train
		self.do_debug = do_debug
		self.do_rev_folds = do_rev_folds

		#===== tunnable
		self.BATCH_SIZE = batch_size
		self.BATCH_NUM = batch_num
		self.EPOCH_NUM = epoch_num
		self.DEV_STEP = dev_step

		#===== for deepset
		self.deepsetStyle = deepset_style
		#'the optimal size of the hidden layer is usually between the size of the input and size of the output layers'. Jeff Heaton, author of Introduction to Neural Networks in Java offers a few more.
		self.deepset_ENCODE_HIDDEN_UNITS = deepset_encode_hidden_units
		self.deepset_ENCODE_OUT_UNITS = deepset_encode_out_units

		#==== struct
		self.NUM_H1_UNITS = num_h1_units
		self.TYPE_ACTIVATION = type_activation
		self.TYPE_ACTIVATION_OUT = type_activation_out
		self.LEARNING_RATE = learning_rate
		self.DROPOUT_RATE_FOR_TRAIN = dropout_rate_for_train#0.5
		self.USE_NEG = use_neg
		#==== fixed
		self.name = name
		self.signature = signature
		self.MODEL_FILE_DIR = default_LOG_DIR  # end with '/'
		self.MODEL_FILE_PATH_FULL = None
		self.NUM_OUT_UNITS = 1
		self.NUM_FEATURES = num_features  # init by env, the size of the lowest dimension of input_matrix
		self.DROPOUT_RATE_FOR_TEST = 0.0



	def __str__(self):
		attrs = vars(self)
		attrValStr = ', '.join(str(item) for item in attrs.items())
		return attrValStr

	def setNumFeatures(self, num_features):
		self.NUM_FEATURES = num_features
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
		return self.MODEL_FILE_DIR + self.name+'_'+model_name+'_'+self.signature+'.txt'
traincfg_origin = TrainConfig() # all use default

traincfg_origin_small = TrainConfig(name='origin_small', batch_size=10, batch_num=2, epoch_num=3, signature=str(time.time()))












