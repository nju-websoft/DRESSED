'''
@file: policy_user.py
@author: qxliu
@time: 2018/11/15 22:07
'''
import pickle as pkl  # for model save and load
import random

import numpy as np
from sklearn.neural_network import multilayer_perceptron as mlp

from ies_umodel.umodel_params import *
import ies_umodel.umodel_db as dutl

default_umVersion='z'

def _getModelFilePath(dsName, modelDir, forModel, umVersion=default_umVersion):
	filePath = modelDir + 'm' + str(umVersion) + '_'+str(dsName); # model path
	if not forModel: #== for logfile
		filePath = filePath.replace('.pkl','_mlog.txt');
	print('filePath:',filePath)
	return filePath

class UserPolicy(object):
	def __init__(self, user_style, dsName
	             , modelDir=DEFAULT_USER_MODEL_DIR, umVersion='z' #=== for init umodel from model file (only when user_style='learned')
	             , train=None, test=None  #=== for init umodel by src (only when user_style='learned')
	             , dataConfig=None #=== for init by RL model
	             ):
		'''
		(20191116: modified from eSummRLPy2.iesumodel.policy_user)
		=========
		user_styles:
			'order': select by the order in candidate list, dislike the last, like the first
			'random': select randomly
			'learned': select by the learned user model
		:param user_style:
		:param dsName: 'dbpedia', 'lmdb', 'dsfaces'
		:param cleaned:
		:param modelDir:
		:param umVersion:
		:param train: only useful when user_style='learned', the training set for get user when model file is not given
		:param test:
		:param fcfg:
		'''
		self.user_style = user_style
		self.dsName = dsName

		if(dataConfig is not None):
			self.dsName = dataConfig.dsName.name

		#=== specific for user_style=='learned'
		self.modelDir = modelDir
		self.umVersion = umVersion # if foldId is not None else UserPolicy.global_FOLD_ID;
		self.train = train # as training set for learned user
		self.test = test # as validation set for user model selection

		if user_style=='learned':
			self.initUserModel()

	def initUserModel(self):
		if self.train is None: #=== load from file
			# raise "no src data for userModel! self.src="+str(self.src)
			self.umodel = self._getModelFromFile(self.modelDir, self.umVersion)
			if type(self.umodel) is str:
				raise Exception("wrong args for getModelFromFile, modelDir=%s, foldId=%s" %(self.modelDir,self.umVersion))
		else: #=== src
			raise Exception("_getModelByTrain")
	def _getModelFromFile(self, dsName, modelDir, umVersion):
		modelPath = _getModelFilePath(modelDir=modelDir, forModel=True, umVersion=umVersion)
		print('initing faces umodel from file: ', modelPath)
		with open(modelPath, 'rb') as file:
			umodel = pkl.load(file)
		return umodel

	def getProbDislike(self,disCandidates):
		prob_dislike=[]
		num = len(disCandidates)
		if (num == 0):  # current == gold, should be end
			return prob_dislike;
		elif (self.user_style == "learned"):
			x_candidates = []
			for tid in disCandidates:
				# fvec = self.fcfg.getFeature(dbName=default_dsName, tid=tid, onlyStatic=onlyStaticForUmodel)
				fvec = dutl.getFeature(self.dsName, tid) # get featureVector of a triple
				x_candidates.append(fvec)

			y = self.umodel.predict_proba(x_candidates)  # return <class 'numpy.ndarray'>s
			# print('probs:',y)
			prob_dislike = y[:, 0]  # the 0-th column of predict_proba, i.e. the prob of class=0(dislike) for each candidates
		return prob_dislike

	def getDislike(self, disCandidates):
		'''
		coppied from state._simulateDislike
		:param eid:
		:param disCandidates:
		:return:
		'''
		disItem = -1
		num = len(disCandidates)
		if (num == 0):  # current == gold, should be end
			return disItem;
		elif (num == 1):
			return disCandidates[0]
		elif (self.user_style == "order"):
			disItem = disCandidates[-1]  # the last item
		elif (self.user_style == "random"):
			disItem = random.choice(disCandidates)
		elif (self.user_style == "learned"):
			x_candidates=[]
			for tid in disCandidates:
				# fvec = self.fcfg.getFeature(dbName=default_dsName, tid=tid, onlyStatic=onlyStaticForUmodel)
				fvec = dutl.getFeature(self.dsName, tid) # get featureVector of a triple
				x_candidates.append(fvec)

			y = self.umodel.predict_proba(x_candidates) # return <class 'numpy.ndarray'>s
			prob_dislike = y[:,0] # the 0-th column of predict_proba, i.e. the prob of class=0(dislike) for each candidates
			idx = list(prob_dislike).index(np.max(prob_dislike))
			disItem = disCandidates[idx]
		else:
			raise Exception("getLikeByAlgo")
		return disItem;



if __name__ == '__main__':
	user = UserPolicy(user_style='learned', dsName='dbpedia')
	summ = [9,5,7,10,4]
	dis = user.getDislike(summ)
	print('dis:',dis)
