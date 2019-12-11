'''
get <state, action> vector
@file: Env.py
@author: qxliu
@time: 2019/2/24 17:11
'''
from data.constants import *
import util.StrUtils as sutl
import data.TripleFeature as tplf

class Env(object):
	def __init__(self, dataConfig
	             ):
		self.tfcfg = tplf.TripleFeature(dataConfig)  # action-related part

		# self.dsname = dataConfig.dsName
		self.cleaned = dataConfig.cleaned
		self.dynFNames = self.tfcfg.otherFNames

		self.dCfg = dataConfig
		self.vecDim = -1

		#====== dynFeatures (un-train)
		self.encodeStyle = dataConfig.encodeStyle # str
		print('encodeStyle',self.encodeStyle)
		self.withCurr = ('curr' in self.encodeStyle)
		self.withDislike = ('dislike' in self.encodeStyle)
		self.withHistory = ('history' in self.encodeStyle)
		self.withDisHis = ('dishis' in self.encodeStyle) # union(history, {dislike})
		self.withCand = ('cand' in self.encodeStyle)

		self.getFDim()
	def __str__(self):
		return '{fcfg:'+str(self.tfcfg.getFNames())+', encodeStyle:'+str(self.encodeStyle)+', fDim:'+str(self.getFDim())+'}'

	def getFDim(self):
		if self.vecDim<0:
			self.vecDim = self.tfcfg.getFDim();

			dynItemSize = 0;
			if self.encodeStyle.startswith('cosine') \
				or self.encodeStyle.startswith('pdiv'):
				dynItemSize = 1
			elif self.encodeStyle.startswith('mean'):
				dynItemSize = self.tfcfg.getFDim()

			if self.withCurr:
				self.vecDim += dynItemSize;
			if self.withDislike:
				self.vecDim += dynItemSize;
			if self.withHistory:
				self.vecDim += dynItemSize;
			if self.withDisHis:
				self.vecDim += dynItemSize;
			if self.withCand:
				self.vecDim += dynItemSize;

		return self.vecDim

	def getStateVecTuple(self, state, deepsetStyle):
		'''
		for deepset, return a tuple with len=3;
		for item not to encode, set a int number
		:param act:
		:param state:
		:return:
		'''
		state_vec_tuple = ()
		if 'curr' in deepsetStyle:
			numPad = state.topK.value-1
			curr_vec_list = self._getStateMatrix(state.curr_rest, numPad)
			state_vec_tuple += (curr_vec_list,)
		else:
			state_vec_tuple += (None,)
		if 'cand' in deepsetStyle:
			numPad = self.dCfg.MAX_CAND_LENGTH
			cand_vec_list = self._getStateMatrix(state.candidates, numPad)
			state_vec_tuple += (cand_vec_list,)
		else:
			state_vec_tuple += (None,)
		if 'disH' in deepsetStyle:
			numPad = self.dCfg.MAX_CAND_LENGTH
			dis_his = list(state.disHistory)
			dis_his.append(state.disItem)
			disH_vec_list = self._getStateMatrix(dis_his, numPad)
			state_vec_tuple += (disH_vec_list,)
		else:
			state_vec_tuple += (None,)
		return state_vec_tuple

	def _getStateMatrix(self, tid_list, numPad):
		vec_list = []
		for tid in tid_list:
			t_vec = self.tfcfg.getFeature(tid)
			vec_list.append(t_vec)

		# #===== add paddings
		item_shape = (self.getFDim(),)
		vec_list = sutl.getPaddedList(vec_list, numPad, item_shape)
		return vec_list

	def getTransMatrix(self, state):
		'''
		for padding
		:param act_list:
		:param state:
		:return: matrix (list of list) with of (act_num, vec_dim);
		 should keep the order with state.candidates
		'''
		vec_list = []
		mask_list = None
		act_list = state.candidates
		if len(act_list) <= 0:
			return vec_list

		for act_tid in act_list:
			act_vec = self.tfcfg.getFeature(act_tid) # np.array
			state_vec = self._encodeState(act_tid, state)  #==== np.array
			sa_vec = np.concatenate((act_vec, state_vec)).tolist() # convert to list
			vec_list.append(sa_vec)
		return vec_list, mask_list
	def _encodeState(self, act, state):
		'''
		may slow
		simple encoder without train (usually as replacement of deepsets)
		:return: np.array
		'''
		state_vec = []
		# curr_vec=[]; dis_vec=[]; his_vec=[]; dishis_vec=[]; cand_vec=[]
		if self.withCurr:
			curr_vec = self._encodeList(act, state.curr_rest) # not include dislike
			state_vec.extend(curr_vec)
		if self.withDislike:
			# dis_vec = self.tfcfg.getFeature(state.disItem)
			dis_vec = self._encodeList(act, [state.disItem])
			state_vec.extend(dis_vec)
		if self.withHistory:
			his_vec = self._encodeList(act, state.disHistory)
			state_vec.extend(his_vec)
		if self.withDisHis:
			dis_his = list(state.disHistory)
			dis_his.append(state.disItem)
			dishis_vec = self._encodeList(act, dis_his)
			state_vec.extend(dishis_vec)
		if self.withCand:
			cand_rest = list(state.candidates)
			cand_rest.remove(act)  # not include act
			cand_vec = self._encodeList(act, cand_rest)
			state_vec.extend(cand_vec)
		return state_vec

	def _encodeList(self, act, tid_list):
		# list_vec = []
		if 'highest' in self.encodeStyle:
			tid_vecs = []
			if len(tid_list)==0: # for history, cand
				list_vec = np.zeros(self.tfcfg.getFDim()).tolist()
			else:
				for tid in tid_list:
					vec = self.tfcfg.getFeature(tid)
					tid_vecs.append(vec)
				list_vec = np.max(tid_vecs, axis=0).tolist() # highest values in each dimention
				assert len(list_vec)==np.shape(tid_vecs)[1] # same dimention
		elif 'mean' in self.encodeStyle:
			tid_vecs = []
			if len(tid_list)==0: # for history, cand
				list_vec = np.zeros(self.tfcfg.getFDim()).tolist()
			else:
				for tid in tid_list:
					vec = self.tfcfg.getFeature(tid)
					tid_vecs.append(vec)
				list_vec = np.mean(tid_vecs, axis=0).tolist() # avg values in each dimention
		elif 'cosine' in self.encodeStyle:
			cosine_list = [];
			if len(tid_list)==0: # for history, cand
				list_vec = [0.0]
			else:
				for tid in tid_list:
					cosine = self.tfcfg.getCosine(act, tid)
					cosine_list.append(cosine)
				cosine_out = None
				if 'avg' in self.encodeStyle:
					cosine_out = np.mean(cosine_list)
				if 'max' in self.encodeStyle:
					cosine_out = np.max(cosine_list)
				assert cosine_out is not None
				list_vec = [cosine_out]
		else :
			raise Exception('illegal encodeStyle! encodeStyle=', self.encodeStyle, ', currently only support "mean" and "cosine"')
		return list_vec
