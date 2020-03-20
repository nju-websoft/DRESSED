'''
get <state, action> vector
@file: Env.py
@author: qxliu
@time: 2019/2/24 17:11
'''

import util.str_utils as sutl


class Env(object):
	def __init__(self, dataConfig
	             ):
		self.dCfg = dataConfig
		self.vecDim = self.dCfg.NUM_FEATURES

	def getStateVecTuple(self, state, deepset_style):
		'''
		for deepset, return a tuple with len=3;
		for item not to encode, set a int number
		:param act:
		:param state:
		:return:
		'''
		state_vec_tuple = ()
		if 'curr' in deepset_style:
			numPad = state.topK.value-1
			curr_vec_list = self._getStateMatrix(state.curr_rest, numPad)
			state_vec_tuple += (curr_vec_list,)
		else:
			state_vec_tuple += (None,)
		if 'cand' in deepset_style:
			numPad = self.dCfg.MAX_CAND_LENGTH
			cand_vec_list = self._getStateMatrix(state.candidates, numPad)
			state_vec_tuple += (cand_vec_list,)
		else:
			state_vec_tuple += (None,)
		if 'disH' in deepset_style:
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
			t_vec = self.dCfg.get_tembed_by_tid(tid)
			vec_list.append(t_vec)

		# #===== add paddings
		item_shape = (self.vecDim,)
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
			act_vec = self.dCfg.get_tembed_by_tid(act_tid) # np.array
			vec_list.append(act_vec)
		return vec_list#, mask_list

