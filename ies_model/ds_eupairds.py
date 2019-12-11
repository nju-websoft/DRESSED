'''
modified from origin_eupairds.py: doing
@file: ds_eupairds.py
@author: qxliu
@time: 2019/3/26 9:47
'''

from data.State import *
from data.Evaluator import *


class EUPairDataset_ds(object):
	def __init__(self, eu_pairs, env, dataConfig, deepsetStyle
				 , user=None  # if use pseudo user
				 , isTraining=None # True/False
				 , useAllLabels=False  # never used, always set false
	             , stepLimit=-1  #===== should only be set when isTrain=True
	             , doShuffle=None # true for train
	             , thread_num = 0 # never used  # sequential
	             , name = ''
	             , logTime=False
	             # , GAMMA = default_GAMMA  # can tune
				 # , useTotalReward=False
				 ):
		self.deepsetSytle = deepsetStyle
		self.data = copy.deepcopy(eu_pairs) # deepcopy to avoid influence origin data by shuffle
		self.size = len(self.data)
		self.env = env
		# self.last_eu_idx = -1;
		self.last_eu_idx = 0;
		self.user = user
		self.name = name
		self.logTime = logTime

		self.dataConfig = dataConfig
		self.initSumm = dataConfig.initSumm if isTraining else dataConfig.initSummForTest
		self.dataConfig.doPadding = False  #==== no padding
		# print('doPadding:', self.dataConfig.doPadding, self.env.doPadding) # TODO: check both be False
		self.env.doPadding = False #===== DONE: remove outside setting of env.doPadding
		self.stepLimit = stepLimit
		self.doShuffle = doShuffle if doShuffle!=None else (True if isTraining else False)
		#=======
		self.isTraining = isTraining
		self.byGreedy = not self.isTraining
		self.thread_num = thread_num # TODO: delete

		# self.GAMMA = GAMMA; #=== important  # TODO: add to outside param
		self.GAMMA = dataConfig.GAMMA;  # === important  # TODO: add to outside param
		self.GAMMA_OUT = dataConfig.GAMMA_OUT
		self.useTotalReward = dataConfig.useTotalReward
		self.bad_reward = dataConfig.bad_reward

		self.evaluator = Evaluator()
	def new_epoch(self, doShuffle=None): #== reinit
		# self.last_eu_idx = -1;
		self.last_eu_idx = 0;
		if doShuffle!=None:
			self.doShuffle = doShuffle
		if self.doShuffle:
			random.shuffle(self.data)  # shuffle before every epoch


	def _go_step_for_test(self, policy_model, state, forUReal=True):

		'''
		modified from eupairds4_pad.py
		:param policy_model:
		:param params:
		:return:
		input_matrix: shape=(padded_step_num, padded_cand_num, sa_vec_dim), each row is the feature vector for an <state, action> pair;
		mask_matrix: shape=(padded_step_num, padded_cand_num, 1), for each step, values in [0:cand_num, ] are ones, values in [cand_num:padded_cand_num, ] are zeros;
		label_list: shape=(padded_step_num, padded_cand_num, 1), for each step, is a list of rewards for each <state, action> pair
		'''
		# forUReal = True

		input_matrix = [];
		# mask_matrix = [];
		label_list = [];  # when useAllLabels = False: shape=(step_num, ); when useAllLabels = True: shape=(step_num*max_desc_size, )
		act_mask_list = []
		# ==== 1. init state
		if (state.done):  # skip over already-gold instances (by inserting an all-zero row)
			return None
		else:  # go steps
			# step_input, step_mask = self.env.getTransMatrix(state)
			step_input, _ = self.env.getTransMatrix(state)
			state_vec_tuple = self.env.getStateVecTuple(state, self.deepsetSytle)  # ===== for deepset

			step_reward, next_state, done, act_tid, trans_act_mask, select_actidx = self._make_labels_for_selected_act(
				# state, policy_model, step_input, step_mask, legal_shape
				state, policy_model, step_input  # different with usePadding
				, state_vec_tuple # different with origin
				, forUReal=forUReal
			)
			label_list.append(step_reward)

			# ============== common parts
			# ==== log result
			input_matrix.append(step_input)
			# mask_matrix.append(step_mask)
			act_mask_list.append(trans_act_mask)
			# ===== control loop
			state = next_state
			assert np.shape(input_matrix)[0] == 1  # step_num
		# return input_matrix, mask_matrix, act_mask_list, label_list, act_tid, state;
		return input_matrix, None, act_mask_list, label_list, act_tid, state;

	def _make_labels_for_selected_act(self, state, policy_model, step_input, state_vec_tuple, forUReal=False):
		select_act_idx = policy_model.predict_act_idx(step_input,
													  byGreedy=self.byGreedy
		                                              , step_cand_num = len(state.candidates)
		                                              , step_state_list_tuple=state_vec_tuple)  # use model to select
		# reward_list = np.zeros(legal_shape[1], ).tolist()
		# print("shapespredict:",np.shape(select_act_idx),np.shape(step_input),np.shape(state.candidates))
		act_mask = np.zeros(len(state.candidates), dtype=np.bool).tolist()
		# ==== prepare next step
		act_tid = state.candidates[select_act_idx]
		next_state, act_reward, done = state.makeMove(action=act_tid) if not forUReal else (None, None, None)
		# reward_list[select_act_idx] = act_reward
		act_mask[select_act_idx] = True

		return act_reward, next_state, done, act_tid, act_mask,    select_act_idx



	def _get_episode_for_eu(self, eu, policy_model, useTensorboard = True):
		'''
		different with padding: way of putting trans;
		when apdding: matrix for trains(steps) are put as elements of a list (by append)
		when not padding: elements of trans(steps) are put as elements of a list (by extend), i.e. state vector for different step are on the same dimention
		therefore, within trans, they are the same (except for do not use padding)
		:param eu:
		:param policy_model:
		:return:
		'''
		input_list = [] #  shape=(candNumForAllSteps, vdim)
		act_mask_list = [] # shape=(candNumForAllSteps, 1)
		trans_num = 0; # stepNum
		max_cand_num = 0;
		signal_list = []

		state_vec_list_tuple = ([],[],[]) # curr, cand, disH; each is a


		#==== 1. init state
		timeStart=time.time()
		state0 = State(initType='origin', euPair=eu, dataConfig=self.dataConfig, user=self.user, initSumm=self.initSumm)
		max_cand_num = len(state0.candidates)

		if self.logTime:
			print('initState:',time.time()-timeStart)
		legal_shape = None #============
		if(state0.done): # skip over already-gold instances (by inserting an all-zero row)
			return None
		else: # go steps
			selected_act_tid_list = [] #== for eval
			reward_imm_list = [];  # immediate rewards  # when useAllLabels = False: shape=(step_num, ); when useAllLabels = True: shape=(step_num*max_desc_size, )
			state = state0
			gold = state0.gold
			for t_idx in count():  # step t, get transMatrix, trans_act_mask, trans_act_num (num of candidates), act_tid,
				if self.stepLimit > 0 and t_idx >= self.stepLimit:
					break;
				timeStart = time.time()
				trans_num += 1;
				# trans_act_num = len(state.candidates)
				step_input, _ = self.env.getTransMatrix(state)
				state_vec_tuple = self.env.getStateVecTuple(state, self.deepsetSytle) # ===== for deepset
				# ==== 2.2 select action: predict by model or use gold

				step_reward, next_state, done, act_tid, trans_act_mask, select_actidx = self._make_labels_for_selected_act(
						# state, policy_model, step_input, step_mask, legal_shape
						state, policy_model, step_input # different with usePadding
						, state_vec_tuple
						)
				reward_imm_list.append(step_reward)  # add a value

				signal = 1 if act_tid in gold else 0  # ====
				selected_act_tid_list.append(act_tid)  # === for eval

				#======= log step info
				#============== common parts
				# ==== log result for episode
				input_list.extend(step_input) # shape=(candNum, vdim)
				act_mask_list.extend(trans_act_mask)
				signal_list.append(signal)
				for si in range(3):
					if state_vec_tuple[si] !=None:
						state_vec_list_tuple[si].append(state_vec_tuple[si])

				# TODO: baseline to reward

				#===== control loop
				state = next_state
				# if self.logTime:
				# 	print('step time: ', time.time()-timeStart)
				if done: # end of episode
					break

			# === out of for_t
			timeStart = time.time()
			eval_results = self.evaluator.eval_for_acts(gold, selected_act_tid_list) \
				if useTensorboard else [1]*7
			if self.logTime:
				print('eval_input:', gold, selected_act_tid_list)
				print('eval time:', len(selected_act_tid_list),
				      time.time() - timeStart)

			if self.useTotalReward:
				totalReward = 0.0
				for idx, r in enumerate(reversed(reward_imm_list)):
					totalReward = r + totalReward * self.GAMMA;
				Gt_list = [totalReward]*trans_num
			elif self.GAMMA_OUT !=1.0:
				Gt_list = np.zeros(trans_num) # as final reward_list (label_list)
				Gt=0
				cumu_gamma = 1
				cumu_gamma_out_list = np.zeros(trans_num) # 1, gamma, gamma^2, .... gamma^t (use gamma_out)
				for idx,r in enumerate(reversed(reward_imm_list)): #reversely, t=trans_num-idx-1
					Gt = r + Gt * self.GAMMA;
					# print('idx', idx, r, Gt) # idx: 0,1,2,3,4...
					Gt_list[trans_num-1-idx] = Gt # Gt, put into list reversely

					cumu_gamma_out_list[idx] = cumu_gamma
					cumu_gamma = cumu_gamma * self.GAMMA_OUT
				# print('rew_imm:',list(reward_imm_list))
				# print('Gt_list1:',list(Gt_list))
				# print('cumu_gamma:',list(cumu_gamma_out_list))
				Gt_list = Gt_list * cumu_gamma_out_list # gamma^t * Gt
				# print('Gt_list2:',list(Gt_list))
			else:
				Gt_list = np.zeros(trans_num) # as final reward_list (label_list)
				Gt=0
				for idx,r in enumerate(reversed(reward_imm_list)): #reversely, t=trans_num-idx-1
					Gt = r + Gt * self.GAMMA;
					# print('idx', idx, r, Gt)
					Gt_list[trans_num-1-idx] = Gt # put into list reversely
		#===== out of if-else
		return input_list, act_mask_list, trans_num, max_cand_num, Gt_list, eval_results\
			, state_vec_list_tuple

	def _get_evaluate_for_eu(self, eu, policy_model, forDev=True, out_file=None):
		'''
		for dev-test for WSDM20;
		log file format:
		"eu:\t"(eid,uid)
		"eval:\t"(step,ndcg)
		"act_list:\t"act_tid_list
		"dislike_list:\t"dislike_tid_list
		"summ_list:\t"summ_list  # include init_summ (as item [0])
		:param eu:
		:param policy_model:
		:return:
		'''
		log_dislike_list = []
		log_act_list = []
		log_signal_list = []
		log_summ_list = []
		log_summ_signal_list = []  # # for eval, generated summaries during interactions (not include initSumm)

		#==== 1. init state
		timeStart=time.time()
		state0 = State(initType='origin', euPair=eu, dataConfig=self.dataConfig, user=self.user, initSumm=self.initSumm)

		log_summ_list.append(list(state0.current))
		log_summ_signal_list.append([(1 if tid in state0.gold else 0) for tid in state0.current])

		if self.logTime:
			print('initState:',time.time()-timeStart)
		if(state0.done): # skip over already-gold instances (by inserting an all-zero row)
			return None
		else: # go steps
			state = state0
			gold = state0.gold

			for t_idx in count():  # step t, get transMatrix, trans_act_mask, trans_act_num (num of candidates), act_tid,
				log_dislike_list.append(state.disItem)
				if self.stepLimit > 0 and t_idx >= self.stepLimit:  # TODO: repair NDCG in eval when use stepLimit
					break;
				_, _, _, _, act_tid, next_state = self._go_step_for_test(policy_model, state, forUReal=False) # should set forUReal=False
				log_act_list.append(act_tid)
				signal = 1 if act_tid in gold else 0  # ====
				log_signal_list.append(signal)

				log_summ_list.append(next_state.current)
				log_summ_signal_list.append([(1 if tid in gold else 0) for tid in next_state.current])

				# TODO: baseline to reward

				#===== control loop
				state = next_state
				# if self.logTime:
				# 	print('step time: ', time.time()-timeStart)
				if state.done: # end of episode
					break

			# === out of for_t
			# timeStart = time.time()
			eval_results = self.evaluator.eval_for_dev(gold, log_act_list) if forDev else \
					self.evaluator.eval_for_test(gold, log_act_list, log_summ_signal_list)
			# if forDev=True: partNDCG
			# if forDev=False: tuple(step,ndcg,ndcf)

			if out_file is not None:
				with open(out_file,'at') as f:
					f.write('eu:\t'+str(eu)+'\n')
					f.write('eval:\t'+str(eval_results)+'\n')
					f.write('gold:\t' + str(gold) + '\n')
					f.write('act_list:\t'+str(log_act_list)+'\n')
					f.write('dislike_list:\t'+str(log_dislike_list)+'\n')
					f.write('summ_list:\t'+str(log_summ_list)+'\n') # include init_summ

		#===== out of if-else
		return eval_results # for dev-test log

	def get_eval(self, policy_model, batch_size, forDev=True, out_file=None):
		batch_start_idx = self.last_eu_idx #self.last_eu_idx+1; # include
		batch_end_idx = batch_start_idx + batch_size; # exclude
		batch_data = self.data[batch_start_idx:batch_end_idx]
		# print('start_end:',batch_start_idx,batch_end_idx,batch_size,len(batch_data))
		self.last_eu_idx = batch_end_idx
		eval_results = []
		for eu in batch_data:  # for each episode
			eu_eval_result = self._get_evaluate_for_eu(eu, policy_model, forDev=forDev,
														  out_file=out_file)  # forDev=True # === different with train: forDevTest=False
			if eu_eval_result is not None:  # remove initially correct eu
				eval_results.append(eu_eval_result)
		return eval_results
	def has_next(self):
		return self.last_eu_idx<self.size
	def get_batch(self, policy_model, batch_size, useTensorboard=True):
		timeStart = time.time()
		#===== 1. get batch data
		batch_start_idx = self.last_eu_idx#self.last_eu_idx+1; # include
		batch_end_idx = batch_start_idx + batch_size; # exclude
		if self.logTime:
			print('batch_start_end:', batch_start_idx, batch_end_idx, batch_size, (batch_end_idx>=self.size))
		# print("has_next:",self.has_next(), self.last_eu_idx, batch_start_idx, self.size,(batch_start_idx >= self.size))
		if (batch_start_idx >= self.size):  # remove the last-unenough batch
			return None
		# if(batch_end_idx>self.size): # remove the last-unenough batch
		# 	return None

		batch_data = self.data[batch_start_idx:batch_end_idx]
		self.last_eu_idx = batch_end_idx;

		# ===== 2. run episode sequential
		batch_triple_list = [] # 0
		batch_act_mask_list = [] # 1
		trans_start_idx_list = [] # 2, 3 # log row scope (cand triples) of each trans; for generating trans_mask
		batch_label_list = []  # 4
		eval_results_list = [] # 5
		batch_trans_num = 0
		trans_start_idx = 0
		avg_eval_results = 0

		batch_trans_triple_num_list = []
		batch_state_vec_list_tuple = ([],[],[]) #7,8,9 (curr,cand,disH)


		for eu in batch_data: # for each episode
			episode_tuple = self._get_episode_for_eu(eu, policy_model, useTensorboard=useTensorboard)
			if episode_tuple is None:
				continue
			batch_triple_list.extend(episode_tuple[0])
			batch_act_mask_list.extend(episode_tuple[1])

			state_vec_list_tuple=episode_tuple[6]
			for si in range(3):
				batch_state_vec_list_tuple[si].extend(state_vec_list_tuple[si])

			# ==== for generating trans_mask
			episode_trans_num = episode_tuple[2]
			max_cand_num = episode_tuple[3]
			batch_trans_num += episode_trans_num
			trans_triple_num = max_cand_num
			for trains_idx in range(episode_trans_num):
				batch_trans_triple_num_list.append(trans_triple_num) #==== for deepset
				#===== for mask
				trans_start_idx_list.append(trans_start_idx)  # save index info
				trans_start_idx += trans_triple_num  # update for the next trans
				trans_triple_num -= 1 # update for the next trans
			#=================

			batch_label_list.extend(episode_tuple[4]) # Gt_list
			eval_results = episode_tuple[5]
			if (len(eval_results) > 0 and eval_results[0] > 0):  # jump over no-interact eus
				eval_results_list.append(eval_results)
		#==== out for_eu
		avg_eval_results = np.mean(eval_results_list, axis=0).tolist() # shape=(7,)
		print("batch info:",self.last_eu_idx,np.shape(batch_triple_list), np.shape(batch_act_mask_list), np.shape(trans_start_idx_list)
		      , np.shape(batch_label_list), time.time()-timeStart,'s')
		if len(batch_triple_list)==0: # when batch_size=1 and meet illegal eu
			return None
		if useTensorboard: # otherwise, avg_eval_results is meaningless (not actually calculated for speed up)
			print("results:",np.shape(eval_results_list),avg_eval_results)

		# ==== for generating trans_mask
		batch_triple_num, _ = np.shape(batch_triple_list)
		batch_trans_mask = np.zeros(shape=(batch_trans_num,
		                             batch_triple_num))  # each column for one triple (as the transpose of triple_matrix), each row for one trans (to be decomposed by tf.untack)
		for trans_idx in range(batch_trans_num):  # set values to the mask
			trans_triple_start = trans_start_idx_list[trans_idx]
			trans_triple_end = batch_triple_num if trans_idx + 1 == batch_trans_num else trans_start_idx_list[
				trans_idx + 1]
			# print('trans_idx', trans_idx, trans_triple_start, trans_triple_end)
			batch_trans_mask[trans_idx,
			trans_triple_start:trans_triple_end] = 1  # make the triples for current trains (row) be 1, others be zeros
		# print('row', list(trans_mask[trans_idx]))

		#===== 3. post-process (inputs, labels, evals)
		return batch_triple_list, batch_trans_mask, batch_trans_num\
			, batch_act_mask_list, batch_label_list\
			, avg_eval_results\
			, eval_results_list \
			, batch_trans_triple_num_list \
			, batch_state_vec_list_tuple[0]\
			, batch_state_vec_list_tuple[1]\
			, batch_state_vec_list_tuple[2] \

class RandomModel(): # used for observe

	def _predict_act_probs(self, step_input, state_list_tuple=([],[],[])):
		act_probs = np.random.random_sample(len(step_input))
		return act_probs

	def _select_act_idx(self, predict_act_probs_list, byGreedy):
		if byGreedy:
			return predict_act_probs_list.index(max(predict_act_probs_list))
		else:
			item_num = len(predict_act_probs_list)
			# print("item_num:",item_num)
			selected_idx = np.random.randint(item_num)
			return selected_idx

	def predict_act_idx(self, step_input, byGreedy, state_list_tuple):
		act_probs = self._predict_act_probs(step_input)
		# print('act_probs:', np.shape(act_probs))
		selected_idx = self._select_act_idx(act_probs, byGreedy)
		return selected_idx
	def update(self, batch_data):
		print('updating:',np.shape(batch_data))  # updating: (6,)
		for i in range(len(batch_data)):
			print('data:',i, np.shape(batch_data[i]))

