'''
origin + deepset
@file: ds_modeling.py
@author: qxliu
@time: 2019/3/26 9:47
'''

from data.constants import *
import tensorflow as tf
TYPE_ACTIVATION_FOR_SET_ENCODER = tf.nn.relu

class PolicyNN_ds(object):
	def __init__(self, dataConfig, trainConfig, sess=None, name='policyNN_ds', num_to_save=3
				 , restore_path=None
				 # , useNeg = True
				 ):
		print('init', name)
		self.sess = sess
		self.isTrain = trainConfig.IS_TRAIN
		self.dataConfig = dataConfig
		self.trainConfig = trainConfig
		self.useAllLabels = trainConfig.USE_ALL_LABELS
		self.name = name
		self.restore_path = restore_path


		self.ENCODE_HIDDEN_UNITS = self.trainConfig.deepset_ENCODE_HIDDEN_UNITS
		self.ENCODE_OUT_UNITS = self.trainConfig.deepset_ENCODE_OUT_UNITS # encode to same size as input; TODO: change output_dim of encoder e.g. = self.trainConfig.NUM_FEATURES/2
		self.use_neg = trainConfig.USE_NEG # default True
		self.is_try = dataConfig.is_try # default False
		# self.epsilon = trainConfig.epsilon
		self.deepsetStyle = trainConfig.deepsetStyle

		# print('ds_model:', self.use_neg, trainConfig.USE_NEG)
		self._build_model()
		self.saver = tf.train.Saver(max_to_keep=num_to_save)#(max_to_keep=3)
		self.init = tf.initialize_all_variables()  # tf.global_variables_initializer()  # must after model building


	# warn: Use `tf.global_variables_initializer` instead.

	# self.saver.restore(restore_path)

	def setSess(self, sess):
		self.sess = sess

	def _deepset_encoder(self, set_tensor, scope_name):
		# input shape: (trans_num, set_size, item_dim)
		# NN, sum pooling, NN
		input_shape_vals = set_tensor.get_shape().as_list()
		trans_num = input_shape_vals[0]
		set_size = input_shape_vals[1]
		# print('set_shape:', input_shape_vals) # curr: [None, 5, 1800]
		with tf.name_scope(scope_name):
			with tf.variable_scope(scope_name+'_v'): # for un-share variables defined by get_variable()
				set_h1 = tf.layers.dense(set_tensor, self.ENCODE_HIDDEN_UNITS#NUM_UNITS_FOR_SET_ENCODER  # hyper param
								, activation=TYPE_ACTIVATION_FOR_SET_ENCODER
								, name='set_h1'
								# settings for weights:  TODO: check the selection rules of initializer and regularizer
								, kernel_initializer=self.w_initializer,
								kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
								# settings for biases:
								, bias_initializer=self.b_initializer  # tf.constant_initializer(0.0)
								)

				if self.isTrain:
					set_h1 = tf.layers.dropout(set_h1, rate=self.dropout_rate)
				set_h2 = tf.layers.dense(set_h1, self.ENCODE_OUT_UNITS # NUM_OUT_UNITS_FOR_SET_ENCODER  # hyper param
								, activation=TYPE_ACTIVATION_FOR_SET_ENCODER
								, name='set_h2'
								# settings for weights:  TODO: check the selection rules of initializer and regularizer
								, kernel_initializer=self.w_initializer,
								kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
								# settings for biases:
								, bias_initializer=self.b_initializer  # tf.constant_initializer(0.0)
								)

				if self.isTrain:
					set_h2 = tf.layers.dropout(set_h2, rate=self.dropout_rate)
				#==== sum pooling
				set_avg = tf.layers.average_pooling1d(set_h2, set_size, (1), padding='valid', name='avg_pool')
				set_sum_temp = set_avg * set_size # result of sum pooling, shape: (trans_num, 1, item_dim)
				# print('shapes:', set_sum_temp.get_shape(), tf.squeeze(set_sum_temp).get_shape())  # shapes: (?, 1, 150)
				# set_sum = tf.reshape(set_sum_temp, shape=(input_shape_vals[0], input_shape_vals[2]), name='sum_reshape') # reshape to (trans_num, item_dim)
				# set_sum = tf.squeeze(set_sum_temp) # to remove the 1-sized dim # will cause the shape be <unkonwn>
				# set_sum = set_sum_temp
		return set_sum_temp

	def _encode_state(self):
		#==== Alt.1 if encode one vector for each action (state_curr may have same rows for actions for the same state), directly return
		# same structure, different param to train
		concat_tuple = ()
		if 'curr' in self.trainConfig.deepsetStyle:
			self.state_curr_tensor = tf.placeholder(name='curr', shape=[None, self.dataConfig.topK.value-1, self.trainConfig.NUM_FEATURES], dtype=tf.float32)  # size = [triple_num * trans_curr_num, vec_dim]
			curr_vec = self._deepset_encoder(self.state_curr_tensor, scope_name='curr') # shape: (trans_num, 1, vec_dim)
			concat_tuple = concat_tuple + (curr_vec,)
		if 'cand' in self.trainConfig.deepsetStyle:
			self.state_cand_tensor = tf.placeholder(name='cand', shape=[None, self.dataConfig.MAX_CAND_LENGTH, self.trainConfig.NUM_FEATURES], dtype=tf.float32)  # size = [triple_num * trans_cand_num, vec_dim]
			cand_vec = self._deepset_encoder(self.state_cand_tensor, scope_name='cand')  # not use candidate set to keep it same for all possible actions in one transation
			concat_tuple = concat_tuple + (cand_vec,)
		if 'disH' in self.trainConfig.deepsetStyle:
			self.state_disH_tensor = tf.placeholder(name='disH', shape=[None, self.dataConfig.MAX_CAND_LENGTH, self.trainConfig.NUM_FEATURES], dtype=tf.float32)  # size = [triple_num * trans_disH_num, vec_dim]
			disH_vec = self._deepset_encoder(self.state_disH_tensor, scope_name='disH')
			concat_tuple = concat_tuple + (disH_vec,)

		# set encoder output shape: (trans_num, item_dim)
		# state encoder output shape: (trans_num, 3*item_dim)
		# return tf.concat(self.curr_vec, self.desc_vec, self.disH_vec) # size = [trans_num, 3*vec_dim]
		# print('concated_shape1:', curr_vec.get_shape(), tf.shape(curr_vec), self.trainConfig.NUM_FEATURES)
		# trans_state_vec = tf.concat((curr_vec, cand_vec, disH_vec), axis=2) # size = [trans_num, 3*vec_dim]
		# print('concated_shape2:', trans_state_vec.get_shape()) # (?, 1, 3*NUM_OUT_UNITS_FOR_SET_ENCODER)
		item_num = len(concat_tuple)
		print('item_num:',item_num)
		if item_num == 0:
			return None
		trans_state_vec = tf.concat(concat_tuple, axis=2) if item_num>1 else concat_tuple[0]
		self.state_vec_dim = self.ENCODE_OUT_UNITS * item_num
		print('state_vec_dim:',self.state_vec_dim)

		# =====2 if encode one vector for each transaction, should expand to triple level for each action vector
		# ==== create trans_expand_matrix, i.e. for each trans/state, copy the state_vec cand_num times, to be concated to action_vec (i.e. triple_matrix)
		def cond(idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array):
			return idx < trans_num

		def body(idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array):
			state_vec = state_vec_list[idx, 0, :] # [1, vec_dim]
			trans_triple_num = trans_triple_num_list[idx]  # triple num of trans_idx
			expand_op = tf.ones([trans_triple_num, 1], tf.float32)  # [cand_num, 1]
			expand_matrix = tf.matmul(expand_op, [state_vec])  # size=[cand_num, state_vec_dim] , copy state_vec cand_num times
			# print('shapes:',state_vec.get_shape(), expand_matrix.get_shape(), trans_expand_array.get_shape())
			# trans_expand_matrix.append(expand_matrix)
			# trans_expand_array = trans_expand_array.write(idx, expand_matrix)  # expand_matrix should be same size every time ?
			trans_expand_array = tf.cond(idx>0, lambda: tf.concat([trans_expand_array, expand_matrix], axis=0), lambda: expand_matrix)

			idx = idx + 1
			return idx, trans_num, state_vec_list, trans_triple_num_list, trans_expand_array

		# temp = self.triple_matrix.get_shape().as_list()
		# total_triple_num = temp[0]
		# print('total_triple_num', temp, total_triple_num, self.trans_num)
		# total_triple_num [None, 600] None
		# trans_expand_array = tf.TensorArray(tf.float32, size=self.trans_num)
		trans_expand_array = tf.Variable(tf.zeros([1, self.state_vec_dim]), expected_shape=[None, self.state_vec_dim], trainable=False)  # will cause bug

		# print('sss_trans_expand_array', trans_expand_array.get_shape())
		idx = tf.get_variable('encoder_idx', dtype=tf.int32, shape=[], initializer=tf.zeros_initializer());
		self.idx_out,_,_,_,trans_expand_array = tf.while_loop(cond, body,
					[idx, self.trans_num, trans_state_vec, self.trans_triple_num_list, trans_expand_array]
					, shape_invariants=[idx.get_shape(), self.trans_num.get_shape(), trans_state_vec.get_shape(), self.trans_triple_num_list.get_shape()
										, tf.TensorShape([None, self.state_vec_dim])]
											)  #
		print('arr_shape', trans_expand_array.get_shape())
		return trans_expand_array  # [trans_num * trans_act_num, vec_dim], state for each triple

	def _encode_triple(self, tripleTensor, scope_name):
		# ==== encode triple
		with tf.name_scope(scope_name):
			with tf.variable_scope(scope_name+'_v'): # for un-share variables defined by get_variable()
				triple_h1 = tf.layers.dense(tripleTensor, self.ENCODE_HIDDEN_UNITS
				                     # self.trainConfig.NUM_FEATURES #self.trainConfig.NUM_H1_UNITS  TODO
				                     , activation=self.trainConfig.TYPE_ACTIVATION
				                     , name='h0'
				                     # settings for weights:  TODO: check the selection rules of initializer and regularizer
				                     , kernel_initializer=self.w_initializer,
				                     kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
				                     # settings for biases:
				                     , bias_initializer=self.b_initializer  # tf.constant_initializer(0.0)
				                     )
				if self.isTrain:
					triple_h1 = tf.layers.dropout(triple_h1, rate=self.dropout_rate)
				triple_h2 = tf.layers.dense(triple_h1, self.ENCODE_OUT_UNITS # NUM_OUT_UNITS_FOR_SET_ENCODER  # hyper param
								, activation=TYPE_ACTIVATION_FOR_SET_ENCODER
								, name='set_h2'
								# settings for weights:  TODO: check the selection rules of initializer and regularizer
								, kernel_initializer=self.w_initializer,
								kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
								# settings for biases:
								, bias_initializer=self.b_initializer  # tf.constant_initializer(0.0)
								)

				if self.isTrain:
					triple_h2 = tf.layers.dropout(triple_h2, rate=self.dropout_rate)
		return triple_h2

	def _build_model(self):
		time_start = time.time()
		print('building model....')
		self.w_initializer = tf.contrib.layers.xavier_initializer()
		self.b_initializer = tf.constant_initializer(0.0)


		with tf.name_scope(self.name+'_input-layer'):
			self.triple_vec_list = tf.placeholder(name='candidates', shape=[None, self.trainConfig.NUM_FEATURES], dtype=tf.float32)  # size = [triple_num, 1] = [total_step_num * candidate_num_for_each_step, vec_dim]
			# self.signal_list = tf.placeholder(name='signal_list', shape=[None], dtype=tf.float32)  # size = total_step_num = trans_num
			self.trans_mask = tf.placeholder(name='trans_mask', shape=[None, None], dtype=tf.bool)  # size = [trans_num, all_triple_num]
			self.trans_num = tf.placeholder(name='trans_num', shape=(), dtype=tf.int32)  # scalar

			#==== for deepset
			self.trans_triple_num_list = tf.placeholder(name='trans_triple_num_list', shape=[None], dtype=tf.float32)  # size = total_step_num = trans_num
			self.state_curr_tensor = tf.placeholder(name='temp', shape=[None], dtype=tf.float32)
			self.state_cand_tensor = tf.placeholder(name='temp', shape=[None], dtype=tf.float32)
			self.state_disH_tensor = tf.placeholder(name='temp', shape=[None], dtype=tf.float32)  # for feeddict, will be inited according to self.trainConfig.deepsetStyle in self._encode_state

			#===== only for train
			self.action_mask = tf.placeholder(name='action_mask', shape=[None], dtype=tf.bool)  # size = trans_num, mask out selected triples for training
			self.reward_list = tf.placeholder(name='rewards', shape=[None], dtype=tf.float32)  # size = total_step_num = trans_num

			self.dropout_rate = tf.placeholder(name='dropout_rate', shape=None, dtype=tf.float32)  # for control train/test mode of dropout layer

			# ===== for observe
			self.ndcg = tf.placeholder(name='ndcg', shape=[7], dtype=tf.float32)  # vector: avg [step, ndcg1,2,3,4,5,10]

		with tf.name_scope('encode-layer'):  # encode state
				# ==== encode triples
				triple_encoded_list = self._encode_triple(self.triple_vec_list,'triple') \
					if 'triple' in self.trainConfig.deepsetStyle \
					else self.triple_vec_list
				# ==== encode state
				if self.deepsetStyle=='x':
					print('==========DeepSets is removed!, deepsetStyle=', self.deepsetStyle, )
					self.input_vec = triple_encoded_list
				else:
					self.state_vec_list = self._encode_state()  # [triple_num, vec_dim], each row is the state for this action
					if self.state_vec_list == None: # set not use DeepSets, or nothing to encode
						self.input_vec = triple_encoded_list
					else:
						self.input_vec = tf.concat([triple_encoded_list #self.triple_vec_list
												   , self.state_vec_list],
											   1)  # for each row, new_vec_dim = concat(act_vec, state_vec)


		with tf.name_scope("mlp-layer"): # encode triple+state
			h1 = tf.layers.dense(self.input_vec, self.trainConfig.NUM_H1_UNITS
			                     , activation=self.trainConfig.TYPE_ACTIVATION
			                     , name = 'h1'
			                     # settings for weights:  TODO: check the selection rules of initializer and regularizer
			                     , kernel_initializer=self.w_initializer,kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
			                     # settings for biases:
			                     , bias_initializer=self.b_initializer #tf.constant_initializer(0.0)
			                     )
			# output layer, sigmoid
			if self.isTrain:
				h1 = tf.layers.dropout(h1, rate=self.dropout_rate)  # tf.nn.rnn_cell.DropoutWrapper(h1, input_keep_prob=keep_prob)

			self.h_out = tf.layers.dense(h1, self.trainConfig.NUM_OUT_UNITS, activation=self.trainConfig.TYPE_ACTIVATION_OUT
			                             , name='h_out'
			                             # settings for weights:  TODO: check the selection rules of initializer and regularizer
			                             , kernel_initializer=self.w_initializer,
			                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01)
			                             # settings for biases:
			                             , bias_initializer=self.b_initializer  # tf.constant_initializer(0.0)
			                             )
			if self.isTrain:
				self.h_out = tf.layers.dropout(self.h_out, rate=self.dropout_rate)  # tf.nn.rnn_cell.DropoutWrapper(self.h_out, input_keep_prob=keep_prob)



		# ==== do softmax within each trans
		with tf.name_scope('softmax-layer'):
			h_out_flat = tf.squeeze(self.h_out)  # from shape=(?,1) to shape=(?,), to have same shape as probs to compute tf.concat
			# print('h_out_shape', tf.shape(h_out_flat))

			def cond(idx, trans_num, trans_mask, h_out, probs):
				return idx < trans_num

			def body(idx, trans_num, trans_mask, h_out, probs):
				row = trans_mask[idx, :]
				trans_triple_outs = []
				trans_triple_outs = tf.concat([trans_triple_outs, tf.boolean_mask(h_out, row)], 0)  # shape=(trans_triple_num,) #==== do mask
				# print('trans_triple_outs', trans_triple_outs)
				trans_triple_probs = tf.nn.softmax(trans_triple_outs, axis=0)  # do softmax
				probs = tf.concat([probs, trans_triple_probs], 0)  # probs.append(trans_triple_probs)
				idx = idx + 1
				return idx, trans_num, trans_mask, h_out, probs

			probs = tf.Variable([]);
			idx = tf.get_variable('idx', dtype=tf.int32, shape=[], initializer=tf.zeros_initializer());
			self.loop_result = tf.while_loop(cond, body, [idx, self.trans_num, self.trans_mask, h_out_flat, probs]
			                                 , shape_invariants=[idx.get_shape(), self.trans_num.get_shape(),
			                                                     self.trans_mask.get_shape(), h_out_flat.get_shape(),
			                                                     tf.TensorShape([None])
			                                                     ]
			                                 , parallel_iterations=1)
			self.softmax_out = self.loop_result[4]  # probs
			# print('probs',self.softmax_out.get_shape(), len(self.loop_result))

			self.selected_action_probs = tf.boolean_mask(self.softmax_out, mask=self.action_mask)  # size = trans_num
			# print('shapes:', tf.shape(self.selected_action_probs), tf.shape(self.reward_list))
			# assert tf.shape(self.selected_action_probs)==tf.shape(self.reward_list)


		#==== compute loss
		with tf.name_scope('loss'):
			# use tf.clip_by_value() to avoid zero prob be an illegal input to tf.log
			# use tf.reduce_mean() to mean over all losses, i.e. use mean of losses in a batch to update the model
			neg = -1 if self.use_neg else 1;
			if self.is_try:
				print('build:', self.use_neg, neg)
				self.neg = neg
				self.a = self.reward_list
				self.b = neg * self.reward_list
				self.c = self.selected_action_probs
				self.d = tf.clip_by_value(self.selected_action_probs, 1e-10, 1.0)
				self.e = tf.log(self.d)
				self.f = self.reward_list * tf.log(tf.clip_by_value(self.selected_action_probs, 1e-10, 1.0))
				self.g = neg * self.reward_list * tf.log(tf.clip_by_value(self.selected_action_probs, 1e-10, 1.0))
			self.loss = tf.reduce_mean(neg* self.reward_list * tf.log(tf.clip_by_value(self.selected_action_probs, 1e-10, 1.0))
			                           , name='loss')
		with tf.name_scope('train'):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.trainConfig.LEARNING_RATE)#, epsilon=self.epsilon)

		with tf.name_scope('gradient'):
			self.train_op = self.optimizer.minimize(self.loss)

		#=== for log to tenserboard
		reward_mean = tf.reduce_mean(self.reward_list)
		tf.summary.scalar('reward', reward_mean)
		metricNames = ['step', 'ndcg1', 'ndcg2', 'ndcg3', 'ndcg4', 'ndcg5', 'ndcg10']
		for i in range(7):
			tf.summary.scalar(metricNames[i], self.ndcg[i])
		tf.summary.scalar('loss', self.loss)
		# tf.summary.scalar('cross_entropy', self.centropy)
		# tf.summary.scalar('percision', self.precision)
		tf.summary.histogram('h_out', self.h_out)
		tf.summary.histogram('probs', self.selected_action_probs)
		# tf.summary.merge([tf.summary.histogram("%s-grad" % str(gidx), g[0]) for gidx, g in enumerate(self.grads_and_vars)])  # plot gradients

		# # w1 = tf.get_default_graph().get_tensor_by_name(os.path.split(h1.name)[0]+'/kernel:0')
		tf.summary.histogram('h1', h1)
		for var in tf.trainable_variables():
			tf.summary.histogram(var.name, var)
		self.merged = tf.summary.merge_all()


		print('finished building... time cost:',(time.time()-time_start),'s')

	def update(self, batch_data, useTensorboard=True):
		print('sizes:1',np.shape(batch_data), np.shape(batch_data[0]),np.shape(batch_data[7]),np.shape(batch_data[8]))
		feed_dict = {
			self.triple_vec_list: batch_data[0]
			, self.trans_mask: batch_data[1]
			, self.trans_num: batch_data[2]
			, self.action_mask: batch_data[3]
			, self.reward_list: batch_data[4]
			, self.ndcg: batch_data[5]
			, self.trans_triple_num_list: batch_data[7]
			, self.state_curr_tensor: batch_data[8]
			, self.state_cand_tensor: batch_data[9]
			, self.state_disH_tensor: batch_data[10]
		}
		if useTensorboard:
			loss, _, summ_merged, ndcg = self.sess.run([self.loss, self.train_op, self.merged, self.ndcg], feed_dict=feed_dict)
			return loss, summ_merged, ndcg
		else:
			if self.is_try:
				loss, _,a,b,c,d,e,f,g = self.sess.run([self.loss, self.train_op
													, self.a,self.b,self.c,self.d,self.e,self.f,self.g], feed_dict=feed_dict)
				return loss, self.neg,a,b,c,d,e,f,g #==== for observe
			else:
				loss, _ = self.sess.run([self.loss, self.train_op],feed_dict=feed_dict)
				return loss
	def observe(self, batch_data):
		print('sizes:1', np.shape(batch_data), np.shape(batch_data[0]),np.shape(batch_data[7]),np.shape(batch_data[8]))
		feed_dict = {
			self.triple_vec_list: batch_data[0]
			, self.trans_mask: batch_data[1]
			, self.trans_num: batch_data[2]
			, self.action_mask: batch_data[3]
			, self.reward_list: batch_data[4]
			, self.ndcg: batch_data[5]
			, self.trans_triple_num_list: batch_data[7]
			, self.state_curr_tensor: batch_data[8] # curr
			, self.state_cand_tensor: batch_data[9] # cand
			, self.state_disH_tensor: batch_data[10] # disH
		}
		loss, summ_merged, ndcg = self.sess.run([self.loss, self.merged, self.ndcg], feed_dict=feed_dict)
		return loss, summ_merged, ndcg
	#============ predicts: only for one step
	def _predict_act_probs(self, step_triple_vec_list, step_cand_num, step_state_list_tuple=([],[],[])):
		trans_num = 1
		triple_num, _ = np.shape(step_triple_vec_list)
		trans_mask = np.ones(shape=(1, triple_num), dtype=bool)

		# print('args:\t',self.state_curr_tensor, self.triple_vec_list)
		act_probs = self.sess.run(self.softmax_out
			, feed_dict={self.triple_vec_list:step_triple_vec_list # row: candidate triples, col: vecdim
			, self.trans_mask: trans_mask # col all triples, row: one for each step
			, self.trans_num: trans_num
			, self.trans_triple_num_list: [step_cand_num]
			, self.state_curr_tensor: [step_state_list_tuple[0]]
			, self.state_cand_tensor: [step_state_list_tuple[1]]
			, self.state_disH_tensor: [step_state_list_tuple[2]]
			, self.dropout_rate: self.trainConfig.DROPOUT_RATE_FOR_TEST
					if self.trainConfig.IS_TRAIN else self.trainConfig.DROPOUT_RATE_FOR_TRAIN
											   })
		#====== shape=[episode_num * step_num, max_desc_size, 1] = (1, max_desc_size, 1), e.g. (1, 103, 1)
		# TODO: remove outside dimention,
		act_probs = np.squeeze(act_probs).tolist()  # shape=(max_desc_size, )
		return act_probs

	def _select_act_idx(self, predict_act_probs_list, byGreedy):
		# print('predict_act_probs_list:',np.shape(predict_act_probs_list), len(np.shape(predict_act_probs_list)))
		if (len(np.shape(predict_act_probs_list))==0): # one num,
			selected_idx = 0
		else:
			triple_num = np.shape(predict_act_probs_list)[0]
			if(triple_num==1):
				selected_idx = 0
			elif byGreedy:
				if math.isnan(np.max(predict_act_probs_list)):
					print('action_probs', predict_act_probs_list)
					raise Exception("nan probs! shape="+str(np.shape(predict_act_probs_list))+", probs="+str(predict_act_probs_list))
				selected_idx = predict_act_probs_list.index(max(predict_act_probs_list))
			else:  # by random
				selected_idx = np.random.randint(triple_num)
		return selected_idx


	def predict_act_idx(self, step_triple_vec_list, byGreedy, step_cand_num, step_state_list_tuple=([],[],[])):
		# print('step_triple_vec_list:',np.shape(step_triple_vec_list), len(np.shape(step_triple_vec_list)))
		triple_num = np.shape(step_triple_vec_list)[0]
		if (triple_num == 1 and len(np.shape(step_triple_vec_list))==2):
			selected_idx = 0
		else:
			act_probs = self._predict_act_probs(step_triple_vec_list, step_cand_num, step_state_list_tuple) # type = list
			selected_idx = self._select_act_idx(act_probs, byGreedy)
		return selected_idx

