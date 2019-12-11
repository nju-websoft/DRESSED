'''
@file: State.py
@author: qxliu
@time: 2019/2/24 19:09
'''
import math
import random

from data.constants import *
import util.dbUtil as dbtl

class State:

	def __init__(self, initType
				 #==== for initType = 'origin'
				 , euPair=None
				 , dataConfig=None
				 # , topK=None
				 # , initSumm = None
				 # , cleaned = None
				 , user=None # only when using real user, self.user=None
				 #, feedback_method=None
				 # , initCtrl_gold_num=None #, topK=None
	             , initSumm=None  # should set when 'origin', may different for test and for train
				 #==== for initType = 'copy'
				 , oldState=None, action=None
				 #===== for initType = 'outside', define from outside
				 , step=None, eid=None, curr=None, cand=None, gold=None, dsHistory=None
				 #===== for initType = 'nprf', not use dataConfig
				 , desc=None, topK=None, cleaned=None
				 #====== for try reward
				 # , bad_reward = 0
				 ):
		"""
		can be inited in two ways:
		1. State('origin', euPair, topK, initSumm) : often call by outside functions, to generate s0;
		2. State('copy', oldState=oldState, action=action) : often call by inner functions, to generate the next state by self.makeMove()
		:param initType: 'origin' or 'copy'
		:param euPair: tuple of (ei,uid), e.g.(5,2)
		:param topK: TOPK.top5 or TOPK.top10
		:param initSumm: util.utils.Init_Summ values, e.g. Init_Summ.relin
		# old: e.g. "relin_100", "faces_e_r_100", "linksum_r_wiki_100"
		:param oldState: a State object to be coppied
		:param action: tid to be used to replace disItem in oldState.current
		"""
		self.dataConfig = dataConfig
		if(initType=='origin'):
			self._initByOrigin(euPair, dataConfig.topK, initSumm, dataConfig.randGoldNum, dataConfig.cleaned, user);
			eid = euPair[0]
		elif(initType=="copy"):
			self._initByCopy(oldState, action)
			eid = oldState.euPair[0]
		elif(initType=='outside'):
			topK = dataConfig.topK if dataConfig is not None else topK
			cleaned = dataConfig.cleaned if dataConfig is not None else cleaned
			self._initByOutside(step, eid, curr, cand, gold, dsHistory, user, topK, cleaned)
		else:
			raise Exception('wrong init option! initType=%s, should be "origin", "copy" or "outside"' % initType)

		self.done = not self.notEnd()
		self.reward_strategy = 'rdcg' if dataConfig is None else dataConfig.reward_strategy
		self.curr_rest = list(self.current)
		self._initDislike(eid)


	def _initByOrigin(self, euPair, topK, initSumm, initCtrl_gold_num, cleaned, user):
		"""
		init the first state, i.e. s0
		:param euPair:
		:param topK:
		:param initSumm:
		"""
		# print("in origin")
		# 1. basic elements
		self.t = 0;
		self.euPair = euPair;
		self.topK = topK
		# self.initSumm = initSumm;
		self.cleaned = cleaned;
		# self.feedback_method = feedback_method;
		# self.user = UserPolicy(feedback_method, cleaned)
		self.user = user


		# 2. init data from util (cached)
		self.desc = dbtl.getDescByEid(euPair[0], cleaned)
		self.gold = dbtl.getGoldByEUPair(euPair, topK, cleaned) # if default_GOLD is 'user' else dbtl.getAlgoResult(euPair[0], default_GOLD, topK=topK, cleaned=cleaned)
		# self.current = dbtl.getAlgoResult(euPair[0], initSumm, topK=topK, cleaned=cleaned) if (initSumm != 'random' and initSumm!=Init_Summ.random) \
		# 	if (initSumm != 'random' and initSumm != Init_Summ.random) \
		# 	else self._getRandomInit(self.desc, topK=topK, initCtrl_gold_num=initCtrl_gold_num)
		self.current = dbtl.getAlgoResult(euPair[0], initSumm, topK=topK, cleaned=cleaned)
		# 3. init data by set operation
		self.candidates = np.setdiff1d(self.desc, self.current)
		random.shuffle(self.candidates)  #========= shuffle triples
		# print('candidates:', self.candidates, self.desc, self.current)
		# print('len', len(self.current), len(self.candidates), len(self.desc))
		# print('eu:',euPair,self.current,self.desc,initSumm)
		if(len(self.candidates) !=
				   len(self.desc) - len(self.current)): # check error (if current contains item not in desc)
			raise Exception("wrong desc and current! curr="+str(self.current)+", cand="+str(self.candidates)+", desc="+str(self.desc))

		self.disHistory = np.array([])

	def _initByCopy(self, oldState, action):
		"""
		init a next state of the oldState
		:param oldState:
		:param action:
		"""
		self.dataConfig = oldState.dataConfig
		# print("in copy")
		# 1. basic elements
		self.t = oldState.t + 1;
		self.euPair = oldState.euPair;
		self.topK = oldState.topK;
		self.cleaned = oldState.cleaned
		# self.feedback_method = oldState.feedback_method;
		self.user = oldState.user;
		# self.initSumm = oldState.initSumm; # actually useless after inited current

		# 2. action-independent data
		# self.desc = oldState.desc; # actually useless after inited candidates
		self.gold = oldState.gold;

		# 3. actiion-related data
		# print('old', oldState.current, oldState.disItem)
		currentTemp = np.setdiff1d(oldState.current,[oldState.disItem]) # remove disliked
		# print('new', currentTemp)
		# self.current = np.delete(oldState.current, oldState.disItem)#===========
		self.current = np.append(currentTemp, [action]) # add the action selected by action
		self.candidates = np.setdiff1d(oldState.candidates, [action]) # remove the selected action from candidates
		self.disHistory = np.append(oldState.disHistory, oldState.disItem) # add old disItem to history

		# # 4. simulate dislike # moved to outside
		# self._initDislike(self.euPair[0])
		# # self.disItem = self._simulateDislike(self.feedback_method, cleaned=self.cleaned);
		# # disCandidates = np.setdiff1d(self.current, self.gold)  # select dislike from the in-correct items in curr
		# # self.disItem, self.disProb = self.user.getDislike(self.euPair[0], disCandidates)

	# def _initByOutside(self, step, eid, curr, cand, gold, feedback_method, topK=5, cleaned=True):
	def _initByOutside(self, step, eid, curr, cand, gold, dsHistory, user, topK=5, cleaned=True):
		# print('by outside')
		'''
		for analysis, given any content of state
		:param step:
		:param eid:
		:param curr:
		:param cand:
		:param gold:
		:param feedback_method:
		:param topK:
		:param cleaned:
		:return:
		'''
		self.t = step
		self.euPair = (eid, None);
		self.topK = topK
		self.cleaned = cleaned
		# self.user = UserPolicy(feedback_method, cleaned)
		self.user = user

		self.current = curr
		self.gold = gold
		self.candidates = cand

		self.disHistory = np.array([]) if len(dsHistory)==0 else dsHistory

		# self._initDislike(eid) # moved to outside


	def notEnd(self):
		"""
		check whether has element in candidates (check whether leng(candadites)>0, if >0, notEnd=true)
		:return:
		"""
		if self.gold == None: # when use ureal, maybe None
			return len(self.candidates)>0
		else:
			#==== remove illegal golds
			if len(self.gold)!=len(self.current):
				print('remove eu with wirld gold length! len(gold):',len(self.gold),'!=len(curr):',len(self.current))
				return True
			#=====
			remain_gold = np.setdiff1d(self.gold, self.current)
			if len(remain_gold)>0 and len(self.candidates)>0:
				return True
			else:
				return False;

	def _initDislike(self, eid):
		if self.user!=None:  # user only be None when using real user, self.user=None
			disCandidates = np.setdiff1d(self.current, self.gold)
			val = self.user.getDislike(eid, disCandidates)
			self.disItem = val[0] if isinstance(val, tuple) else val

			if self.disItem>0:
				self.curr_rest.remove(self.disItem)
		else:
			self.disItem=None;
			self.disProb=None;
	def setDislikeFromUReal(self, dislikeTid):
		'''
		only used for real user (when using real user, self.user=None)
		:param dislikeTid:
		:return:
		'''
		self.disItem = dislikeTid
		if self.disItem>0:
			self.curr_rest.remove(self.disItem)
			# print('currest:', self.curr_rest)



	def makeMove(self, action, useRouge=False, withNewState=True):
		"""
		create a new state caused by the action
		:param action:
		:return:
		"""
		if(action not in self.candidates): # punish on illegal action, and stay at this state
			reward = self.dataConfig.bad_reward
			newS = self
			raise Exception('illegal action, act='+str(action)+' not in', self.candidates) # TODO: not allow illegal actions to appear
		else: # legal action
			newS = None
			if withNewState:
				newS = State('copy', oldState=self, action=action)
			nextDis = None if newS is None else newS.disItem
			reward = self._getActReward(action, self.reward_strategy,nextDis)

		done = None if newS is None else not newS.notEnd()
		return (newS, reward, done)
	def getStateBaseline(self):  # TODO: check baseline
		'''
		a fucntion of state, not related to action
		'''
		# rel = 1
		# prob = 0.5
		# position = self.t + 1;
		# dcgItem = (pow(2, rel) - 1) / math.log(position + 1, 2)
		# baseline = dcgItem * prob;

		#==== equivalent implementation
		# baseline = 0.5/math.log((self.t+2),2)
		baseline = 1.0/(len(self.candidates)*math.log((self.t+2),2)) # (1.0/len(self.candidates))/math.log((self.t+2),2)
		return baseline

	def _getActRewardByRouge(self, action):
		'''
		20181217: use Rouge SU4 as rel
		:param action:
		:return:
		'''
		position = self.t + 1;
		rel = getRougeForReward(self.gold, action)  #=====
		dcgItem = (pow(2, rel) - 1) / math.log(position+1, 2)
		return dcgItem if(rel>0) else 0
	def _getActRewardByDislike(self, action,newDis):
		'''
		20190806: if not immediately disliked, rel=1, else 0
		:param action:
		:return:
		'''
		# print('rByDis:',action,newDis,1 if action!=newDis else 0)
		return 1 if action!=newDis else 0


	def _getActRewardByDCG(self, action):
		"""
		immediate reward of the action that created next state, one element of DCG
		:return:
		"""
		position = self.t + 1;
		rel = 1 if(action in self.gold) else 0;
		dcgItem = (pow(2, rel) - 1) / math.log(position+1, 2)
		# print('dcgItem', dcgItem, rel)
		return dcgItem if(rel>0) else self.dataConfig.bad_reward

	def _getActReward(self, action, reward_strategy='rdcg', newDis=None):
		if reward_strategy=='rdcg':
			return self._getActRewardByDCG(action)
		elif reward_strategy=='rdislike':
			return self._getActRewardByDislike(action,newDis)
		elif reward_strategy=='rrouge':
			return self._getActRewardByRouge(action)
		else:
			raise Exception("illegal reward strategy! reward_strategy="+str(reward_strategy))

	def getLikedAction(self, preferGold=True):
		'''
		for supervised pretrain
		:param preferGold: True will select from gold first
		:return:
		'''
		goldCand = []
		if preferGold:
			goldCand = np.setdiff1d(self.gold, self.current) # get the remained gold for candidates

		if (len(goldCand) > 0): # remaining golds in candidates
			return self.user.getLiked(self.euPair[0], goldCand)
		else: # preferGold=False or no gold exists in candidates
			return self.user.getLiked(self.euPair[0], self.candidates)

	def __str__(self):
		return "("+str(list(self.current))+str(list(self.candidates))+")"
	def printInfo(self, withGold=False, withCand=False, withDisHist=False):
		print('eu:', self.euPair
		      , ('golds:' + str(list(self.gold))) if withGold else ''
		      , 'curr:', list(self.current)
		      , 'dis:', self.disItem)
		if withCand:
			print('cand',list(self.candidates))
		if withDisHist:
			print('disH', list(self.disHistory))


def getRougeForReward(gold, action):
	'''
	for change reward
	:param gold:
	:param action:
	:return:
	'''
	print('error in State.getRougeForReward(), removed function!')
	return None





# if __name__ == '__main__':
	# euPair = (5, 2)
	# topK = TOPK.top5
	# initSumm = Init_Summ.relin#"relin_100"
	# s = State('origin', euPair, topK, initSumm, cleaned=True, feedback_method='order')
	# print('gold', s.gold)
	# while(s.notEnd()):
	#     action = random.choice(s.candidates)
	#     (s1, r0) = s.makeMove(action)
	#     # print(s1.t, r0, s1.current, s1.gold, s1.candidates, s1.disItem, s1.disHistory)
	#     # print(s1.t, r0, s1.current, s1.disItem, s1.disHistory)
	#     print(s1)
	#     s = s1
