'''
@file: State.py
@author: qxliu
@time: 2019/2/24 19:09
'''
import random
import numpy as np
import math

class State:

	def __init__(self, initType
				 #==== for initType = 'origin'
				 , euPair=None
				 , dataConfig=None
				 , user=None # only when using real user, self.user=None
				 #==== for initType = 'copy'
				 , oldState=None, action=None
				 #===== for initType = 'outside', define from outside
				 , step=None, eid=None, curr=None, cand=None, gold=None, dsHistory=None, topK=None, cleaned=None
				 ):
		"""
		can be inited in two ways:
		1. State('origin', euPair, topK, initSumm) : often call by outside functions, to generate s0;
		2. State('copy', oldState=oldState, action=action) : often call by inner functions, to generate the next state by self.makeMove()
		:param initType: 'origin' or 'copy'
		:param euPair: tuple of (ei,uid), e.g.(5,2)
		:param topK: TOPK.top5 or TOPK.top10
		# :param initSumm: util.utils.Init_Summ values, e.g. Init_Summ.relin
		# old: e.g. "relin_100", "faces_e_r_100", "linksum_r_wiki_100"
		:param oldState: a State object to be coppied
		:param action: tid to be used to replace disItem in oldState.current
		"""
		if(initType=='origin'):
			self.dataConfig = dataConfig
			self._initByOrigin(euPair, dataConfig.topK, dataConfig.cleaned, user);
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
		self.reward_strategy = self.dataConfig.reward_strategy
		self.bad_reward = self.dataConfig.bad_reward
		self.curr_rest = list(self.current)
		self._initDislike(eid)


	def _initByOrigin(self, euPair, topK, cleaned, user):
		"""
		init the first state, i.e. s0
		:param euPair:
		:param topK:
		"""
		# print("in origin")
		# print("in origin", self.dataConfig==None)
		# 1. basic elements
		self.t = 0;
		self.euPair = euPair;
		self.topK = topK
		self.cleaned = cleaned;
		self.user = user

		# 2. init container from util (cached)
		self.desc = self.dataConfig.get_desc_by_eid(euPair[0])
		self.gold = self.dataConfig.get_gold_by_eupair(euPair)
		self.current = self.dataConfig.get_inisumm_by_eid(euPair[0])
		# 3. init container by set operation
		self.candidates = np.setdiff1d(self.desc, self.current)
		random.shuffle(self.candidates)  #========= shuffle triples
		if(len(self.candidates) != len(self.desc) - len(self.current)): # check error (if current contains item not in desc)
			raise Exception("wrong desc and current! curr="+str(self.current)+", cand="+str(self.candidates)+", desc="+str(self.desc))

		self.disHistory = np.array([])

	def _initByCopy(self, oldState, action):
		"""
		init a next state of the oldState
		:param oldState:
		:param action:
		"""
		self.dataConfig = oldState.dataConfig
		# print("in copy", oldState.dataConfig==None, self.dataConfig==None)
		# 1. basic elements
		self.t = oldState.t + 1;
		self.euPair = oldState.euPair;
		self.topK = oldState.topK;
		self.cleaned = oldState.cleaned
		self.user = oldState.user;

		# 2. action-independent container
		self.gold = oldState.gold;

		# 3. actiion-related container
		currentTemp = np.setdiff1d(oldState.current,[oldState.disItem]) # remove disliked
		self.current = np.append(currentTemp, [action]) # add the action selected by action
		self.candidates = np.setdiff1d(oldState.candidates, [action]) # remove the selected action from candidates
		self.disHistory = np.append(oldState.disHistory, oldState.disItem) # add old disItem to history

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
		self.user = user

		self.current = curr
		self.gold = gold
		self.candidates = cand

		self.disHistory = np.array([]) if len(dsHistory)==0 else dsHistory

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
			val = self.user.getDislike(disCandidates)
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



	def makeMove(self, action, withNewState=True):
		"""
		create a new state caused by the action
		:param action:
		:return:
		"""
		if(action not in self.candidates): # punish on illegal action, and stay at this state
			reward = self.bad_reward
			newS = self
			raise Exception('illegal action, act='+str(action)+' not in', self.candidates) # TODO: not allow illegal actions to appear
		else: # legal action
			newS = None
			if withNewState:
				newS = State('copy', oldState=self, action=action)
			# nextDis = None if newS is None else newS.disItem
			reward = self._getActReward(action, self.reward_strategy)#,nextDis)

		done = None if newS is None else not newS.notEnd()
		return (newS, reward, done)

	def _getActRewardByDCG(self, action):
		"""
		immediate reward of the action that created next state, one element of DCG
		:return:
		"""
		position = self.t + 1;
		rel = 1 if(action in self.gold) else 0;
		dcgItem = (pow(2, rel) - 1) / math.log(position+1, 2)
		# print('dcgItem', dcgItem, rel)
		return dcgItem if(rel>0) else self.bad_reward

	def _getActReward(self, action, reward_strategy='rdcg'):
		if reward_strategy=='rdcg':
			return self._getActRewardByDCG(action)
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

