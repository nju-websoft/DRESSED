import math
import numpy as np

'''
20190301: update self._getIDCG() to use cache
===========
UseCorrNum： for computing IDCG, assumption about the total num of ground truths (all correct items)
True -- assume all correct items are in the vec (fit for isumm scenario);
False -- assume corrNum>>pos, there remians correct items not in the vec (fit for IR scenario);
'''
DEFAULT_UseCorrNum = True
DEFAULT_BETA = 0.6  # for NDCF
ZERO_THRESHOLD = 1e-6


class Evaluator:

	def eval_for_acts(self, gold, act_tids):
		part = [(1 if val in gold else 0) for val in act_tids]
		step = self._getStep(part)
		partNDCGpList = []
		for pos in [1, 2, 3, 4, 5, 10]:
			partNDCGp = self._getNDCG(part, pos)
			partNDCGpList.append(partNDCGp)

		# items in reuld vec
		result = [step];
		result.extend(partNDCGpList) #make list

		return result

	def eval_for_dev(self, gold, act_tids):
		part = [(1 if tid in gold else 0) for tid in act_tids]  # 0,1-list of acts by gold
		partNDCG = self._getNDCG(part, len(part)) # without cut-off
		return partNDCG

	def eval_for_test(self, gold, act_tids, summ_signal_list):
		part = [(1 if tid in gold else 0) for tid in act_tids]  # 0,1-list of acts by gold
		step = self._getStep(part)
		partNDCG = self._getNDCG(part, len(part)) # without cut-off
		assert partNDCG > 0 # not initially good
		assert len(summ_signal_list)>0
		partNDCF = self._getNDCF(summ_signal_list[1:], len(gold)) # use full gold for NDCF
		return step, partNDCG, partNDCF


	def _getNDCF(self, part_summ_signal_list, gold_size, beta = DEFAULT_BETA):
		'''
		:param part_summ_signal_list: list of k-sized 0/1 arrays, item i is a summ-signal at step i;
			e.g. [[1,0,0,1,0],[1,0,1,1,0],[1,0,1,1,1],[1,1,1,1,1]]
		:param gold_size: 5 or 10, for recall
		:param beta: 0.6, (0.6^10<0.01)
		:return:
		'''
		sum_norminator = 0
		sum_denorminator = 0
		for step_id, summ_signal in enumerate(part_summ_signal_list):
			correct = float(sum(summ_signal))
			prec = correct/len(summ_signal)
			recall = correct/gold_size
			f_denorminator = (prec+recall)
			fmeasure = 2*prec*recall/f_denorminator if f_denorminator>ZERO_THRESHOLD else 0
			sum_norminator += fmeasure * math.pow(beta,step_id)
			sum_denorminator += math.pow(beta, step_id)
		return sum_norminator/sum_denorminator

	def eval(self, topK, binaryVecFull=None, base=None, part=None):
		base = binaryVecFull[0:topK.value] if ((base is None) and (binaryVecFull is not None)) else base;
		part = binaryVecFull[topK.value:] if ((part is None) and (binaryVecFull is not None)) else part;
		# print('binaryVec', binaryVecFull, base, part)
		step = self._getStep(part)
		corrTotalNum = sum(binaryVecFull)
		baseDCG = self._getDCG(base) if base is not None else -1
		# baseIDCG = self._getIDCG(base, useCorrNum=False) if base is not None else -1
		baseIDCG = self._getIDCG(base, pos=corrTotalNum, useCorrNum=False) if base is not None else -1
		# print('bbbase:',baseDCG, base, baseIDCG, sum(binaryVecFull))
		# baseNDCG = self._getNDCG(base, useCorrNum=False) if base is not None else -1
		baseNDCG = self._getNDCG(base, pos=corrTotalNum, useCorrNum=False) if base is not None else -1
		partDCG = self._getDCG(part)
		partIDCG = self._getIDCG(part)
		partNDCG = self._getNDCG(part)
		partNDCGpList = []
		for pos in [1,2,3,4,5,10]:
			partNDCGp = self._getNDCG(part, pos)
			partNDCGpList.append(partNDCGp)
		# items in reuld vec
		result = [step];
		result.extend(partNDCGpList) #make list
		result.extend([partDCG,partIDCG,partNDCG])
		result.extend([baseDCG,baseIDCG,baseNDCG])
		return result

	def _getStep(self, binaryVec):
		step=-1;
		for i in range(len(binaryVec)):
			step = i+1 if binaryVec[i]==1 else step; # not count in the following zeros
		step = 0 if step < 0 else step  #=====
		return step;

	def _getDCGAtT(self, binary, t):
		numerator = 1 if binary == 1 else 0;
		denominator = math.log(t + 1, 2)
		dcgt = numerator / denominator;
		return dcgt
	def _getDCG(self, binaryVec, pos=None):
		if pos is None:
			pos = len(binaryVec)
		else:
			pos = min(pos, len(binaryVec))
		dcg = 0
		# print('bi',binaryVec,pos,list(range(pos)))
		for i in range(pos):
			t = i+1;
			dcg += self._getDCGAtT(binaryVec[i], t)
		return dcg;
	def _getIDCG(self, binaryVec=None, pos=None, useCorrNum=DEFAULT_UseCorrNum):
		'''

		:param binaryVec:
		:param pos:
		:param useCorrNum: True -- assume all correct items are in the vec (fit for isumm scenario);
						False -- assume corrNum>>pos, there remians correct items not in the vec (fit for IR scenario)
		:return:
		'''
		if binaryVec is not None:
			if pos is None:
				pos = len(binaryVec)
			else:
				pos = min(pos, len(binaryVec))
		#=== 1. get corrCount
		if useCorrNum and (binaryVec is not None):
			corrCount = 0;
			for item in binaryVec:
				corrCount += 1 if item==1 else 0;
			pos = min(pos, corrCount)
		#=== 2. get idcg
		dcg = 0;
		# print("corrCount", corrCount)
		for i in range(pos):  # range(corrCount):
			t = i + 1;
			numerator = 1;
			denominator = math.log(t + 1, 2)
			dcgt = numerator / denominator;
			# print("idcgt",pos, t,denominator,dcgt)
			dcg += dcgt
		return dcg;
	def _getNDCG(self, binaryVec, pos=None, useCorrNum=DEFAULT_UseCorrNum):
		if pos is None:
			pos = len(binaryVec)
		else:
			pos = min(pos, len(binaryVec))
		dcg = self._getDCG(binaryVec, pos)
		idcg = self._getIDCG(binaryVec, pos, useCorrNum=useCorrNum)
		ndcg = dcg/idcg if idcg!=0 else 0

		# print("ndcg",pos,dcg,idcg,ndcg)
		return ndcg;

	def _getNDCGListForSteps(self, binaryVec, posOrderedAscList):
		ndcgList = []
		maxPos = posOrderedAscList[-1]
		dcg = 0
		for i in range(maxPos):
			pos = i+1
			dcg += self._getDCGAtT(binaryVec[i], pos)
			idcg = self._getIDCG(binaryVec, pos)
			ndcg = dcg/idcg if idcg!=0 else 0
			if pos in posOrderedAscList:
				ndcgList.append(ndcg)
		return ndcgList

	def getMetricNames(self):
		return ['step','ndcg1','ndcg2','ndcg3','ndcg4','ndcg5','ndcg10','partDCG','partIDCG','partNDCG','baseDCG','baseIDCG','baseNDCG']

def getIDCG(goldNum):
	pos = goldNum
	dcg = 0;
	for i in range(pos):  # range(corrCount):
		t = i + 1;
		numerator = 1;
		denominator = math.log(t + 1, 2)
		dcgt = numerator / denominator;
		# print("idcgt",pos, t,denominator,dcgt)
		dcg += dcgt
	return dcg;


def getPRF(items, golds):
	prec_list = []
	recall_list = []
	fmeasure_list = []
	for gold in golds:
		corr = set(items).intersection(gold)
		prec = len(corr)/len(items)
		recall = len(corr)/len(gold)
		fmeasure = 2*prec*recall/(prec+recall) if (prec+recall)!=0 else 0
		# print('prf',gold, corr, prec, recall, fmeasure)
		prec_list.append(prec)
		recall_list.append(recall)
		fmeasure_list.append(fmeasure)
	return np.mean(prec_list), np.mean(recall_list), np.mean(fmeasure_list)


#
# if __name__ == '__main__':
	# # vec = [1,0,0,1,0, 1,0,1,1,1,0,0,0,0,0,0,1]
	# # vec = [1,1,0,0,0, 1,0,1,1,1,0,0,0,0,0,0,1]
	# vec = [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
	# eval = Evaluator().eval(binaryVecFull=vec, topK=TOPK.top5)
	# print(eval[-3])
	# eval._
	# result1 = Evaluator().eval(binaryVecFull=vec, topK=TOPK.top5)
	# print("result",result1)
	# result2 = Evaluator().eval(binaryVecFull=vec, topK=TOPK.top5)
	# result3 = np.array(result1)+np.array(result2)
	# print(list(result3))
	# avgResult = np.dot(result3, 1/2)
	# print(list(avgResult))
	# print(Evaluator().getMetricNames())

	# #==== check idcg
	# # eval = Evaluator()
	# # for pos in range(21):
	# #     log = math.log(pos+1, 2)
	# #     denominator = 1/log if log>0 else 0
	# #     idcg = eval._getIDCG(pos=pos)
	# #     print("idcg",pos, idcg, log, denominator)
	#
	#
	# #==== print
	# maxNum = math.pow(2,5)#(2,8);
	# eval = Evaluator()
	# for i in range(int(maxNum)):
	#     bitVec = [int(x) for x in '{:08b}'.format(i)]
	#     bitVec = bitVec[3:8]
	#     dcg = eval._getDCG(bitVec)
	#     ndcg = eval._getNDCG(bitVec,useCorrNum=False)
	#     idcg = eval._getIDCG(bitVec,useCorrNum=False)
	#     print(i, dcg, ndcg, idcg, bitVec)

	# #===== check getPRF()
	# items = [1,2,3,4,5]
	# golds = [[1,2,3,4,5],[1,6,6,6,6],[1,2,6,6,6],[6,3,4,5,6],[6,4,3,5,2],[6,6,6,6,6]]
	# print(getPRF(items, golds))
	#
	# eva = Evaluator()
	# gold = [24, 36, 38, 88, 114]
	# tids = [36, 24, 66, 45, 37, 117, 100, 35, 113, 98, 72, 89, 65, 106, 41, 52, 53, 57, 44, 50, 61, 64, 73, 104, 119,
	#         81, 39, 93, 40, 103, 47, 101, 126, 80, 125, 83, 60, 87, 38, 109, 49, 85, 75, 107, 62, 92, 102, 90, 124, 48,
	#         68, 86, 111, 51, 105, 74, 97, 82, 70, 69, 116, 59, 76, 99, 122, 114, 54, 94, 120, 79, 110, 84, 67, 108, 88]
	# # for round_idx in range(5):
	# # 	timeStart = time.time()
	# # 	for i in range(10000): # time: 1.1080634593963623
	# # 		result = eva.eval_for_acts(gold, tids)
	# # 	print('time:',time.time()-timeStart)
	# # 	print(result)
	# result = eva.eval_for_dev(gold, tids)
	# print(result) # 0.7270697750329694
	# pass

