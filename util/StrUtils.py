'''
@file: StrUtils.py
@author: qxliu
@time: 2019/2/24 22:38
'''
import numpy as np


import glob
import os


def _euPair2DigitStr(eid,uid):  # euPair to qid, e.g. euPair=(30,5), euDigitStr='03005'
	eidStr = str(eid)
	zeroPrefix = (3-len(eidStr))*'0'  # 3 digits for eid, (because max(eid)=140, max(len(str(eid)))=3)
	eidStr = zeroPrefix + eidStr

	uidStr = str(uid)
	zeroPrefix = (2-len(uidStr))*'0'  # 2 digits for uid, (because max(uid)=24, max(len(str(uid)))=2)
	uidStr = zeroPrefix + uidStr

	return eidStr + uidStr

def _digitStr2euPair(euDigitStr):  # qid to euPair
	eidStr = euDigitStr[:3]
	eid = int(eidStr)
	uidStr = euDigitStr[3:]
	uid = int(uidStr)

	return eid, uid


def retain_file(path, tagger, retain_file, retain_by_endwith=True):
	'''
	:param path: dir
	:param tagger: files to remove
	:param retain_file: file to
	:return:
	'''
	files = glob.glob(os.path.join(path, tagger + '*'))
	# print('pattern:',os.path.join(path, tagger + '*'))
	# print('fnum:',files)
	for _file in files:
		dirname, filename = os.path.split(_file)
		# print('_file:',_file, not _file.endswith(retain_file), not filename.startswith(retain_file))
		print('\t',filename)
		if retain_by_endwith and not filename.endswith(retain_file): # if retain_by_endwith=True, only retain the file with given suffix
			os.remove(_file)
		if not retain_by_endwith and not filename.startswith(retain_file):# if retain_by_endwith=False, only retain the file with given prefix
			os.remove(_file)

#======= 5. util methods
def val2Binary(valList, gold):
	valBinaryList = [(1 if val in gold else 0) for val in valList]
	return valBinaryList

# from sklearn.preprocessing import normalize
def normalizeVec(vec):
	'''
	normalize a vector, i.e. make ||vec||_2 = 1, where ||vec||_2 = [sum_i (i^2)]^(1/2)
	:param vec:
	:return:
	'''
	norm2 = np.linalg.norm(vec, ord=2)
	return vec / norm2


def runTimer(batch_time, batch_num, epoch_num):
	epoch_time = batch_num * batch_time;
	fold_time = epoch_num * epoch_time;
	print('batch time:',timeConvert(batch_time))
	print('epoch time:',timeConvert(epoch_time))
	print('fold time:',timeConvert(fold_time))

def timeConvert(timeVal, metric='s'):
	allMetrics = ['ms','s','min','h','d','w']
	allDividers = [1000,60,60,24,7]
	curr_idx = allMetrics.index(metric)
	curr_val = float(timeVal)
	for i in range(curr_idx, len(allMetrics)):
		divider = allDividers[i]
		if timeVal>divider:
			timeVal = timeVal/divider
			curr_idx = i+1
		else:
			break
	return timeVal,allMetrics[curr_idx]

def getLastMaskedIdx(mask_list):
	mask_item = np.ones(np.shape(mask_list[0])).tolist() # should .tolist() (if is np.array() will cause exception)
	return getLastIdx(mask_list, mask_item)

def getLastIdx(targetList, val):
	inv_idx = targetList[::-1].index(val)
	return (len(targetList)-1)- inv_idx

def convertStr2IntList(listStr):
	tmp = listStr.split(', ')  # has empty item
	tmp2 = [int(x) for x in tmp if x!=''] # remove empty items and convert to int
	# print('tmp', tmp, tmp2)
	return tmp2

def convertStr2FloatList(listStr):
	tmp = listStr.replace('[','').replace(']','').split(', ')  # has empty item
	tmp2 = [float(x) for x in tmp if x!=''] # remove empty items and convert to int
	# print('tmp', tmp, tmp2)
	return tmp2

def getPaddedList(origin_list, max_size
				  , item_shape=None # only used when len(origin_list)==0
                  , is_mask = False
				  ):
	paddedList = None
	origin_size = len(origin_list) if origin_list is not None else 0
	final_shape = (max_size, ) + item_shape
	if origin_size == 0: # all padding, when origin_list=[] or None
		if is_mask:
			paddedList = np.zeros(final_shape, dtype=bool).tolist()
		else:
			paddedList = np.zeros(final_shape).tolist()
	elif origin_size > max_size: # do cut
		paddedList = origin_list[0:max_size]
	else:
		remain_size = max_size - origin_size
		padding_shape = (remain_size, ) + item_shape
		if is_mask:
			padding = np.zeros(padding_shape, dtype=bool).tolist()
		else:
			padding = np.zeros(padding_shape).tolist()
		paddedList = list(origin_list)
		paddedList.extend(padding)
	return paddedList

if __name__ == '__main__':
	# str = ", 19, 17, 1, 16, 22, "
	# convertStr2IntList(str)
	# sstr = '[2.876855432987213e-05, -5.206714073816935e-05, 0.05689753840366999]'
	# val = convertStr2FloatList(sstr)
	# print(val)

	# #==== getPaddedList()
	# x = [[2,3,4],[4,5,6]]
	# item_shape = np.shape(x[0])
	# max_size = 5
	# y = getPaddedList(x, max_size, item_shape)
	# print(np.shape(y))
	# print(y)
	#
	# x = None
	# item_shape = (3,)
	# max_size = 5
	# y = getPaddedList(x, max_size, item_shape)
	# print(np.shape(y))
	# print(y)
	#
	# x = [1,2,3]
	# item_shape = ()
	# max_size = 5
	# y = getPaddedList(x, max_size, item_shape)
	# print(np.shape(y))
	# print(y)
	# #=============

	# time_val = 187.6748457
	# metric = 's'
	# out = timeConvert(time_val, metric)
	# print(out)
	# batch_time = 8.72049856185913
	# batch_num = 12
	# epoch_num = 300

	batch_time = 250#10.0
	batch_num = 1#600/25.0
	epoch_num = 300
	out = runTimer(batch_time,batch_num,epoch_num)
	print(out)
