'''
train for model selection
20190225: add pad_ (use padding)
201903xx: add origin_ (not use padding)
20190327: add ds_ (for deepset)
TODO: add att_ (use attention)
20191117: move from eSummRLPy2 to eSummRLPy2_forFACES
@file: train_for_eval.py
@author: qxliu
@time: 2019/2/25 11:30
'''
import argparse
import logging
import sys

import ies_train.xargs as xargs
import ies_umodel.policy_user as puser
from container.DataConfig import *
from container.Env import *
from ies_model.ds_eupairds import *
from ies_model.ds_modeling import *
from ies_umodel.policy_user import *

from container.TrainConfig import *
from util.load_files import OUT_LOG_DIR

# RESULT_LIST_IDX = 6 # idx of eval_results_list

def prepareParams(argsList=None
				  , forUreal=False
				  , urealLoggerFileName=None
				  , oldlogger=None):
	'''

	:param argsList: list like sys.argv[1:] (e.g. 'command line params'.split(' '))
	:return:
	'''
	signature = time.time()

	#==== settings
	#========= input args
	parser = argparse.ArgumentParser(description='train_for_eval')
	xargs.add_data_options(parser) # define args
	xargs.add_train_options(parser)
	xargs.add_test_options(parser)

	opt = parser.parse_args() if argsList==None else parser.parse_args(argsList)

	# model_save_path_full = default_LOG_DIR + opt.rlName + '_' + opt.dataSplit + '_' + str(
	# 	opt.ds_name.name) + '_' + opt.run_name + '_' + str(signature)
	model_save_path_full = os.path.join(OUT_LOG_DIR, '{}_{}_{}_{}_{}'.format(opt.rlName, opt.dataSplit, opt.ds_name.name, opt.run_name, signature))
	#========= logger
	if oldlogger is not None: # pass logger from outside
		logger = oldlogger
	else: # create new logger
		if forUreal:
			logger_file_path_full = urealLoggerFileName
		else: # for trainTest
			logger_file_path_full = model_save_path_full + '.txt'

		for handler in logging.root.handlers[:]:
			logging.root.removeHandler(handler)
		logging.basicConfig(filename=logger_file_path_full #model_save_path_full + '.txt'
		                    , filemode='a'
		                    , format='%(asctime)s %(message)s'
		                    , datefmt='%H:%M:%S'
		                    , level=logging.DEBUG
		                    )
		logger = logging.getLogger(__name__)
	logger.info('My PID is {0}'.format(os.getpid()))
	logger.info('args:'+str(sys.argv))
	logger.info('allopts:'+str(opt))

	#========== init configs
	dataConfig = DataConfig(
		ds_name=opt.ds_name
		, dataSplit = opt.dataSplit
		, foldId = opt.foldId
		, byUser = opt.byUser
		, forUreal = forUreal
		, gamma=opt.gamma
		, gamma_out=opt.gamma_out
	)
	trainConfig = TrainConfig(
		num_repeats = opt.num_repeats
		, do_rev_folds = opt.do_rev_folds
		, batch_size = opt.batch_size
		, batch_num = opt.batch_num
		, epoch_num = opt.epoch_num
		, dev_step = opt.dev_step
		, learning_rate = opt.learning_rate
		, name = opt.run_name
		, signature = signature
		#====== for deepset
		, deepset_style = opt.deepset_style
	)
	env = Env(dataConfig=dataConfig)
	trainConfig.setModelFileName(model_save_path_full)

	umodel = UserPolicy('learned', dataConfig.ds_name, dataConfig=dataConfig)  # 'z'
	actmodel = PolicyNN_ds(dataConfig, trainConfig, num_to_save=opt.num_to_save, restore_path=opt.restore_path)
	dataClass = EUPairDataset_ds

	# #======== prepare container
	datasetList = None  # none for Ureal
	if (dataConfig.dataSplit =='e5fold'):
		if dataConfig.foldId is not None and dataConfig.foldId >= 0:  # ==== only one fold
			print('datasplit:', dataConfig.dataSplit, ', has foldId=', dataConfig.foldId)
			train, dev, test = dataConfig.get_train_valid_test(dataConfig.foldId)  #=====
			ds_train = dataClass(train, env, dataConfig, deepset_style=trainConfig.deepset_style,
			                     name='trainf' + str(dataConfig.foldId), user=umodel, is_train=True,
			                     doShuffle=True)#, useAllLabels=useAllLabels)
			ds_dev = dataClass(dev, env, dataConfig, deepset_style=trainConfig.deepset_style,
								name='devf' + str(dataConfig.foldId),
								user=umodel, is_train=False,
								doShuffle=False)#, useAllLabels=useAllLabels)
			ds_test = dataClass(test, env, dataConfig, deepset_style=trainConfig.deepset_style,
			                    name='testf' + str(dataConfig.foldId), user=umodel, is_train=False,
			                    doShuffle=False)#, useAllLabels=useAllLabels)
			datasetList = [(ds_train, ds_dev, ds_test)]
		else:  # === all 5 fold
			# print('datasplit:', dataConfig.dataSplit, ', no foldId=', dataConfig.foldId)
			train_list, valid_list, test_list = dataConfig.get_train_valid_test() #=====
			datasetList = []
			for fold_idx in range(5):
				ds_train = dataClass(train_list[fold_idx], env, dataConfig, deepset_style=trainConfig.deepset_style,
				                     name='trainf' + str(fold_idx),
				                     user=umodel, is_train=True,
				                     doShuffle=True)#, useAllLabels=useAllLabels)
				ds_dev = dataClass(valid_list[fold_idx], env, dataConfig, deepset_style=trainConfig.deepset_style,
								name='devf' + str(fold_idx),
								user=umodel, is_train=False,
								doShuffle=False)#, useAllLabels=useAllLabels)
				ds_test = dataClass(test_list[fold_idx], env, dataConfig, deepset_style=trainConfig.deepset_style,
				                    name='testf' + str(fold_idx),
				                    user=umodel, is_train=False,
				                    doShuffle=False)#, useAllLabels=useAllLabels)
				datasetList.append((ds_train, ds_dev, ds_test))

	logger.info('dataConfig:'+str(dataConfig))
	logger.info('trainConfig:'+str(trainConfig))
	logger.info('userModel:'+str(puser._getModelFilePath(dataConfig.ds_name.name, umodel.modelDir, forModel=True)))

	return trainConfig, datasetList, actmodel, logger,dataConfig

def _initSummBatch(trainConfig, datasetList, repeat_idx, withSumm=True):
	summwList = []  # list of tuples, each tuple for one fold
	batchSizeList = []  # list of tuples, each tuple for one fold
	batchNumList = []  # list of tuples, each tuple for one fold
	for fold_idx, datasetTuple in enumerate(datasetList):  # for folds
		summwTuple = []
		batchSizeTuple = []
		batchNumTuple = []
		for ds_idx, dataset in enumerate(datasetTuple):  # train/dev/test
			if withSumm:
				# trainConfig.MODEL_FILE_PATH_FULL + '_r' + str(repeat_idx) + '-f' + str(fold_idx) + '-d' + str(ds_idx)+str(dataset.name)
				summ_path = '{}_r{}-f{}-d{}{}'.format(trainConfig.MODEL_FILE_PATH_FULL, repeat_idx, fold_idx, ds_idx, dataset.name, )
				summwTuple.append(tf.summary.FileWriter(
					summ_path, tf.get_default_graph()))
			batch_size = int(np.min([dataset.size, trainConfig.BATCH_SIZE])) # all use batch (important!) # removed: if ds_idx == 0 else dataset.size  # only split to batches for train container(i.e. ds_idx==0)
			batchSizeTuple.append(batch_size)

			max_batch_num = int(np.round(dataset.size / float(batch_size)))
			batch_num = int(np.min([max_batch_num, trainConfig.BATCH_NUM])) # all use batch (important!) # removed: if ds_idx == 0 else 1  # only split to batches for train container(i.e. ds_idx==0)
			batchNumTuple.append(batch_num)

		summwList.append(summwTuple)
		batchSizeList.append(batchSizeTuple)
		batchNumList.append(batchNumTuple)
	return summwList, batchSizeList, batchNumList






# def _train_one_model(repeat_idx, fold_idx, model, trainConfig, datasetTuple, summwTuple, batchSizeTuple, batchNumTuple, logger):
# 	min_dev_loss = 100  #=== for early stop
#
# 	with tf.Session() as sess:
# 		model.setSess(sess)
# 		model.init.run()
# 		model_save_path = None
#
# 		for epoch_idx in range(trainConfig.EPOCH_NUM):
# 			epoch_timeStart = time.time()
# 			epoch_avg_results = dict() # <dataset.name, avg_result>
# 			epoch_ds_timeCost = []
# 			model_save_path = trainConfig.MODEL_FILE_PATH_FULL + '_r' + str(repeat_idx) + '-f' + str(fold_idx) + '-e' + str(epoch_idx)  # add epoch_idx for early stop
# 			saved = False
# 			print('epoch_idx:', epoch_idx)
#
# 			for ds_idx, dataset in enumerate(datasetTuple):  # train/dev/test
# 				ds_result_list = []  # list of ndcg_lists
# 				ds_timeStart = time.time()
#
# 				if (epoch_idx+1) % trainConfig.DEV_STEP != 0 and ds_idx > 0 \
# 						and epoch_idx != trainConfig.EPOCH_NUM - 1:  # only run dev-test(ds_idx>0) on specific epoches to speed up: 1.%trainConfig.DEV_STEP , 2. last epoch
# 					continue
#
# 				dataset.new_epoch() #===== important
#
# 				batch_num = batchNumTuple[ds_idx]
# 				batch_size = batchSizeTuple[ds_idx]
# 				epoch_loss_sum=None #=== 20190328 add
# 				batch_count = 0
# 				for batch_idx in range(batch_num):
# 					batch_data = dataset.get_batch(model, batch_size)
# 					if batch_data is not None:
# 						batch_count+=1
# 						ds_result_list.extend(batch_data[RESULT_LIST_IDX]) # list of metrics for eids, list of metric-list
# 						if ds_idx == 0:  # === train
# 							loss, summ_merged, ndcg = model.update(batch_data)
# 							summwTuple[ds_idx].add_summary(summ_merged, epoch_idx)
# 						else:  # ===== draw summ or early stop (for dev or test, only have one batch)
# 							loss, summ_merged, ndcg = model.observe(batch_data)
# 							summwTuple[ds_idx].add_summary(summ_merged, epoch_idx)
# 							epoch_loss_sum = loss if epoch_loss_sum==None else epoch_loss_sum+loss
# 					#==== out if batch_data
# 				#==== out for batch_idx
# 				#====== 20190328 add: to save (only influence dsSplit=all, and deepset+batch_size32 when batch_size(test)<size(test))
# 				if ds_idx == 1 and epoch_loss_sum!=None:
# 					epoch_loss = epoch_loss_sum / batch_count
# 					min_dev_loss = epoch_loss
# 					model.saver.save(sess, save_path=model_save_path)
# 					saved = True
# 					logger.info('save model:' + str(epoch_idx) + ', batch_count:' + str(batch_count) + ', loss:' + str(epoch_loss) + ', path:' + str(model_save_path))
#
# 				epoch_ds_timeCost.append(time.time()-ds_timeStart)
# 				ds_avg_results = list(np.mean(ds_result_list, axis=0)) if batch_num * batch_size > 1 else ds_result_list
# 				epoch_avg_results[dataset.name] = ds_avg_results
# 				print('epoch:', epoch_idx, 'ds results:', dataset.name, ds_avg_results, 'size:', len(ds_result_list))
# 				logger.info('epoch:' + str(epoch_idx) + '\t' + dataset.name + '\t' + str(ds_avg_results)+'\tsize:\t'+str(len(ds_result_list))) #===== 20190328 add
#
# 			# === out for ds_idx
# 			timeCost = str(time.time() - epoch_timeStart)
# 			print('finished an epoch, epoch_idx=', epoch_idx, 'time cost:', timeCost,
# 				      "dsTimes: " + str(epoch_ds_timeCost), 'dsSizes:', [x.size for x in datasetTuple])
# 			logger.info('epochAll:' + str(epoch_idx) + ', \t\t\ttime cost: ' + timeCost + "\tdsTimes: " + str(epoch_ds_timeCost)+', saved: '+str(saved)+', loss='+str(loss)+', min_dev_loss='+str(min_dev_loss))
#
# 		#==== out epohch_idx
# 		model.saver.save(sess, save_path=model_save_path) # save final model anyway
# 		return epoch_avg_results  # result of the final epoch (i.e final epoch on test set)
#
#
# def run_train(trainConfig, datasetList, model, logger):
# 	timeStartAll = time.time()
# 	for repeat_idx in range(trainConfig.num_repeats):
# 		summwList, batchSizeList, batchNumList = _initSummBatch(trainConfig, datasetList, repeat_idx)
# 		logger.info('repeat_idx:'+str(repeat_idx))
# 		logger.info('batchSizeList:'+str(batchSizeList)+', batchNumList:'+str(batchNumList))
#
# 		fold_ndcgs_list = []
# 		for fold_idx, datasetTuple in enumerate(datasetList):  # for folds
# 			dsItemSizeList=[]
# 			for ds_idx, dsItem in enumerate(datasetTuple):
# 				dsItemSizeList.append(dsItem.size)
# 			logger.info('foldDsSize:'+str(dsItemSizeList))
# 			summwTuple = summwList[fold_idx]
# 			batchSizeTuple = batchSizeList[fold_idx]
# 			batchNumTuple = batchNumList[fold_idx]
# 			fold_ndcgs = _train_one_model(repeat_idx=repeat_idx, fold_idx=fold_idx
# 			                 , model=model, trainConfig=trainConfig, datasetTuple=datasetTuple
# 			                 , summwTuple=summwTuple, batchSizeTuple=batchSizeTuple, batchNumTuple=batchNumTuple
# 			                 , logger = logger)
# 			fold_ndcgs_list.append(fold_ndcgs)
# 		#==== out of for_fold_idx
# 		if (len(datasetList) > 1):  # more than one fold
# 			fold_ndcgs_avg = list(np.mean(fold_ndcgs_list, axis=0))
# 			logger.info('avg for '+str(len(fold_ndcgs_list))+' folds: '+str(fold_ndcgs_avg))  # ===== 20190328 add
#
# 	print('finished all, time cost: ' + str(time.time() - timeStartAll))
# 	print('convert time: ', sutl.timeConvert(time.time() - timeStartAll))
# 	logger.info('finished all, time cost: ' + str(time.time() - timeStartAll)+", convert time: "+str(sutl.timeConvert(time.time() - timeStartAll)))  # ===== 20190328 add
#
#
# if __name__ == '__main__':
# 	trainConfig, datasetList, actmodel, logger,dataConfig = prepareParams() # use args defined in train.xargs
# 	run_train(trainConfig, datasetList, actmodel, logger)
# 	# print('finished all')
