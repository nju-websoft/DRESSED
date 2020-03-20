'''
use 5-fold train-dev-test;
run 25 repeats for each fold, totally 125 runs
@file: train_multirun.py
@author: qxLiu
@time: 2019/7/7 10:15
'''

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import time
from ies_train.param_parser import prepareParams, _initSummBatch, sutl, tf, np

RESULT_LIST_IDX = 6

def _train_one_model(repeat_idx, fold_idx
					 , model, trainConfig, datasetTuple
					 , summwTuple, batchSizeTuple, batchNumTuple, logger):

	ds_num = len(datasetTuple)
	dev_ds_idx = 1 if ds_num>2 else None  # train/dev/test or train/test
	do_test = True
	best_dev_ndcg = 0  # to find currently best dev
	best_epoch_idx = -1
	final_test_evals = None
	useTensorboard = False

	with tf.Session() as sess:
		model.setSess(sess)
		model.init.run()
		model_save_path = None

		for epoch_idx in range(trainConfig.EPOCH_NUM): # epoch
			epoch_timeStart = time.time()

			# if trainConfig.do_debug and epoch_idx > 2:
			# 	continue  # for debug

			# epoch_avg_results = dict() # <dataset.name, avg_result>
			epoch_ds_timeCost = []
			model_save_path = trainConfig.MODEL_FILE_PATH_FULL +'_repeat{0}/model_r{1}-f{2}-e{3}'.format(str(repeat_idx), str(repeat_idx), str(fold_idx), str(epoch_idx))  # add epoch_idx for early stop
			test_out_path = trainConfig.MODEL_FILE_PATH_FULL+'_repeat{0}/test_result_r{1}-f{2}-e{3}.txt'.format(str(repeat_idx), str(repeat_idx), str(fold_idx), str(epoch_idx))

			epoch_loss_sum = 0  # loss on train
			epoch_dev_metrics = []  # list of ndcg, each item for one eu in dev
			epoch_test_metrics = []  # list of tuple(step, ndcg, ndcf), each item for one eu in test

			for ds_idx, dataset in enumerate(datasetTuple):  # train/dev/test
				# ds_result_list = []  # list of ndcg_lists
				ds_timeStart = time.time()

				if (epoch_idx+1) % trainConfig.DEV_STEP != 0 and ds_idx > 0 \
						and epoch_idx != trainConfig.EPOCH_NUM - 1:  # only run dev-test(ds_idx>0) on specific epoches to speed up: 1.%trainConfig.DEV_STEP , 2. last epoch
					continue
				if ds_num>2 and ds_idx>1 and not do_test: # only run test(ds_idx>1) when dev is not bad when has dev split(split_num>2)
					continue
				forDev = (ds_idx == dev_ds_idx)
				dataset.new_epoch() #===== important

				batch_num = batchNumTuple[ds_idx]
				batch_size = batchSizeTuple[ds_idx]
				batch_count = 0
				for batch_idx in range(batch_num):

					# if trainConfig.do_debug and batch_idx > 1:
					# 	continue  # for debug

					if dataset.has_next():
						batch_count+=1
						if ds_idx == 0:  # === train
							batch_data = dataset.get_batch(model, batch_size, useTensorboard=useTensorboard)
							if batch_data is None: # jump over illegal eu, already gold at init
								batch_count -= 1
								continue
							loss = model.update(batch_data, useTensorboard=useTensorboard)
							epoch_loss_sum += loss
						elif forDev: #=== dev  # ===== draw summ or early stop (for dev or test, only have one batch)
							eval_results = dataset.get_eval(model, batch_size, forDev=forDev, out_file=None) # forDev=True
							epoch_dev_metrics.extend(eval_results)
						elif do_test: #=== test  # ds_idx==2 and do_test=True, i.e. only run test when this epoch on dev is better than previous
							eval_results = dataset.get_eval(model, batch_size, forDev=forDev, out_file=test_out_path)   # forDev=False
							epoch_test_metrics.extend(eval_results)

				#==== out if batch_data
				epoch_ds_timeCost.append(time.time()-ds_timeStart)
				#==== out for batch_idx
				#======= 20190708 add: to save by early stop on ndcg of dev
				if ds_idx==0: # loss for train
					epoch_loss = epoch_loss_sum / batch_count
					logger.info('epoch:' + str(epoch_idx) + '\t' + dataset.name + '\tloss:' + str(epoch_loss)
								+'\tbatch_count:'+str(batch_count)
					            +'\ttime:'+str(time.time()-ds_timeStart))

				elif forDev:
					epoch_dev_ndcg = np.mean(epoch_dev_metrics,axis=0)
					logger.info('epoch:' + str(epoch_idx) + '\t' + dataset.name + '\tndcg:' + str(
						epoch_dev_ndcg) + '\tsize:\t' + str(len(epoch_dev_metrics))
					            + '\ttime:' + str(time.time() - ds_timeStart))

					if(epoch_dev_ndcg>best_dev_ndcg): # do model save, do test
						do_test = True
						best_dev_ndcg = epoch_dev_ndcg
						best_epoch_idx = epoch_idx
						model.saver.save(sess, save_path=model_save_path)
						logger.info('save model:' + str(epoch_idx)+ ', dev_ndcg:' + str(best_dev_ndcg) + ', path:' + str(model_save_path))
					else:
						do_test = False
				elif do_test:
					epoch_test_avg = np.mean(epoch_test_metrics, axis=0)
					final_test_evals = list(epoch_test_avg)
					logger.info('epoch:' + str(epoch_idx) + '\t' + dataset.name + '\t(step,ndcg,ndcf):' + str(
						epoch_test_avg) + '\tsize:\t' + str(len(epoch_test_metrics))
					            + '\ttime:' + str(time.time() - ds_timeStart))

			# === out for ds_idx
			timeCost = str(time.time() - epoch_timeStart)
			print('finished an epoch, epoch_idx=', epoch_idx, 'batch_nums=', batchNumTuple, 'time cost:', timeCost,
				      "dsTimes: " + str(epoch_ds_timeCost), 'dsSizes:', [x.size for x in datasetTuple])
			logger.info('epochAll:' + str(epoch_idx) + ', batch_num='+str(batchNumTuple)+ ', time cost: ' + timeCost + "\tdsTimes: " + str(epoch_ds_timeCost)
			            )

		sutl.retain_file(trainConfig.MODEL_FILE_PATH_FULL + '_repeat{0}/'.format(str(repeat_idx))
							 , "model_r{0}-f{1}-e".format(str(repeat_idx), str(fold_idx)),
							 'model_r{0}-f{1}-e{2}'.format(str(repeat_idx), str(fold_idx), str(best_epoch_idx))
							 , retain_by_endwith=False)  # remove redundant models, by prefix (i.e. models saved for earlier epoches)

		return best_epoch_idx, best_dev_ndcg, final_test_evals

from time import asctime

def run_train(trainConfig, datasetList, model, logger):
	timeStartAll = time.time()

	metric_file = trainConfig.MODEL_FILE_PATH_FULL+'_metrics.txt'
	with open(metric_file, 'a+', encoding='utf-8') as f:
		timeStr = asctime(time.localtime())
		f.write(timeStr+'\trepeat_idx, fold_idx, best_epoch_idx, best_dev_ndcg, test_evals(step, ndcg, ndcf), time\n')

	for repeat_idx in range(trainConfig.num_repeats):
		summwList, batchSizeList, batchNumList = _initSummBatch(trainConfig, datasetList, repeat_idx, withSumm=False) # not generate tensorflow summ
		logger.info('====================== repeat_idx:' + str(repeat_idx))
		logger.info('batchSizeList:' + str(batchSizeList))
		logger.info('batchNumList:' + str(batchNumList))

		fold_ndcgs_list = []
		# for fold_idx, datasetTuple in enumerate(datasetList):  # for folds
		for fold_idx, datasetTuple in reversed(list(enumerate(datasetList))) if trainConfig.do_rev_folds else enumerate(datasetList):  # for folds
			logger.info('---------------------- fold_idx:' + str(fold_idx))
			foldTimeStart = time.time()
			dsItemSizeList = []
			for ds_idx, dsItem in enumerate(datasetTuple):
				dsItemSizeList.append(dsItem.size)
			logger.info('foldDsSize:' + str(dsItemSizeList))
			summwTuple = summwList[fold_idx]
			batchSizeTuple = batchSizeList[fold_idx]
			batchNumTuple = batchNumList[fold_idx]
			fold_returns = _train_one_model(repeat_idx=repeat_idx, fold_idx=fold_idx
			                              , model=model, trainConfig=trainConfig, datasetTuple=datasetTuple
			                              , summwTuple=summwTuple, batchSizeTuple=batchSizeTuple,
			                              batchNumTuple=batchNumTuple
			                              , logger=logger)  #===== use local
			# fold_ndcgs_list.append(fold_ndcgs)
			with open(metric_file, 'a+', encoding='utf-8') as f:
				timeStr = asctime(time.localtime())
				f.write(timeStr + '\t{0}\t{1}\t{2}\t{3}\n'.format(str(repeat_idx), str(fold_idx), str(fold_returns),str(time.time()-foldTimeStart)))

	print('finished all, time cost: ' + str(time.time() - timeStartAll))
	print('convert time: ', sutl.timeConvert(time.time() - timeStartAll))
	logger.info('finished all, time cost: ' + str(time.time() - timeStartAll) + ", convert time: " + str(
		sutl.timeConvert(time.time() - timeStartAll)))  # ===== 20190328 add

def run_e5FoldTrainDevTest():
	trainConfig, datasetList, actmodel, logger, dataConfig = prepareParams()  # use args defined in train.xargs
	run_train(trainConfig, datasetList, actmodel, logger)

if __name__ == '__main__':
	run_e5FoldTrainDevTest()
