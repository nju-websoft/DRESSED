#!/usr/bin/env bash
declare -a ds_list=("dbpedia" "lmdb" "dsfaces")

for ds_name in "${ds_list[@]}"
do
echo $ds_name
for ((repidx=0; repidx<25; repidx++))
do
runname="run$repidx"
echo "$runname"
command="python -m ies_train.train_multirun $ds_name $runname -epoch_num 50 -dev_step 1 -batch_size 1"
echo "$command"
if eval $command; then
	echo "command succeeded"
else
	echo "command failed at $runname"
	break
fi
done
done