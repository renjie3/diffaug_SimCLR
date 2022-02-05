#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

JOB_INFO="Train simclr with differentiable data augmentation. Removed Grayscale"
MYCOMMEND="python main.py --batch_size 512 --epochs 1000 --arch resnet18 --feature_dim 2"

cat ./slurm_files/sconfigs1.sb > submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND}\"" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log