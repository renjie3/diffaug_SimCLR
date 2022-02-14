#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="6:00:00"
MYCPU="5"
MYGPUTYPE="v100"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# 1778 0 1600 0.25 1333 0.5
# 1389 0 1167 0.5 
# 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', ### 'DBindex_product_inst_cluster_GT', #####'DBindex_cluster_GT_org_sample_only'
# all_in_flag random_last_3batch

MYCOMMEND="python main.py --batch_size 128 --epochs 1167 --arch resnet18 --data_name cifar10_1024_4class --train_mode curriculum --curriculum DBindex_ratio_inst_cluster_GT --mass_candidate mass_candidate --train_data_drop_last --load_model --load_model_path random_initial_model1 --start_batch_num_ratio 0.5 --curriculum_scheduler 0_1_1"

MYCOMMEND2="python main.py --batch_size 128 --epochs 1167 --arch resnet18 --data_name cifar10_1024_4class --train_mode curriculum --curriculum DBindex_ratio_inst_cluster_GT --mass_candidate mass_candidate --train_data_drop_last --load_model --load_model_path random_initial_model1 --reorder_reverse --start_batch_num_ratio 0.5 --curriculum_scheduler 0_1_1"

MYCOMMEND3="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --train_data_drop_last"

# MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
echo "#SBATCH --gres=gpu:${MYGPUTYPE}:1"  >> submit.sb
echo "#SBATCH --time=${MYTIME}             # limit of wall clock time - how long the job will run (same as -t)" >> submit.sb
echo "#SBATCH --cpus-per-task=${MYCPU}           # number of CPUs (or cores) per task (same as -c)" >> submit.sb
# echo "#SBATCH --nodelist=nvl-001" >> submit.sb
echo "#SBATCH -o ${MY_JOB_ROOT_PATH}/logfile/%j.log" >> submit.sb
echo "#SBATCH -e ${MY_JOB_ROOT_PATH}/logfile/%j.err" >> submit.sb
cat ./slurm_files/sconfigs2.sb >> submit.sb
echo "JOB_INFO=\"${JOB_INFO}\"" >> submit.sb
echo "MYCOMMEND=\"${MYCOMMEND} --job_id \${SLURM_JOB_ID}_1\"" >> submit.sb
echo "MYCOMMEND2=\"${MYCOMMEND2} --job_id \${SLURM_JOB_ID}_2\"" >> submit.sb
echo "MYCOMMEND3=\"${MYCOMMEND3} --job_id \${SLURM_JOB_ID}_3\"" >> submit.sb
cat ./slurm_files/sconfigs3.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

echo $MY_RETURN

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

#print the information of a job into one file
date >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
fi
echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
