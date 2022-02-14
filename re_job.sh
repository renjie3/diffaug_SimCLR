#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="3:50:00"
MYCPU="5"
MYGPUTYPE="v100s"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND="python main.py --batch_size 512 --epochs 600 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --half_batch"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 600 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --reorder_reverse --half_batch"

# MYCOMMEND3="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical32_16_period_dim30_shuffle_diffmean_knn32 --theory_test_data hierarchical32_16_period_dim30_shuffle_diffmean_test1_knn32 --random_drop_feature_num 0 1 1 --gaussian_aug_std 7 --theory_normalize --thoery_schedule_dim 30"

# # MYCOMMEND2="No_commend2"
# MYCOMMEND3="No_commend3"

source ./re_job_cmd.sh

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
cat ./slurm_files/sconfigs3_re_job.sb >> submit.sb
MY_RETURN=`sbatch submit.sb`

# MY_RETURN="Submitted batch job 45160890"

MY_SLURM_JOB_ID=`echo $MY_RETURN | awk '{print $4}'`

touch ./FLAG_ROOM/RUNNING_FLAG_${MY_SLURM_JOB_ID}

echo $MY_RETURN

# #print the information of a job into one file
# date >>${MY_JOB_ROOT_PATH}/history_job.log
# echo $MY_SLURM_JOB_ID >>${MY_JOB_ROOT_PATH}/history_job.log
# echo $JOB_INFO >>${MY_JOB_ROOT_PATH}/history_job.log
# echo $MYCOMMEND >>${MY_JOB_ROOT_PATH}/history_job.log
# if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
# then
#     echo $MYCOMMEND2 >>${MY_JOB_ROOT_PATH}/history_job.log
# fi
# if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
# then
#     echo $MYCOMMEND3 >>${MY_JOB_ROOT_PATH}/history_job.log
# fi
# echo "---------------------------------------------------------------" >>${MY_JOB_ROOT_PATH}/history_job.log
