#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="3:50:00"
MYCPU="5"
MYGPUTYPE="v100"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# 1778 0 1600 0.25 1333 0.5
# 1389 0 1167 0.5
# normal 1021
# 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', ### 'DBindex_product_inst_cluster_GT', #####'DBindex_cluster_GT_org_sample_only'
# all_in_flag random_last_3batch random_initial_model1
# whole_cifar10 
# --pretrain_model_path normal_45305664_1_20220207212804_0.5_200_512_model
# normal_45921554_1_20220219224232_0.5_200_128_model
# normal_45934552_2_20220220113003_0.5_200_128_best_test_acc_model

MYCOMMEND="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_kmeans --perturb_batchsize 0 --load_model --load_model_path normal_45934552_2_20220220113003_0.5_200_128_best_test_acc_model --my_train_loader --train_data_drop_last --attack_alpha 2 --attack_steps 4 --weight_dbindex_loss 0.01 --start_dbindex_loss_epoch -1 --num_clusters 4 5 7 10 15 20 --repeat_num 1"

MYCOMMEND2="python main.py --batch_size 512 --epochs 1500 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --curriculum DBindex_cluster_kmeans --perturb_batchsize 0 --load_model --load_model_path random_initial_model1 --my_train_loader --train_data_drop_last --attack_alpha 2 --attack_steps 4 --weight_dbindex_loss 0.05 --start_dbindex_loss_epoch 1000 --num_clusters 4 5 7 10 15 20 --repeat_num 1"

MYCOMMEND3="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --train_data_drop_last"

# --my_data_loader
# normal_45625752_1_20220214102432_0.5_200_128_model normal_45625753_1_20220214102432_0.5_200_128_model normal_45625754_1_20220214102433_0.5_200_128_model normal_45625755_1_20220214102434_0.5_200_128_model normal_45625756_1_20220214102435_0.5_200_128_model normal_45625757_1_20220214102435_0.5_200_128_model normal_45625758_1_20220214102436_0.5_200_128_model normal_45625759_1_20220214102437_0.5_200_128_model
# normal_45625752_2_20220214102432_0.5_200_128_model normal_45625753_2_20220214102432_0.5_200_128_model normal_45625754_2_20220214102433_0.5_200_128_model normal_45625755_2_20220214102434_0.5_200_128_model normal_45625756_2_20220214102434_0.5_200_128_model normal_45625757_2_20220214102436_0.5_200_128_model normal_45625758_2_20220214102436_0.5_200_128_model normal_45625759_2_20220214102437_0.5_200_128_model

# normal_45623148_1_20220214092107_0.5_200_128_model normal_45623149_1_20220214092107_0.5_200_128_model normal_45623150_1_20220214092108_0.5_200_128_model normal_45623151_1_20220214092109_0.5_200_128_model normal_45623152_1_20220214092110_0.5_200_128_model normal_45623153_1_20220214092111_0.5_200_128_model normal_45623154_1_20220214092111_0.5_200_128_model normal_45623155_1_20220214092112_0.5_200_128_model
# normal_45623148_2_20220214092107_0.5_200_128_model normal_45623149_2_20220214092107_0.5_200_128_model normal_45623150_2_20220214092108_0.5_200_128_model normal_45623151_2_20220214092109_0.5_200_128_model normal_45623152_2_20220214092109_0.5_200_128_model normal_45623153_2_20220214092111_0.5_200_128_model normal_45623154_2_20220214092111_0.5_200_128_model normal_45623155_2_20220214092112_0.5_200_128_model

MYCOMMEND2="No_commend2"
MYCOMMEND3="No_commend3"

cat ./slurm_files/sconfigs1_cmse.sb > submit.sb
# cat ./slurm_files/sconfigs1.sb > submit.sb
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
