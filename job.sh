#!/bin/bash
cd `dirname $0`
MY_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $MY_JOB_ROOT_PATH

MYTIME="30:50:00"
MYCPU="5"
MYGPUTYPE="v100"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# 1778 0 1600 0.25 1333 0.5
# 1389 0 1167 0.5
# normal 1021
# 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', ### 'DBindex_product_inst_cluster_GT', #####'DBindex_cluster_GT_org_sample_only' DBindex_cluster_momentum_kmeans_repeat_v2 _mean_dbindex DBindex_cluster_momentum_kmeans_repeat_v2_weighted_cluster
# all_in_flag random_last_3batch random_initial_model1
# whole_cifar10 
# --pretrain_model_path normal_45305664_1_20220207212804_0.5_200_512_model
# normal_45921554_1_20220219224232_0.5_200_128_model
# normal_45934552_2_20220220113003_0.5_200_128_best_test_acc_model
# 4 5 7 10 15 20
# 10 15 20 40
# python main.py --batch_size 512 --epochs 1500 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path random_initial_model1 --train_data_drop_last --my_train_loader --kornia_transform  

MYCOMMEND="python main.py --batch_size 512 --epochs 800 --arch resnet18 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_49260763_1_20220322155227_0.5_200_512_200_model --kornia_transform --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch 10 --restore_k_when_start --num_clusters 1000 1500 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence"

MYCOMMEND2="python main.py --batch_size 512 --epochs 500 --arch resnet18 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_48899696_1_20220319160643_0.5_200_512_1000_model --kornia_transform --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 500 --repeat_num 1"

MYCOMMEND3="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --train_data_drop_last"

# normal_46334423_2_20220301154842_0.5_200_512_best_test_acc_model normal_46334424_2_20220301154842_0.5_200_512_model normal_46334425_2_20220301154959_0.5_200_512_model normal_46334426_2_20220301155011_0.5_200_512_model normal_46334427_2_20220301155120_0.5_200_512_model normal_46334428_2_20220301155152_0.5_200_512_model normal_46334429_2_20220301155216_0.5_200_512_model normal_46334430_2_20220301155216_0.5_200_512_model

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
