#!/bin/bash
cd `dirname $0`
cd ..
PIERMARO_JOB_ROOT_PATH=`pwd`
# echo $MY_JOB_ROOT_PATH
cd $PIERMARO_JOB_ROOT_PATH
DATE_NAME=${1}
echo $$

WHOLE_EPOCH=1000
SINGLE_EPOCH=50
REJOB_TIMES=`expr $WHOLE_EPOCH / $SINGLE_EPOCH`
MYGPUTYPE="v100s"

JOB_INFO="cifar10 baseline"
# MYCOMMEND="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_data_drop_last --train_mode inst_suppress --not_shuffle_train_data"

# MYCOMMEND2="python main.py --batch_size 512 --epochs 300 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --not_shuffle_train_data"

# whole_cifar10 DBindex_cluster_momentum_kmeans_wholeset DBindex_cluster_momentum_kmeans_repeat_v2 normal_48210871_1_20220316164538_0.5_200_512_1000_model
# normal_49260763_1_20220322155227_0.5_200_512_200_piermaro_model       200_epoch_whole_cifar10_base
# normal_48899799_1_20220319160643_0.5_200_512_1000_model       880_epoch_base
# normal_49449742_1_20220323194707_0.5_200_512_200_piermaro_model  200_epch_base_pytroch_transform
# 10 30 100
# 200 500
# 1000 1500
# random_initial_model1
# train_dbindex_loss

PIERMARO_MYCOMMEND="python main.py --batch_size 512 --epochs $SINGLE_EPOCH --piermaro_whole_epoch ${WHOLE_EPOCH} --arch resnet18 --dataset cifar10 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 850 --restore_k_when_start --num_clusters 200 500 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence --reassign_step 1 --final_high_conf_percent 0.3 --recording_more_info"

PIERMARO_MYCOMMEND2="python main.py --batch_size 512 --epochs $SINGLE_EPOCH --piermaro_whole_epoch ${WHOLE_EPOCH} --arch resnet18 --dataset cifar100 --data_name whole_cifar100 --train_mode normal --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path random_initial_model1 --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 510 --restore_k_when_start --num_clusters 200 500 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence --reassign_step 1 --confidence_thre 0.3"

PIERMARO_MYCOMMEND3="python3 -u ssl_perturbation_save_model.py --config_path configs/cifar10 --exp_name path/to/your/experiment/folder --version resnet18 --train_data_type CIFAR10 --noise_shape 1024 3 32 32 --perturb_type theory_model --epochs 1000 --min_min_attack_fn non_eot --strong_aug --class_4 --gray_train no --gray_test no --theory_train_data hierarchical32_16_period_dim30_shuffle_diffmean_knn32 --theory_test_data hierarchical32_16_period_dim30_shuffle_diffmean_test1_knn32 --random_drop_feature_num 0 1 1 --gaussian_aug_std 7 --theory_normalize --thoery_schedule_dim 30"

PIERMARO_MYCOMMEND2="No_commend2"
PIERMARO_MYCOMMEND3="No_commend3"

echo "MYCOMMEND=\"${PIERMARO_MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND2=\"${PIERMARO_MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYCOMMEND3=\"${PIERMARO_MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
echo "MYGPUTYPE=\"${MYGPUTYPE}\"" >> re_job_cmd/${DATE_NAME}.sh

PIERMARO_RETURN=`sh ./re_job.sh ${DATE_NAME}`
echo $PIERMARO_RETURN
PIERMARO_SLURM_JOB_ID=`echo $PIERMARO_RETURN | awk '{print $4}'`
SUBJOB_ID=$PIERMARO_SLURM_JOB_ID

# subjob_id_list
subjob_id_str=${SUBJOB_ID}

date >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $PIERMARO_SLURM_JOB_ID >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $JOB_INFO >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
echo $PIERMARO_MYCOMMEND >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
if [[ "$PIERMARO_MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $PIERMARO_MYCOMMEND2 >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
fi
if [[ "$PIERMARO_MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $PIERMARO_MYCOMMEND3 >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log
fi
echo -e "---------------------------------------------------------------" >>${PIERMARO_JOB_ROOT_PATH}/history_piermaro_job.log

for((i=1;i<${REJOB_TIMES};i++));
do
    test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
    while [ $? -eq 0 ]
    do
        sleep 30s
        echo $i
        test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
    done

    PIERMARO_RESTART_EPOCH=`expr $i \* $SINGLE_EPOCH`

    PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_1 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
    MYCOMMEND="${PIERMARO_MYCOMMEND} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"

    if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_2 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND2="${PIERMARO_MYCOMMEND2} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
    then
        PIERMARO_MODEL_PATH=`ls ./results | grep ${PIERMARO_SLURM_JOB_ID}_3 | grep _piermaro_model | awk 'BEGIN{FS=".pth"} {print $1}'`
        MYCOMMEND3="${PIERMARO_MYCOMMEND3} --load_piermaro_model --load_piermaro_model_path ${PIERMARO_MODEL_PATH} --piermaro_restart_epoch ${PIERMARO_RESTART_EPOCH}"
    fi

    echo "MYCOMMEND=\"${MYCOMMEND}\"" > re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND2=\"${MYCOMMEND2}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYCOMMEND3=\"${MYCOMMEND3}\"" >> re_job_cmd/${DATE_NAME}.sh
    echo "MYGPUTYPE=\"${MYGPUTYPE}\"" >> re_job_cmd/${DATE_NAME}.sh

    SUBJOB_RETURN=`sh ./re_job.sh ${DATE_NAME}`
    echo $SUBJOB_RETURN
    SUBJOB_ID=`echo $SUBJOB_RETURN | awk '{print $4}'`
    subjob_id_str="${subjob_id_str} ${SUBJOB_ID}"

done

test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
while [ $? -eq 0 ]
do
    sleep 30s
    echo $i
    test -e ./FLAG_ROOM/RUNNING_FLAG_${SUBJOB_ID}
done

date >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $PIERMARO_SLURM_JOB_ID >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $JOB_INFO >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo $MYCOMMEND >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
if [[ "$MYCOMMEND2" != *"No_commend2"* ]]
then
    echo $MYCOMMEND2 >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
fi
if [[ "$MYCOMMEND3" != *"No_commend3"* ]]
then
    echo $MYCOMMEND3 >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
fi
echo ${subjob_id_str} >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
echo -e "---------------------------------------------------------------" >>${PIERMARO_JOB_ROOT_PATH}/finish_piermaro_history.log
