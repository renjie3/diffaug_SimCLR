





# MY_CMD="python main2.py --batch_size 512 --epochs 1000 --arch resnet18 --data_name cifar10_20000_4class --train_mode curriculum --curriculum DBindex_high2low --train_data_drop_last --not_shuffle_train_data --load_model --load_model_path normal_45305664_1_20220207212804_0.5_200_512_model --local 1 --no_save"

# MY_CMD="python main.py --batch_size 128 --epochs 20 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --curriculum DBindex_cluster_GT --curriculum_scheduler 0_1_1 --train_data_drop_last --load_model --load_model_path random_initial_model1 --start_batch_num_ratio 0 --reorder_reverse --local 2 --no_save --my_train_loader"

MY_CMD="python main2.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode curriculum --curriculum DBindex_cluster_GT --mass_candidate mass_candidate_replacement --train_data_drop_last --load_model --load_model_path normal_45305664_1_20220207212804_0.5_200_512_model --reorder_reverse --start_batch_num_ratio 1 --curriculum_scheduler 0_1_1"

# MY_CMD="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode curriculum --curriculum DBindex_cluster_GT --train_data_drop_last --load_model --load_model_path normal_45305664_1_20220207212804_0.5_200_512_model --reorder_reverse --start_batch_num_ratio 0.5 --curriculum_scheduler 0_1_1"

# MY_CMD="python main.py --batch_size 512 --epochs 600 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --train_data_drop_last --half_batch --local 1 --no_save" 

# random_initial_model1

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

# [1.49317038 1.46205029 1.44700492 1.44585339 1.43239688 1.42751081 1.38831038 1.38641607]
# [1.544758001290737, 1.5256435889739641, 1.4890058229628536, 1.4689529423617615, 1.4505241649116285, 1.4442768025963217, 1.413575579095807, 1.2936461359176392]

# -2.2450463896281634, -2.2592076745811145, -2.3955943181903603, -2.4210430086115675, -2.5788670962595592, -2.509177539712095, -2.6954360871099228, -3.0220215544053035
# 2.43755589 2.45037112 2.57667078 2.58379051 2.61611997 2.70373485 2.81782222 2.8491028

# 10 hours: 45573480