# MY_CMD="python main2.py --batch_size 512 --epochs 1000 --arch resnet18 --data_name cifar10_20000_4class --train_mode curriculum --curriculum DBindex_high2low --train_data_drop_last --not_shuffle_train_data --load_model --load_model_path normal_45305664_1_20220207212804_0.5_200_512_model --local 1 --no_save"

# MY_CMD="python main.py --batch_size 128 --epochs 20 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --curriculum DBindex_cluster_GT --curriculum_scheduler 0_1_1 --train_data_drop_last --load_model --load_model_path random_initial_model1 --start_batch_num_ratio 0 --reorder_reverse --local 2 --no_save --my_train_loader"

# 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', ### 'DBindex_product_inst_cluster_GT', #####'DBindex_cluster_GT_org_sample_only'

# models=(normal_45652097_1_20220214233854_0.5_200_128_model normal_45652098_1_20220214233856_0.5_200_128_model normal_45652099_1_20220214233858_0.5_200_128_model normal_45652100_1_20220214233859_0.5_200_128_model normal_45652101_1_20220214233854_0.5_200_128_model normal_45652102_1_20220214233856_0.5_200_128_model normal_45652103_1_20220214233857_0.5_200_128_model normal_45652104_1_20220214233858_0.5_200_128_model
# normal_45652097_2_20220214233854_0.5_200_128_model normal_45652098_2_20220214233856_0.5_200_128_model normal_45652099_2_20220214233858_0.5_200_128_model normal_45652100_2_20220214233859_0.5_200_128_model normal_45652101_2_20220214233854_0.5_200_128_model normal_45652102_2_20220214233856_0.5_200_128_model normal_45652103_2_20220214233857_0.5_200_128_model normal_45652104_2_20220214233858_0.5_200_128_model)


# for((i=0;i<16;i++));
# do
# MY_CMD="python main.py --batch_size 128 --epochs 1000 --arch resnet18 --data_name cifar10_1024_4class --train_mode just_plot --curriculum DBindex_high2low --mass_candidate mass_candidate --train_data_drop_last --load_model --load_model_path ${models[${i}]} --start_batch_num_ratio 0.5 --curriculum_scheduler 0_1_1 --my_data_loader"
# echo $MY_CMD
# $MY_CMD
# done

# para=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# para2=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

# for((i=0;i<11;i++));
# do
# curriculum_45893765_1_20220219111714_0.5_200_128_model DBindex_high2low
# curriculum_45893779_1_20220219111714_0.5_200_128_model DBindex_cluster_GT
# curriculum_45870333_1_20220218202830_0.5_200_128_model DBindex_ratio_inst_cluster_GT
# train_dbindex_loss_47850754_1_20220313165236_0

# MY_CMD="python main.py --batch_size 512 --epochs 500 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path train_dbindex_loss_47850754_1_20220313165236_0.5_200_128_piermaro_model --my_train_loader --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch -1 --restore_k_when_start --num_clusters 4 15 --kmeans_just_plot --load_momentum_model --my_test_loader --repeat_num 1 --local 2 --no_save"

# MY_CMD="python main.py --batch_size 512 --epochs 500 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path train_dbindex_loss_46301913_1_20220228195148_0.5_200_128_best_test_acc_model --my_train_loader --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch -1 --restore_k_when_start --num_clusters 4 15 --kmeans_just_plot --kmeans_just_plot_test --my_test_loader --repeat_num 1 --local 2 --no_save" whole_cifar10

# normal_48592770_1_20220317173822_0.5_200_512_1000_model kmeans_plot --my_train_loader --kornia_transform
# train_dbindex_loss_49137477_1_20220321201720_0.5_200_512_500_model normal_48899799_1_20220319160643_0.5_200_512_1000_model
# train_dbindex_loss_47850749_1_20220313165236_0.5_200_128_best_test_acc_model normal_46284817_1_20220228105741_0.5_200_128_best_test_acc_model
# normal_49260763_1_20220322155227_0.5_200_512_200_model

MY_CMD="python main.py --batch_size 512 --epochs 100 --arch resnet18 --dataset cifar10 --data_name whole_cifar10 --train_mode normal --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_48899799_1_20220319160643_0.5_200_512_1000_model --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 200 500 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence --reassign_step 1 --local 2 --no_save"

# MY_CMD="python linear.py --batch_size 512 --epochs 100 --model_path results/train_dbindex_loss_49854381_1_20220327175242_0.5_200_512_400_model.pth"

MY_CMD="python main.py --batch_size 512 --epochs 100 --arch resnet50 --dataset cifar10 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 200 500 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence --reassign_step 1 --local 2 --no_save"

# MY_CMD="python main2.py --batch_size 512 --epochs 100 --arch resnet18 --dataset cifar10 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_48899799_1_20220319160643_0.5_200_512_1000_model --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 10 30 100 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --flag_select_confidence --reassign_step 1 --final_high_conf_percent 0.8 --local 3 --no_save"

MY_CMD="python main5.py --batch_size 512 --epochs 100 --arch resnet18 --dataset cifar10 --data_name whole_cifar10 --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_48899799_1_20220319160643_0.5_200_512_1000_model --train_data_drop_last --weight_dbindex_loss 1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 150 1000 --repeat_num 1 --use_wholeset_centroid --use_mean_dbindex --use_org_sample_dbindex --reassign_step 1 --final_high_conf_percent 0.8 --keep_gradient_on_center --inter_class_type batch --dbindex_type half --use_GT --local 1 --no_save"
# normal_49136163_1_20220321200117_0.5_200_128_1000_model

# MY_CMD="python linear.py --arch resnet50 --batch_size 512 --epochs 100 --model_path results/normal_50093884_1_20220329102753_0.5_200_512_1000_model.pth"

# MY_CMD="python main.py --batch_size 128 --epochs 500 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_48992808_1_20220320153839_0.5_200_128_1000_model --kornia_transform --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch 1 --restore_k_when_start --num_clusters 20 40 --repeat_num 1 --use_out_dbindex --use_sim --local 2 --no_save"

# MY_CMD="python main.py --batch_size 512 --epochs 100 --piermaro_whole_epoch 1600 --arch resnet18 --data_name cifar10_1024_4class  --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path random_initial_model1 --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch 1000 --restore_k_when_start --num_clusters 10 15 20 40 --load_piermaro_model --load_piermaro_model_path train_dbindex_loss_48260488_1_20220316005605_0.5_200_512_piermaro_model --piermaro_restart_epoch 1500"

# python main.py --batch_size 512 --epochs 500 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_wholeset --load_model --load_model_path normal_46334424_2_20220301154842_0.5_200_512_best_test_acc_model --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch -1 --restore_k_when_start --num_clusters 4 5 7 10 15 20 --repeat_num 1 --local 2 --no_save

# MY_CMD="python main.py --batch_size 512 --epochs 1500 --arch resnet18 --data_name cifar10_1024_4class --train_mode train_dbindex_loss --curriculum DBindex_cluster_momentum_kmeans_repeat_v2_mean_dbindex --load_model --load_model_path random_initial_model1 --my_train_loader --train_data_drop_last --weight_dbindex_loss 0.1 --start_dbindex_loss_epoch 1000 --restore_k_when_start --num_clusters 4 5 7 10 15 20 --repeat_num 1"

# MY_CMD="python check_dbindex.py --data_name cifar10_20000_4class --load_model differentiable_45160867_1_20220205220613_0.5_200_512_model" 

# random_initial_model1

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD
# done

# [1.49317038 1.46205029 1.44700492 1.44585339 1.43239688 1.42751081 1.38831038 1.38641607]
# [1.544758001290737, 1.5256435889739641, 1.4890058229628536, 1.4689529423617615, 1.4505241649116285, 1.4442768025963217, 1.413575579095807, 1.2936461359176392]

# -2.2450463896281634, -2.2592076745811145, -2.3955943181903603, -2.4210430086115675, -2.5788670962595592, -2.509177539712095, -2.6954360871099228, -3.0220215544053035
# 2.43755589 2.45037112 2.57667078 2.58379051 2.61611997 2.70373485 2.81782222 2.8491028

# 10 hours: 45573480

# -13.038444706839016, -11.214536558996112, -10.37817184842927, -9.377252472092408, -10.177998629657333, -7.856866826727006, -7.898224416462478, -6.576466283231039
# -5.987733222407447, -6.785825839079811, -6.301986528801884, -6.045964942611055, -7.825271918385527, -8.0758290334521, -7.102982391667136, -12.969467036995589
# -2.793397692226195, -2.8042373644477787, -2.7888438490049685, -2.7317711742254946, -2.5700642653518786, -2.4965673150571766, -2.3354176198694194, -2.1756680071870496
# -2.2455631229323254, -2.1888644323485367, -2.348398770816585, -2.275140060955429, -2.5149633573902928, -2.513110604846296, -2.8442126783669273, -3.167525347769553