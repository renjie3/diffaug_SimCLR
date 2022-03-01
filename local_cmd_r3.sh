





# MY_CMD="python main2.py --batch_size 512 --epochs 1000 --arch resnet18 --data_name cifar10_20000_4class --train_mode curriculum --curriculum DBindex_high2low --train_data_drop_last --not_shuffle_train_data --load_model --load_model_path normal_45305664_1_20220207212804_0.5_200_512_model --local 1 --no_save"

# MY_CMD="python main.py --batch_size 128 --epochs 20 --arch resnet18 --data_name cifar10_1024_4class --train_mode normal --curriculum DBindex_cluster_GT --curriculum_scheduler 0_1_1 --train_data_drop_last --load_model --load_model_path random_initial_model1 --start_batch_num_ratio 0 --reorder_reverse --local 2 --no_save --my_train_loader"

# 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', ### 'DBindex_product_inst_cluster_GT', #####'DBindex_cluster_GT_org_sample_only'
models1=(normal_45935891_1_20220220130320_0.5_200_128_model normal_45935892_1_20220220130320_0.5_200_128_model normal_45935893_1_20220220130515_0.5_200_128_model normal_45935894_1_20220220130607_0.5_200_128_model normal_45935895_1_20220220130607_0.5_200_128_model normal_45935896_1_20220220130607_0.5_200_128_model normal_45935897_1_20220220130607_0.5_200_128_model normal_45935898_1_20220220130636_0.5_200_128_model)
models2=(normal_45935891_2_20220220130320_0.5_200_128_model normal_45935892_2_20220220130320_0.5_200_128_model normal_45935893_2_20220220130515_0.5_200_128_model normal_45935894_2_20220220130607_0.5_200_128_model normal_45935895_2_20220220130607_0.5_200_128_model normal_45935896_2_20220220130607_0.5_200_128_model normal_45935897_2_20220220130607_0.5_200_128_model normal_45935898_2_20220220130636_0.5_200_128_model)
# models2=(normal_45934549_2_20220220112954_0.5_200_128_model normal_45934550_2_20220220112955_0.5_200_128_model normal_45934551_2_20220220112956_0.5_200_128_model normal_45934552_2_20220220113003_0.5_200_128_model normal_45934553_2_20220220113003_0.5_200_128_model normal_45934554_2_20220220113003_0.5_200_128_model normal_45934555_2_20220220113003_0.5_200_128_model normal_45934556_2_20220220113004_0.5_200_128_model)
# models2=(curriculum_45623571_2_20220214094551_0.5_200_128_model curriculum_45623572_2_20220214094552_0.5_200_128_model curriculum_45623573_2_20220214094554_0.5_200_128_model curriculum_45623574_2_20220214094554_0.5_200_128_model curriculum_45623575_2_20220214094554_0.5_200_128_model curriculum_45623576_2_20220214094554_0.5_200_128_model curriculum_45623577_2_20220214094603_0.5_200_128_model curriculum_45623578_2_20220214094603_0.5_200_128_model)


for((i=0;i<8;i++));
do
MY_CMD="python main_visualization.py --batch_size 128 --epochs 1200 --arch resnet18 --data_name cifar10_1024_4class --train_mode auto_aug --curriculum DBindex_ratio_inst_cluster_GT --load_model --load_model_path ${models2[${i}]} --my_train_loader --train_data_drop_last --attack_alpha 2  --augmentation_prob 1.0 0.5 0.8 0.2 --color_jitter_strength 1.0 --save_aug_file_name temp3.txt --attack_steps 4 --local 2 --no_save"
echo $MY_CMD
$MY_CMD
done

# MY_CMD="python main.py --batch_size 128 --epochs 2000 --arch resnet18 --data_name cifar10_1024_4class --train_mode curriculum --curriculum DBindex_cluster_GT --mass_candidate mass_candidate --train_data_drop_last --load_model --load_model_path random_initial_model1 --start_batch_num_ratio 0.5 --curriculum_scheduler 0_1_1 --my_data_loader --local 2 --no_save"

# # MY_CMD="python main.py --batch_size 512 --epochs 600 --arch resnet18 --data_name cifar10_20000_4class --train_mode inst_suppress --train_data_drop_last --half_batch --local 1 --no_save" 

# # random_initial_model1

# echo $MY_CMD
# echo ${MY_CMD}>>local_history.log
# $MY_CMD

# [1.49317038 1.46205029 1.44700492 1.44585339 1.43239688 1.42751081 1.38831038 1.38641607]
# [1.544758001290737, 1.5256435889739641, 1.4890058229628536, 1.4689529423617615, 1.4505241649116285, 1.4442768025963217, 1.413575579095807, 1.2936461359176392]

# -2.2450463896281634, -2.2592076745811145, -2.3955943181903603, -2.4210430086115675, -2.5788670962595592, -2.509177539712095, -2.6954360871099228, -3.0220215544053035
# 2.43755589 2.45037112 2.57667078 2.58379051 2.61611997 2.70373485 2.81782222 2.8491028

# 10 hours: 45573480

# -13.038444706839016, -11.214536558996112, -10.37817184842927, -9.377252472092408, -10.177998629657333, -7.856866826727006, -7.898224416462478, -6.576466283231039
# -5.987733222407447, -6.785825839079811, -6.301986528801884, -6.045964942611055, -7.825271918385527, -8.0758290334521, -7.102982391667136, -12.969467036995589
# -2.793397692226195, -2.8042373644477787, -2.7888438490049685, -2.7317711742254946, -2.5700642653518786, -2.4965673150571766, -2.3354176198694194, -2.1756680071870496
# -2.2455631229323254, -2.1888644323485367, -2.348398770816585, -2.275140060955429, -2.5149633573902928, -2.513110604846296, -2.8442126783669273, -3.167525347769553