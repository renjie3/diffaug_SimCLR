MY_CMD="python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --dataset cifar10 --neg gt_diff_label --load_model --load_model_path vanilla_62545265_1_128_0.5_200_512_1000_final_model --local 2"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

