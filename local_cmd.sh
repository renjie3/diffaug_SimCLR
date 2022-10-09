MY_CMD="python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --dataset cifar10 --neg gt_test2 --local 5"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

