MY_CMD="python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test2 --local 1"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD