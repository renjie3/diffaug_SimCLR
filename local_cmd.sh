JOB_ID=`cat job_id.log`
echo $JOB_ID
NEXT_JOB_ID=`expr $JOB_ID + 1`
echo $NEXT_JOB_ID > job_id.log


MY_CMD="python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 --local 4"

echo $MY_CMD
echo ${MY_CMD}>>local_history.log
$MY_CMD

