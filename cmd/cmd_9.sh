cd /egr/research-dselab/renjie3/renjie/CL_cluster/diffaug_SimCLR
MY_CMD="nohup python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 --job_id 9 --local 4 >.results/9.log 2>.results/9.err "

if [ $? -eq 0 ];then
echo -e "grandriver JobID:9 
 Python_command: 
 nohup python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 
 " | mail -s "[Done] grandriver " thurenjie@outlook.com
else
echo -e "grandriver JobID:9 
 Python_command: 
 nohup python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 
 " | mail -s "[Fail] grandriver " thurenjie@outlook.com
fi
