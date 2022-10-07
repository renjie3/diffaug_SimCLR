cd /egr/research-dselab/renjie3/renjie/CL_cluster/diffaug_SimCLR
MY_CMD="python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 --job_id 14 --local 4 "
${MY_CMD}

if [ $? -eq 0 ];then
echo -e "grandriver JobID:14 \n Python_command: \n python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 \n " | mail -s "[Done] grandriver " thurenjie@outlook.com
else
echo -e "grandriver JobID:14 \n Python_command: \n python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 \n " | mail -s "[Fail] grandriver " thurenjie@outlook.com
fi
