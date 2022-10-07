cd /egr/research-dselab/renjie3/renjie/CL_cluster/diffaug_SimCLR
MY_CMD="nohup python main.py --arch resnet18 --batch_size 512 --epochs 1000 --method cluster --neg gt_test5 --job_id 7 --local 4 >.results/7.log 2>.results/7.err "

 if [ $? -eq 0 ];then
echo -e "grandriver JobID:${JOB_ID} 
 Python_command: 
 ${MY_CMD} 
 " | mail -s "[Done] grandriver ${SLURM_JOB_ID}" thurenjie@outlook.com
else
echo -e "grandriver JobID:${JOB_ID} 
 Python_command: 
 ${MY_CMD} 
 " | mail -s "[Fail] grandriver ${SLURM_JOB_ID}" thurenjie@outlook.com
fi
