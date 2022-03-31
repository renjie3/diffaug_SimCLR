cat piermaro_cmd.sh > 8_piermaro_cmd_code.sh

N_TASK=4

for((i=0;i<${N_TASK};i++));
do
sh 8_piermaro_start.sh &

sleep 3s

done
