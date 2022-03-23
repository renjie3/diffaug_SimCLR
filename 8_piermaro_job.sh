cat piermaro_cmd.sh > 8_piermaro_cmd_code.sh

for((i=0;i<4;i++));
do
sh 8_piermaro_start.sh &

sleep 3s

done
