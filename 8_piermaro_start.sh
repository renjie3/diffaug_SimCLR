#!/bin/bash
cd `dirname $0`
MY_PATH=`pwd`
# echo $MY_PATH
cd $MY_PATH

DATE_NAME=`date '+%Y%m%d%H%M%S'`
cat 8_piermaro_cmd_code.sh > piermaro_${DATE_NAME}.sh
cat 8_piermaro_cmd_code.sh > ./piermaro/history_piermaro_${DATE_NAME}.sh
echo piermaro_${DATE_NAME}.sh
sh piermaro_${DATE_NAME}.sh ${DATE_NAME} > piermaro_log/${DATE_NAME}.log

wait

rm -f piermaro_${DATE_NAME}.sh