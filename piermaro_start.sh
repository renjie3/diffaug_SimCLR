#!/bin/bash
cd `dirname $0`
MY_PATH=`pwd`
# echo $MY_PATH
cd $MY_PATH

DATE_NAME=`date '+%Y%m%d%H%M%S'`
cat piermaro_cmd.sh > ./piermaro_cmd/piermaro_${DATE_NAME}.sh
cat piermaro_cmd.sh > ./piermaro/history_piermaro_${DATE_NAME}.sh
echo piermaro_${DATE_NAME}.sh
sh ./piermaro_cmd/piermaro_${DATE_NAME}.sh ${DATE_NAME}

rm -f ./piermaro_cmd/piermaro_${DATE_NAME}.sh