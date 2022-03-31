import pandas as pd
import os
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--max_epoch', default=2000, type=int, help='Use pretrained model to get DBindex')

# args parse
args = parser.parse_args()

f = open('./tobe_read.txt', 'r')
job_ids = f.readlines()
f.close()

filePath = './results/'
file_names = os.listdir(filePath)
# print(file_names)

# print(job_ids)

job1_test_acc_results = []
job2_test_acc_results = []

job1_train_loss_acc_results = []
job2_train_loss_acc_results = []

model1_list = []
model2_list = []

std_list = []

# csv_name = "_statistics_final_10_line.csv"
csv_name = "_statistics.csv"

for job_id in job_ids:
    sub_job1 = job_id.strip() + '_1'
    sub_job2 = job_id.strip() + '_2'

    for name in file_names:
        if sub_job1 in name and csv_name in name:
        # if sub_job1 in name and "_statistics.csv" in name:
            print(sub_job1)
            results = pd.read_csv('./results/{}'.format(name), index_col='epoch').to_dict()
            max_key = 0
            for key in results['test_acc@1']:
                if max_key < key:
                    max_key = key
            max_key = min(max_key, args.max_epoch)
            job1_test_acc_results.append(results['test_acc@1'][max_key])
            # job1_train_loss_acc_results.append(results['best_train_loss_acc'][max_key])
        
        if sub_job2 in name and csv_name in name:
            print(sub_job2)
            results = pd.read_csv('./results/{}'.format(name), index_col='epoch').to_dict()
            max_key = 0
            for key in results['test_acc@1']:
                if max_key < key:
                    max_key = key
            max_key = min(max_key, args.max_epoch)
            job2_test_acc_results.append(results['test_acc@1'][max_key])
            # job2_train_loss_acc_results.append(results['best_train_loss_acc'][max_key])


        if sub_job1 in name and "model" in name and "piermaro" not in name and "best_test_acc" not in name:
        # if sub_job1 in name and "_statistics.csv" in name:
            model1_list.append(name.replace(".pth", ""))
            
        if sub_job2 in name and "model" in name and "piermaro" not in name and "best_test_acc" not in name:
            model2_list.append(name.replace(".pth", ""))
        

# ±
# job1_test_acc_results += job2_test_acc_results
# job1_train_loss_acc_results += job2_train_loss_acc_results

std_list.append(np.std(job1_test_acc_results))
# std_list.append(np.std(job1_train_loss_acc_results))
std_list.append(np.std(job2_test_acc_results))
# std_list.append(np.std(job2_train_loss_acc_results))

job1_test_acc_results.append(np.mean(job1_test_acc_results))
# job1_train_loss_acc_results.append(np.mean(job1_train_loss_acc_results))
job2_test_acc_results.append(np.mean(job2_test_acc_results))
# job2_train_loss_acc_results.append(np.mean(job2_train_loss_acc_results))

all_result = [job1_test_acc_results, job2_test_acc_results]

for idx,result in enumerate(all_result):
    print_str = ''
    for i in result:
        print_str += "{}\n".format(i)
    print(print_str)

for idx,result in enumerate(all_result):
    print("{:.2f}±{:.2f}".format(result[len(result) - 1], std_list[idx]))

print(' '.join(model1_list))
# print(' '.join(model2_list))
    