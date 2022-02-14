import pandas as pd
import os
import numpy as np

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

for job_id in job_ids:
    sub_job1 = job_id.strip() + '_1'
    sub_job2 = job_id.strip() + '_2'
    for name in file_names:
        if sub_job1 in name and "_statistics.csv" in name:
            print(sub_job1)
            results = pd.read_csv('./results/{}'.format(name), index_col='epoch').to_dict()
            job1_test_acc_results.append(results['best_test_acc'][len(results['best_test_acc'])])
            job1_train_loss_acc_results.append(results['best_train_loss_acc'][len(results['best_train_loss_acc'])])
        
        if sub_job2 in name and "_statistics.csv" in name:
            print(sub_job1)
            results = pd.read_csv('./results/{}'.format(name), index_col='epoch').to_dict()
            job2_test_acc_results.append(results['best_test_acc'][len(results['best_test_acc'])])
            job2_train_loss_acc_results.append(results['best_train_loss_acc'][len(results['best_train_loss_acc'])])


job1_test_acc_results.append(np.mean(job1_test_acc_results))
job1_train_loss_acc_results.append(np.mean(job1_train_loss_acc_results))
job2_test_acc_results.append(np.mean(job2_test_acc_results))
job2_train_loss_acc_results.append(np.mean(job2_train_loss_acc_results))


all_result = [job1_test_acc_results, job1_train_loss_acc_results, job2_test_acc_results, job2_train_loss_acc_results]

for result in all_result:
    print_str = ''
    for i in result:
        print_str += "{}\n".format(i)
    print(print_str)

for result in all_result:
    print(result[len(result) - 1])
    