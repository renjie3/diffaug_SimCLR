
import os

# f = open('./tobe_read.txt', 'r')
# job_ids = f.readlines()
# f.close()

filePath = './piermaro_log/'
file_names = os.listdir(filePath)
pid_list = []

for filename in file_names:

    f = open(filePath + filename, 'r')
    pid = f.readline().strip()
    f.close()

    pid_list.append(pid)

print(' '.join(pid_list))