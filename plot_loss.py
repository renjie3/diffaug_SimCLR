import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--model_path1', default='', type=str, help='Use pretrained model to get DBindex')
parser.add_argument('--model_path2', default='', type=str, help='Use pretrained model to get DBindex')

# args parse
args = parser.parse_args()

model_path1 = args.model_path1.replace("_model", "")
model_path2 = args.model_path2.replace("_model", "")
pd_reader = pd.read_csv("./results/{}_statistics.csv".format(model_path1))
epoch = pd_reader.values[:,0]
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]
# acc_top5 = pd_reader.values[:,3]
# best_test_acc = pd_reader.values[:,4]
# best_test_acc_loss = pd_reader.values[:,5]
# best_train_loss_acc = pd_reader.values[:,6]
# best_train_loss = pd_reader.values[:,7]

loss1 = pd_reader.values[:,1]
acc1 = pd_reader.values[:,2]

pd_reader = pd.read_csv("./results/{}_statistics.csv".format(model_path2))
loss2 = pd_reader.values[:,1]
acc2 = pd_reader.values[:,2]

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

cut = min(len(loss1), len(loss2))

p2 = ax.plot(epoch[:cut], loss1[:cut],'r-', label = 'loss1')
p3 = ax.plot(epoch[:cut], loss2[:cut], 'b-', label = 'loss2')
ax.legend()
p3 = ax1.plot(epoch[:cut], acc1[:cut], 'r-', label = 'test_acc1')
p3 = ax1.plot(epoch[:cut], acc2[:cut], 'b-', label = 'test_acc2')
ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax1.set_ylabel('acc')
plt.title('Training loss and acc')
plt.savefig("./results/{}_compare_reverse.png".format(model_path1))
plt.close()

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

cut = 200

p2 = ax.plot(epoch[:cut], loss1[:cut],'r-', label = 'loss1')
p3 = ax.plot(epoch[:cut], loss2[:cut], 'b-', label = 'loss2')
ax.legend()
p3 = ax1.plot(epoch[:cut], acc1[:cut], 'r-', label = 'test_acc1')
p3 = ax1.plot(epoch[:cut], acc2[:cut], 'b-', label = 'test_acc2')
ax1.legend()

#显示图例
# p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# plt.legend()
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
ax1.set_ylabel('acc')
plt.title('Training loss and acc')
plt.savefig("./results/{}_compare_reverse_cut.png".format(model_path1))
plt.close()