import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--model_path1', default='', type=str, help='Use pretrained model to get DBindex')
parser.add_argument('--model_path2', default='', type=str, help='Use pretrained model to get DBindex')
parser.add_argument('--model_path1_base', default='', type=str, help='Use pretrained model to get DBindex')
parser.add_argument('--start_epoch', default=0, type=int, help='Use pretrained model to get DBindex')

parser.add_argument('--plot_dbindex_acc', action='store_true', default=False)
parser.add_argument('--plot_dbindex_loss', action='store_true', default=False)

# args parse
args = parser.parse_args()

start_epoch = args.start_epoch

model_path1 = args.model_path1.replace("_model", "")
model_path2 = args.model_path2.replace("_model", "")
model_path1_base = args.model_path1_base.replace("_model", "")

pd_reader = pd.read_csv("./results/{}_statistics.csv".format(model_path1))
epoch = pd_reader.values[:,0]
# loss = pd_reader.values[:,1]
# acc = pd_reader.values[:,2]
# acc_top5 = pd_reader.values[:,3]
# best_test_acc = pd_reader.values[:,4]
# best_test_acc_loss = pd_reader.values[:,5]
# best_train_loss_acc = pd_reader.values[:,6]
# best_train_loss = pd_reader.values[:,7]

# epoch,train_loss,test_acc@1,test_acc@5,best_test_acc,best_test_acc_loss,best_train_loss_acc,best_train_loss,GT_dbindex

loss1 = pd_reader.values[:,1]
acc1 = pd_reader.values[:,2]
dbindex1 = pd_reader.values[:,8]

if args.model_path1_base != '':
    pd_reader = pd.read_csv("./results/{}_statistics.csv".format(model_path1_base))
    epoch_base = pd_reader.values[:,0]
    loss1_base = pd_reader.values[:,1]
    acc1_base = pd_reader.values[:,2]
    try:
        dbindex1_base = pd_reader.values[:,8]
    except:
        print('no dbindex in base')
        pass

# print(loss1_base.shape)
# print(type(loss1_base))
# print(loss1.shape)
# print(type(acc1_base))
# input()

    loss1 = np.concatenate([loss1_base, loss1])
    acc1 = np.concatenate([acc1_base, acc1])

if args.model_path2 != '':
    pd_reader = pd.read_csv("./results/{}_statistics.csv".format(model_path2))
    loss2 = pd_reader.values[:,1]
    acc2 = pd_reader.values[:,2]
    # dbindex2 = pd_reader.values[:,8]

fig, ax=plt.subplots(1,1,figsize=(9,6))
ax1 = ax.twinx()

if args.model_path2 != '':
    cut = min(len(loss1), len(loss2))
else:
    cut = len(loss1)
epoch = np.arange(1, cut+1)
# if args.model_path2 != '':
#     p3 = ax.plot(epoch[:cut], loss2[:cut], 'b-', label = 'loss2')
# ax.legend()
if args.plot_dbindex_loss:
    p2 = ax.plot(epoch[start_epoch:cut], loss1[start_epoch:cut],'r-', label = 'loss1')
    p3 = ax1.plot(epoch[start_epoch:cut], dbindex1[start_epoch:cut], 'b-', label = 'test_acc1')
    # p3 = ax.plot(epoch[start_epoch:cut], dbindex1[start_epoch:cut],'g-', label = 'loss1')

if args.plot_dbindex_acc:
    p3 = ax1.plot(epoch[start_epoch:cut], acc1[start_epoch:cut], 'b-', label = 'test_acc1')
    p3 = ax.plot(epoch[start_epoch:cut], dbindex1[start_epoch:cut],'g-', label = 'loss1')

if not args.plot_dbindex_loss and not args.plot_dbindex_acc:
    # p2 = ax.plot(epoch[start_epoch:cut], loss1[start_epoch:cut],'r-', label = 'loss1')
    p3 = ax1.plot(epoch[start_epoch:cut], acc1[start_epoch:cut], 'b-', label = 'test_acc1')


if args.model_path2 != '':
    p3 = ax1.plot(epoch[start_epoch:cut], acc2[start_epoch:cut], 'g-', label = 'test_acc2')
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

# fig, ax=plt.subplots(1,1,figsize=(9,6))
# ax1 = ax.twinx()

# cut = 200

# p2 = ax.plot(epoch[:cut], loss1[:cut],'r-', label = 'loss1')
# if args.model_path2 != '':
#     p3 = ax.plot(epoch[:cut], loss2[:cut], 'b-', label = 'loss2')
# ax.legend()
# p2 = ax1.plot(epoch[:cut], acc1[:cut], 'r-', label = 'test_acc1')
# if args.model_path2 != '':
#     p3 = ax1.plot(epoch[:cut], acc2[:cut], 'b-', label = 'test_acc2')
# ax1.legend()

# #显示图例
# # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
# # plt.legend()
# ax.set_xlabel('epoch')
# ax.set_ylabel('loss')
# ax1.set_ylabel('acc')
# plt.title('Training loss and acc')
# plt.savefig("./results/{}_compare_reverse_cut.png".format(model_path1))
# plt.close()