import numpy as np
import pickle
import os
from sklearn import metrics
import torch

from thop import profile, clever_format
from model import Model
from utils import train_diff_transform, train_diff_transform_prob, CIFAR10Pair
import utils

from inst_suppress_utils import *

import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
parser.add_argument('--load_model', default="", type=str, help='Feature dim for latent vector')
parser.add_argument('--data_name', default="", type=str, help='Feature dim for latent vector')
parser.add_argument('--save_file_name', default="", type=str, help='Feature dim for latent vector')
args = parser.parse_args()

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')

# print(metrics.davies_bouldin_score(train_data[:,:,0,0], train_targets))

train_data = CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
train_data_loader = ByIndexDataLoader(train_data)
batch_idx = np.arange(1024)
pos_1, targets = train_data_loader.get_batch(batch_idx)
pos_1, targets = pos_1.cuda(), targets.cuda()

# data = sampled_data["test_data"]
# targets = np.array(sampled_data["test_targets"])

model = Model(args.feature_dim, arch=args.arch).cuda()
flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
flops, params = clever_format([flops, params])
print('# Model Params: {} FLOPs: {}'.format(params, flops))

if args.load_model != '':
    load_model_path = './results/{}.pth'.format(args.load_model)
    checkpoints = torch.load(load_model_path, map_location=device)
    model.load_state_dict(checkpoints)
    # logger.info("File %s loaded!" % (load_model_path))

DB_inst_mean = []
DB_cluter_mean = []

repeat_num = 5
targets = targets.repeat((repeat_num, )).detach().cpu().numpy()
inst_label = torch.arange(pos_1.shape[0])
inst_label = inst_label.repeat((repeat_num, )).detach().cpu().numpy()

for i in range(10):
    print(i)
    sample = []
    model.eval()
    for _ in range(repeat_num):
        aug_sample = train_diff_transform(pos_1)
        feature, out = model(aug_sample)
        sample.append(feature)
    sample = torch.cat(sample, dim=0).detach().cpu().numpy()

    DB_inst_mean.append(metrics.davies_bouldin_score(sample, inst_label))
    DB_cluter_mean.append(metrics.davies_bouldin_score(sample, targets))

DB_inst_mean = np.mean(DB_inst_mean)
DB_cluter_mean = np.mean(DB_cluter_mean)

f = open("./temp_results/{}".format(args.save_file_name), "a")
f.write("{}\t{}\t{}\n".format(DB_inst_mean, DB_cluter_mean, DB_inst_mean / DB_cluter_mean))
f.close()
