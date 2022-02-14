import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--arch', default='resnet18', type=str, help='The backbone of encoder')
parser.add_argument('--local', default='', type=str, help='Run on dev node.')
parser.add_argument('--job_id', default='local', type=str, help='job_id')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--data_name', default='cifar10_1024_4class', type=str, help='The backbone of encoder')
parser.add_argument('--not_shuffle_train_data', action='store_true', default=False)
parser.add_argument('--train_data_drop_last', action='store_true', default=False)
parser.add_argument('--use_out_reorder', action='store_true', default=False)
parser.add_argument('--reorder_reverse', action='store_true', default=False)
parser.add_argument('--half_batch', action='store_true', default=False)
parser.add_argument('--train_mode', default='normal', type=str, choices=['normal', 'inst_suppress', 'curriculum'], help='What samples to plot')
parser.add_argument('--curriculum', default='', type=str, choices=['', 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT', 'DBindex_cluster_GT_org_sample_only', 'mass_candidate', 'mass_candidate_replacement'], help='How to reorder curriculum learning.')
parser.add_argument('--mass_candidate', default='', type=str, choices=['', 'mass_candidate', 'mass_candidate_replacement'], help='How to')
parser.add_argument('--curriculum_scheduler', default='0_0.5_1', type=str, choices=['', '0_0.5_1', '0_1_1'], help='curriculum_scheduler')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--piermaro_whole_epoch', default='', type=str, help='Whole epoch when use re_job to train')
parser.add_argument('--piermaro_restart_epoch', default=0, type=int, help='The order of epoch when use re_job to train')
parser.add_argument('--start_batch_num_ratio', default=0, type=float, help='start_batch_num_ratio')
parser.add_argument('--DBindex_use_org_sample', action='store_true', default=False)
parser.add_argument('--my_train_loader', action='store_true', default=False)
parser.add_argument('--shuffle_new_batch_list', action='store_true', default=False)
parser.add_argument('--all_in_flag', action='store_true', default=False)
parser.add_argument('--random_last_3batch', action='store_true', default=False)
# parser.add_argument('--train_data_drop_last', action='store_true', default=False)

# args parse
args = parser.parse_args()

flag_shuffle_train_data = not args.not_shuffle_train_data

import os
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
from sklearn import metrics

import utils
from model import Model
from utils import train_diff_transform

from inst_suppress_utils2 import *

import datetime

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')

def train_batch(net, pos_1, pos_2, target, train_optimizer):
    net.train()
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
    pos_1, pos_2 = train_diff_transform(pos_1), train_diff_transform(pos_2)

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    this_batch_size = pos_1.shape[0]
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * this_batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * this_batch_size, -1)

    # compute loss
    pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    return loss.item() * this_batch_size, this_batch_size

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, epochs, ):
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, _, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature, out = net(data)

            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]
            sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / temperature).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)

            pred_labels = pred_scores.argsort(dim=-1, descending=True)
            total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
                                     .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))

    return total_top1 / total_num * 100, total_top5 / total_num * 100

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre):

    if args.my_train_loader:
        batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=flag_shuffle_train_data, drop_last=args.train_data_drop_last)

    if args.load_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('./results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_test_acc_loss': [], 'best_train_loss_acc': [], 'best_train_loss': []}
    if not os.path.exists('results'):
        os.mkdir('results')
    best_test_acc = 0.0
    best_test_acc_loss = 7.1
    best_train_loss = 10
    best_train_loss_acc = 0.0
    for epoch in range(1, epochs + 1):
        net.train()
        total_loss, total_num = 0.0, 0
        if args.my_train_loader:
            train_bar = tqdm(batch_idx_list)
            for batch_idx in train_bar:
                pos_1, target = data_loader.get_batch(batch_idx)
                this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer)

                total_num += this_batch_size
                total_loss += this_loss
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        else:
            train_bar = tqdm(data_loader)
            for pos_1, pos_2, target in train_bar:
                this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer)

                total_num += this_batch_size
                total_loss += this_loss
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        train_loss =  total_loss / total_num

        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if train_loss < best_train_loss:
            best_train_loss_acc = test_acc_1
            best_train_loss = train_loss
            if not args.no_save:
                torch.save(net.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_test_acc:
            best_test_acc = test_acc_1
            best_test_acc_loss = train_loss

        results['best_test_acc'].append(best_test_acc)
        results['best_test_acc_loss'].append(best_test_acc_loss)
        results['best_train_loss'].append(best_train_loss)
        results['best_train_loss_acc'].append(best_train_loss_acc)

        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, args.piermaro_restart_epoch + epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        torch.save(net.state_dict(), 'results/{}_piermaro_model.pth'.format(save_name_pre))
        for key in results.keys():
            length = len(results[key])
            results[key] = results[key][length-10:length]
        data_frame = pd.DataFrame(data=results, index=range(args.piermaro_restart_epoch + epochs - 9, args.piermaro_restart_epoch + epochs + 1))
        data_frame.to_csv('results/{}_statistics_final_10_line.csv'.format(save_name_pre), index_label='epoch')

def train_inst_suppress(net, data_loader, train_optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre):

    batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=flag_shuffle_train_data, drop_last=args.train_data_drop_last)
    # batch_idx_list = get_batch_idx_group(100, 15, shuffle=flag_shuffle_train_data, drop_last=args.train_data_drop_last)

    # for epoch in range(1, epochs + 1):
    #     new_batch_idx_list = reorder_DBindex(net, batch_idx_list, data_loader, args.use_out_reorder)

    #     net.train()
    #     total_loss, total_num = 0.0, 0
    #     train_bar = tqdm(new_batch_idx_list)
    #     for batch_idx in train_bar:
    #         pos_1, target = data_loader.get_batch(batch_idx)
    #         this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer)
    #         total_num += this_batch_size
    #         total_loss += this_loss
    #         train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    # return total_loss / total_num

    if args.load_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('./results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc': [], 'best_acc_loss': []}
    if not os.path.exists('results'):
        os.mkdir('results')
    best_acc = 0.0
    best_acc_loss = 7.1
    for epoch in range(1, epochs + 1):
        new_batch_idx_list = reorder_DBindex(net, batch_idx_list, data_loader, epoch, args.use_out_reorder, args.reorder_reverse, args.half_batch, args.DBindex_use_org_sample)
        net.train()
        total_loss, total_num, train_bar = 0.0, 0, tqdm(new_batch_idx_list)
        for batch_idx in train_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer)

            total_num += this_batch_size
            total_loss += this_loss
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        train_loss =  total_loss / total_num

        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            best_acc_loss = train_loss
            if not args.no_save:
                torch.save(net.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['best_acc'].append(best_acc)
        results['best_acc_loss'].append(best_acc_loss)

        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, args.piermaro_restart_epoch + epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        torch.save(net.state_dict(), 'results/{}_piermaro_model.pth'.format(save_name_pre))
        for key in results.keys():
            length = len(results[key])
            results[key] = results[key][length-10:length]
        data_frame = pd.DataFrame(data=results, index=range(args.piermaro_restart_epoch + epochs - 9, args.piermaro_restart_epoch + epochs + 1))
        data_frame.to_csv('results/{}_statistics_final_10_line.csv'.format(save_name_pre), index_label='epoch')

def curriculum(net, data_loader, train_optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre):

    batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=flag_shuffle_train_data, drop_last=args.train_data_drop_last)

    if args.load_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('./results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_test_acc_loss': [], 'best_train_loss_acc': [], 'best_train_loss': []}
    if not os.path.exists('results'):
        os.mkdir('results')
    best_test_acc = 0.0
    best_test_acc_loss = 10
    best_train_loss = 10
    best_train_loss_acc = 0.0
    if args.piermaro_whole_epoch != '':
        whole_epoch = int(args.piermaro_whole_epoch)
    else:
        whole_epoch = epochs
    all_in_flag = False
    for epoch in range(1, epochs + 1):
        if args.mass_candidate in ["mass_candidate", "mass_candidate_replacement"]:
            new_batch_idx_list = sample_from_mass(net, data_loader, epoch, args.batch_size, args.use_out_reorder, args.reorder_reverse, args.curriculum, args.mass_candidate, all_in_flag, args.random_last_3batch)
        else:
            new_batch_idx_list = reorder_DBindex(net, batch_idx_list, data_loader, epoch, args.use_out_reorder, args.reorder_reverse, args.half_batch, args.curriculum)
        net.train()
        schedule_batch_idx_list = get_scheduler(new_batch_idx_list, args.piermaro_restart_epoch+epoch, whole_epoch, args.start_batch_num_ratio, args.curriculum_scheduler, args.shuffle_new_batch_list)
        if len(schedule_batch_idx_list) == len(new_batch_idx_list) and args.all_in_flag:
            all_in_flag = True
        total_loss, total_num, train_bar = 0.0, 0, tqdm(schedule_batch_idx_list)
        for batch_idx in train_bar:
            pos_1, target = data_loader.get_batch(batch_idx)
            this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer)

            total_num += this_batch_size
            total_loss += this_loss
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        train_loss =  total_loss / total_num

        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, epoch, epochs)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if train_loss < best_train_loss:
            best_train_loss_acc = test_acc_1
            best_train_loss = train_loss
            if not args.no_save:
                torch.save(net.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_test_acc:
            best_test_acc = test_acc_1
            best_test_acc_loss = train_loss

        results['best_test_acc'].append(best_test_acc)
        results['best_test_acc_loss'].append(best_test_acc_loss)
        results['best_train_loss'].append(best_train_loss)
        results['best_train_loss_acc'].append(best_train_loss_acc)

        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, args.piermaro_restart_epoch + epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        torch.save(net.state_dict(), 'results/{}_piermaro_model.pth'.format(save_name_pre))
        for key in results.keys():
            length = len(results[key])
            results[key] = results[key][length-10:length]
        data_frame = pd.DataFrame(data=results, index=range(args.piermaro_restart_epoch + epochs - 9, args.piermaro_restart_epoch + epochs + 1))
        data_frame.to_csv('results/{}_statistics_final_10_line.csv'.format(save_name_pre), index_label='epoch')
        

if __name__ == '__main__':
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    if args.train_mode == "normal":
        if not args.my_train_loader:
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=2, pin_memory=True, drop_last=args.train_data_drop_last)
        else:
            train_loader = ByIndexDataLoader(train_data)
    elif args.train_mode == "inst_suppress" or args.train_mode == "curriculum":
        train_loader = ByIndexDataLoader(train_data)
    memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim, arch=args.arch).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))

    if args.load_model:
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        model.load_state_dict(checkpoints)
        # logger.info("File %s loaded!" % (load_model_path))

    if args.piermaro_whole_epoch == '':
        save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.train_mode, args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, k, batch_size, epochs)
    else:
        if args.load_model:
            save_name_pre = args.load_model_path
            save_name_pre = save_name_pre.replace("_piermaro_model", "").replace("_model", "")
        else:
            save_name_pre = '{}_{}_{}_{}_{}_{}'.format(args.train_mode, args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, k, batch_size, args.piermaro_whole_epoch)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = np.max(train_data.targets) + 1

    if args.train_mode == "normal":
        train(model, train_loader, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre)

    elif args.train_mode == "inst_suppress":
        train_inst_suppress(model, train_loader, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre)

    elif args.train_mode == "curriculum":
        curriculum(model, train_loader, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre)

