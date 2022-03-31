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
parser.add_argument('--data_name', default='whole_cifar10', type=str, help='The backbone of encoder')
parser.add_argument('--not_shuffle_train_data', action='store_true', default=False)
parser.add_argument('--train_data_drop_last', action='store_true', default=False)
parser.add_argument('--use_out_reorder', action='store_true', default=False)
parser.add_argument('--use_out_dbindex', action='store_true', default=False)
parser.add_argument('--reorder_reverse', action='store_true', default=False)
parser.add_argument('--half_batch', action='store_true', default=False)
parser.add_argument('--train_mode', default='normal', type=str, choices=['normal', 'inst_suppress', 'curriculum', 'just_plot', 'auto_aug', 'adversarial_training', 'train_dbindex_loss'], help='What samples to plot')
parser.add_argument('--curriculum', default='', type=str, choices=['', 'no', 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_cluster_kmeans', 'DBindex_cluster_momentum_kmeans', 'DBindex_ratio_inst_cluster_GT', 'DBindex_product_inst_cluster_GT', 'DBindex_cluster_GT_org_sample_only', 'mass_candidate', 'mass_candidate_replacement', 'DBindex_cluster_momentum_kmeans_wholeset', 'DBindex_cluster_momentum_kmeans_repeat_v2', 'DBindex_cluster_momentum_kmeans_repeat_v2_weighted_cluster', 'DBindex_cluster_momentum_kmeans_repeat_v2_mean_dbindex'], help='How to reorder curriculum learning.')
parser.add_argument('--mass_candidate', default='', type=str, choices=['', 'mass_candidate', 'mass_candidate_replacement'], help='How to')
parser.add_argument('--curriculum_scheduler', default='0_0.5_1', type=str, choices=['', '0_0.5_1', '0_1_1'], help='curriculum_scheduler')
parser.add_argument('--load_piermaro_model', action='store_true', default=False)
parser.add_argument('--load_piermaro_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str, help='Path to load model.')
parser.add_argument('--piermaro_whole_epoch', default='', type=str, help='Whole epoch when use re_job to train')
parser.add_argument('--piermaro_restart_epoch', default=0, type=int, help='The order of epoch when use re_job to train')
parser.add_argument('--start_batch_num_ratio', default=0, type=float, help='start_batch_num_ratio')
parser.add_argument('--DBindex_use_org_sample', action='store_true', default=False)
parser.add_argument('--my_train_loader', action='store_true', default=False)
parser.add_argument('--my_test_loader', action='store_true', default=False)
parser.add_argument('--shuffle_new_batch_list', action='store_true', default=False)
parser.add_argument('--all_in_flag', action='store_true', default=False)
parser.add_argument('--random_last_3batch', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--pretrain_model_path', default='', type=str, help='Use pretrained model to get DBindex')
parser.add_argument('--candidate_pool_size', default=10, type=int, help='candidate_pool_size')
parser.add_argument('--change_batch_step', default=1, type=int, help='change_batch_step')
parser.add_argument('--drop_last_new_batch', default=0, type=int, help='change_batch_step')
parser.add_argument('--inst_suppress_sheduler_gap', default=100, type=int, help='change_batch_step')
parser.add_argument('--augmentation_prob', default=[0, 0, 0, 0], nargs='+', type=float, help='get augmentation by probility')
parser.add_argument('--save_aug_file_name', default='temp.txt', type=str, help='save_aug_file_name')
parser.add_argument('--color_jitter_strength', default=1, type=float, help='change_batch_step')
parser.add_argument('--ifm_epsilon', default=0.0, type=float, help='ifm_epsilon')
parser.add_argument('--attack_epsilon', default=8, type=float, help='attack_epsilon')
parser.add_argument('--attack_alpha', default=0.8, type=float, help='attack_alpha')
parser.add_argument('--weight_dbindex_loss', default=0.0, type=float, help='attack_alpha')
parser.add_argument('--attack_type', default='linf', type=str, help='attack_type')
parser.add_argument('--perturb_batchsize', default=0, type=int, help='perturb_batchsize')
parser.add_argument('--repeat_num', default=1, type=int, help='perturb_batchsize')
parser.add_argument('--start_dbindex_loss_epoch', default=0, type=int, help='perturb_batchsize')
parser.add_argument('--num_clusters', default=[0], nargs='+', type=int, help='get augmentation by probility')
parser.add_argument('--plot_n_cluster', default=[0], nargs='+', type=int, help='get augmentation by probility')
parser.add_argument('--m', default=0.999, type=float, help='momentum of momentum_encoder')
parser.add_argument('--restore_k_when_start', action='store_true', default=False)
parser.add_argument('--kmeans_just_plot', action='store_true', default=False)
parser.add_argument('--kmeans_plot', action='store_true', default=False)
parser.add_argument('--load_momentum_model', action='store_true', default=False)
parser.add_argument('--kmeans_just_plot_test', action='store_true', default=False)
parser.add_argument('--kornia_transform', action='store_true', default=False)
parser.add_argument('--restore_best_test_acc_model', action='store_true', default=False)
parser.add_argument('--check_every_step', action='store_true', default=False)
parser.add_argument('--use_sim', action='store_true', default=False)
parser.add_argument('--use_wholeset_centroid', action='store_true', default=False)
parser.add_argument('--use_mean_dbindex', action='store_true', default=False)
parser.add_argument('--use_out_kmeans', action='store_true', default=False)
parser.add_argument('--use_org_sample_dbindex', action='store_true', default=False)
parser.add_argument('--flag_select_confidence', action='store_true', default=False)
parser.add_argument('--reassign_step', default=1, type=int, help='gap of reassign labels')
parser.add_argument('--confidence_thre', default=0.3, type=float, help='gap of reassign labels')
parser.add_argument('--dataset', default='cifar10', type=str, help='what dataset to use')
parser.add_argument('--final_high_conf_percent', default=0.5, type=float, help='gap of reassign labels')
parser.add_argument('--use_GT', action='store_true', default=False)
parser.add_argument('--keep_gradient_on_center', action='store_true', default=False)

parser.add_argument('--attack_steps', default=10, type=int, help='perturb number of steps')

# parser.add_argument('--train_data_drop_last', action='store_true', default=False)

# args parse
args = parser.parse_args()

flag_shuffle_train_data = not args.not_shuffle_train_data
attack_epsilon = args.attack_epsilon / 255.0
attack_alpha = args.attack_alpha / 255.0
if args.restore_best_test_acc_model:
    raise("WARNING: restore_best_test_acc_model used.")

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

import utils2 as utils
from model import Model, momentum_Model
from utils2 import train_diff_transform, train_diff_transform_prob, run_kmeans

from inst_suppress_utils import *
from auto_aug_utils import *

from attack3 import PGD, get_dbindex_loss
from kmeans_pytorch import kmeans

import datetime

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
else:
    device = torch.device('cpu')

def train_batch(net, pos_1, pos_2, target, train_optimizer, ifm_epsilon, weight_dbindex_loss=0):

    if np.sum(args.augmentation_prob) != 0:
        my_transform = train_diff_transform_prob(*args.augmentation_prob)
    else:
        my_transform = train_diff_transform
    
    net.train()
    pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)

    if weight_dbindex_loss != 0:
        # get_dbindex_loss(net, x, labels, loss_type, reverse, my_transform)
        dbindex_loss = get_dbindex_loss(net, pos_1.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.flag_select_confidence, args.confidence_thre)
    else:
        dbindex_loss = 0

    if args.kornia_transform:
        pos_1, pos_2 = my_transform(pos_1), my_transform(pos_2)

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix_before_exp = torch.mm(out, out.t().contiguous()) + ifm_epsilon
    pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device), torch.eye(pos_1.shape[0], device=pos_1.device)], dim=0)
    pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=pos_1.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device)], dim=0)
    pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1) * 2 * ifm_epsilon
    sim_matrix_before_exp -= pos_den_mask

    sim_matrix = torch.exp(sim_matrix_before_exp / temperature)
    this_batch_size = pos_1.shape[0]
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * this_batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * this_batch_size, -1)

    # compute loss
    pos_sim = torch.exp((torch.sum(out_1 * out_2, dim=-1) - ifm_epsilon) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    simclr_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    loss = -weight_dbindex_loss * dbindex_loss + simclr_loss

    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    return loss.item() * this_batch_size, this_batch_size

def train_batch_kmeans(net, pos_1, pos_2, pos_org, target, train_optimizer, ifm_epsilon, weight_dbindex_loss, kmean_result):

    if np.sum(args.augmentation_prob) != 0:
        my_transform = train_diff_transform_prob(*args.augmentation_prob)
    else:
        my_transform = train_diff_transform
    
    net.train()
    pos_1, pos_2, pos_org = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True), pos_org.cuda(non_blocking=True)

    # if not args.use_org_sample_dbindex:
    #     if args.kornia_transform:
    #         pos_1, pos_2 = my_transform(pos_1), my_transform(pos_2)
    #     if weight_dbindex_loss != 0:
    #         # get_dbindex_loss(net, x, labels, loss_type, reverse, my_transform)
    #         dbindex_loss_pos_1 = get_dbindex_loss(net, pos_1.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex)
    #         dbindex_loss_pos_2 = get_dbindex_loss(net, pos_2.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex)
    #         dbindex_loss = (dbindex_loss_pos_1 + dbindex_loss_pos_2) / 2
    #     else:
    #         dbindex_loss = 0
    # else:
    #     # raise('transform order wrong. Please use following')
    #     if weight_dbindex_loss != 0:
    #         dbindex_loss = get_dbindex_loss(net, pos_1.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex, args.flag_select_confidence)
    #     else:
    #         dbindex_loss = 0
    #     if args.kornia_transform:
    #         pos_1, pos_2 = my_transform(pos_1), my_transform(pos_2)

    if not args.use_org_sample_dbindex:
        if weight_dbindex_loss != 0:
            # get_dbindex_loss(net, x, labels, loss_type, reverse, my_transform)
            dbindex_loss_pos_1 = get_dbindex_loss(net, pos_1.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex)
            dbindex_loss_pos_2 = get_dbindex_loss(net, pos_2.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex)
            dbindex_loss = (dbindex_loss_pos_1 + dbindex_loss_pos_2) / 2
        else:
            dbindex_loss = 0
    else:
        # raise('transform order wrong. Please use following')
        if weight_dbindex_loss != 0:
            dbindex_loss = get_dbindex_loss(net, pos_org.clone(), target, args.curriculum, args.reorder_reverse, my_transform, args.num_clusters, args.repeat_num, args.use_out_dbindex, args.use_sim, kmean_result, args.use_wholeset_centroid, args.use_mean_dbindex, args.flag_select_confidence, args.confidence_thre, args.keep_gradient_on_center)
        else:
            dbindex_loss = 0

    feature_1, out_1 = net(pos_1)
    feature_2, out_2 = net(pos_2)
    # print(out_1.shape)
    # input()
    # [2*B, D]
    out = torch.cat([out_1, out_2], dim=0)
    # [2*B, 2*B]
    sim_matrix_before_exp = torch.mm(out, out.t().contiguous()) # + ifm_epsilon
    # pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device), torch.eye(pos_1.shape[0], device=pos_1.device)], dim=0)
    # pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=pos_1.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device)], dim=0)
    # pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1) * 2 * ifm_epsilon
    # sim_matrix_before_exp -= pos_den_mask

    sim_matrix = torch.exp(sim_matrix_before_exp / temperature)
    this_batch_size = pos_1.shape[0]
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * this_batch_size, device=sim_matrix.device)).bool()
    # [2*B, 2*B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * this_batch_size, -1)

    # compute loss
    pos_sim = torch.exp((torch.sum(out_1 * out_2, dim=-1) - ifm_epsilon) / temperature)
    # [2*B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    simclr_loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

    loss = -weight_dbindex_loss * dbindex_loss + simclr_loss

    train_optimizer.zero_grad()
    loss.backward()
    train_optimizer.step()

    return loss.item() * this_batch_size, this_batch_size

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch, epochs, ):
    if args.my_test_loader:
        memory_batch_idx_list = get_batch_idx_group(memory_data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=False, drop_last=False)
        test_batch_idx_list = get_batch_idx_group(test_data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=False, drop_last=False)
    net.eval()
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        if args.my_test_loader:
            for batch_idx in tqdm(memory_batch_idx_list, desc='Feature extracting'):
                data, target = memory_data_loader.get_batch(batch_idx)
                feature, out = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(memory_data_loader.data_source.targets, device=feature_bank.device)
        else:
            for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
                feature, out = net(data.cuda(non_blocking=True))
                feature_bank.append(feature)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # # [D, N]
        # feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # loop test data to predict the label by weighted knn search
        if args.my_test_loader:
            test_bar = tqdm(test_batch_idx_list)
            for batch_id in test_bar:
                data, target = test_data_loader.get_batch(batch_id)
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
        else:
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

    if args.load_piermaro_model and args.piermaro_whole_epoch != '':
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
        if args.my_train_loader:
            batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=args.batch_size, shuffle=flag_shuffle_train_data, drop_last=args.train_data_drop_last)
        net.train()
        total_loss, total_num = 0.0, 0
        if args.my_train_loader:
            raise('Please use pytorch loader!')
            train_bar = tqdm(batch_idx_list)
            for batch_idx in train_bar:
                pos_1, target = data_loader.get_batch(batch_idx)
                this_loss, this_batch_size = train_batch(net, pos_1, pos_1, target, train_optimizer, args.ifm_epsilon)

                total_num += this_batch_size
                total_loss += this_loss
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        else:
            train_bar = tqdm(data_loader)
            for pos_1, pos_2, pos_org, target in train_bar:
                this_loss, this_batch_size = train_batch(net, pos_1, pos_2, target, train_optimizer, args.ifm_epsilon)

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
                state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
                torch.save(state, 'results/{}_model.pth'.format(save_name_pre))
                # torch.save(net.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_test_acc:
            best_test_acc = test_acc_1
            best_test_acc_loss = train_loss
            if not args.no_save:
                state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
                torch.save(state, 'results/{}_best_test_acc_model.pth'.format(save_name_pre))

        results['best_test_acc'].append(best_test_acc)
        results['best_test_acc_loss'].append(best_test_acc_loss)
        results['best_train_loss'].append(best_train_loss)
        results['best_train_loss_acc'].append(best_train_loss_acc)

        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, args.piermaro_restart_epoch + epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
        torch.save(state, 'results/{}_piermaro_model.pth'.format(save_name_pre))
        for key in results.keys():
            length = len(results[key])
            results[key] = results[key][length-10:length]
        data_frame = pd.DataFrame(data=results, index=range(args.piermaro_restart_epoch + epochs - 9, args.piermaro_restart_epoch + epochs + 1))
        data_frame.to_csv('results/{}_statistics_final_10_line.csv'.format(save_name_pre), index_label='epoch')
        utils.plot_loss('results/{}_statistics'.format(save_name_pre))


def just_plot(net, data_loader, train_optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre):
    utils.plot_loss('results/{}_statistics'.format(args.load_model_path.replace("_model", "")))

def train_dbindex_loss_pytorch_loader(net, data_loader, train_optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre, train_data, const_train_data):

    if args.load_piermaro_model and args.piermaro_whole_epoch != '':
        results = pd.read_csv('./results/{}_statistics.csv'.format(save_name_pre), index_col='epoch').to_dict()
        for key in results.keys():
            load_list = []
            for i in range(len(results[key])):
                load_list.append(results[key][i+1])
            results[key] = load_list
        best_test_acc = results['best_test_acc'][len(results['best_test_acc'])-1]
        best_train_loss = results['best_train_loss'][len(results['best_train_loss'])-1]
        best_test_acc_loss = results['best_test_acc_loss'][len(results['best_test_acc_loss'])-1]
        best_train_loss_acc = results['best_train_loss_acc'][len(results['best_train_loss_acc'])-1]
    else:
        results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_test_acc': [], 'best_test_acc_loss': [], 'best_train_loss_acc': [], 'best_train_loss': [], 'GT_dbindex': []}
        best_test_acc = 0.0
        best_train_loss = 10
        best_test_acc_loss = 7.1
        best_train_loss_acc = 0.0
    if not os.path.exists('results'):
        os.mkdir('results')
    if not args.check_every_step:
        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, 0, epochs)
    last_dbindex = 0.0
    for epoch in range(1, epochs + 1):
        # print(test_acc_1, test_acc_5)
        net.train()
        total_loss, total_num = 0.0, 0
        if args.restore_k_when_start and args.piermaro_restart_epoch + epoch == args.start_dbindex_loss_epoch and args.curriculum in ['DBindex_cluster_momentum_kmeans', 'DBindex_cluster_momentum_kmeans_wholeset', 'DBindex_cluster_momentum_kmeans_repeat_v2']:
            if args.restore_best_test_acc_model:
                best_acc_model_path = 'results/{}_best_test_acc_model.pth'.format(save_name_pre)
                checkpoints = torch.load(best_acc_model_path, map_location=device)
                net.load_state_dict(checkpoints)
            else:
                net.restore_k_with_q()
                print('start train dbindex restore_k_with_q done')
        if args.my_train_loader:
            raise("Please use train_dbindex_loss().")
        else:
            if args.kmeans_plot:
                plot_kmeans_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
                plot_kmeans_labels_list = []
                plot_kmeans_feature = []
                plot_GT_label = []
                net.train()
                # for kmeans_batch_idx in kmeans_batch_idx_list:
                for plot_pos_1, plot_pos_2, plot_pos_org, plot_target in plot_kmeans_loader:
                    # pos_1, target = data_loader.get_batch(kmeans_batch_idx)
                    plot_pos_org, plot_target = plot_pos_org.cuda(), plot_target.cuda()
                    plot_feature, plot_out = net.momentum_encoder(plot_pos_org)
                    plot_kmeans_feature.append(plot_feature)
                    plot_GT_label.append(plot_target.detach().cpu().numpy())
                    if len(plot_GT_label) >= 4:
                        break
                plot_kmeans_feature = torch.cat(plot_kmeans_feature, dim=0)
                plot_GT_label = np.concatenate(plot_GT_label, axis=0)
                plot_kmean_result = run_kmeans(plot_kmeans_feature.detach().cpu().numpy(), [10] + args.num_clusters, 0, temperature)
                utils.plot_kmeans_epoch(plot_kmeans_feature, plot_GT_label, save_name_pre, plot_kmean_result['im2cluster'], 0, len(plot_kmeans_feature))
                print(save_name_pre)
                
            if args.curriculum == 'DBindex_cluster_momentum_kmeans_wholeset' and args.piermaro_restart_epoch + epoch >= args.start_dbindex_loss_epoch:
                if args.piermaro_restart_epoch + epoch == args.start_dbindex_loss_epoch or (args.piermaro_restart_epoch + epoch) % args.reassign_step == 0:
                    kmeans_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
                    kmeans_labels_list = []
                    cluster_centers_list = []
                    kmeans_feature = []
                    GT_label = []
                    sample_feature = []
                    net.train()
                    for pos_1, pos_2, pos_org, target in kmeans_loader:
                        pos_org, target = pos_org.cuda(), target.cuda()
                        feature_momentum, out_momentum = net.momentum_encoder(pos_org)
                        if args.use_out_kmeans:
                            kmeans_feature.append(out_momentum)
                        else:
                            kmeans_feature.append(feature_momentum)
                        GT_label.append(target.detach().cpu().numpy())

                    with torch.no_grad():
                        for pos_1, pos_2, pos_org, target in kmeans_loader:
                            pos_org, target = pos_org.cuda(), target.cuda()
                            feature, out = net(pos_org)
                            if args.use_out_kmeans:
                                sample_feature.append(out)
                            else:
                                sample_feature.append(feature)
                    
                    kmeans_feature = torch.cat(kmeans_feature, dim=0)
                    sample_feature = torch.cat(sample_feature, dim=0)
                    GT_label = np.concatenate(GT_label, axis=0)

                    torch.cuda.empty_cache()

                    kmean_result = run_kmeans(kmeans_feature.detach().cpu().numpy(), args.num_clusters, 0, temperature)
                    # sample_feature_kmean_result = run_kmeans(sample_feature.detach().cpu().numpy(), args.num_clusters, 0, temperature)
                    
                    if args.use_GT:
                        kmean_result['im2cluster'] = [torch.LongTensor(GT_label).cuda()]
                        args.num_clusters = [10]
                    sample_feature_centroids = utils.get_center(sample_feature.detach().cpu().numpy(), kmean_result, args.num_clusters, args.curriculum)
                    kmean_result['centroids'] = sample_feature_centroids
                    
                    high_conf_label_list = utils.get_reorder_point_2_center_diss(sample_feature.detach().cpu().numpy(), kmean_result, args.num_clusters, args.start_dbindex_loss_epoch, epoch + args.piermaro_restart_epoch, epochs, args.piermaro_whole_epoch, args.final_high_conf_percent)
                    
                    kmeans_labels_list = kmean_result['im2cluster'] + high_conf_label_list
                    kmeans_label_np = torch.stack(kmeans_labels_list, dim=1).detach().cpu().numpy()
                    train_data.targets = kmeans_label_np

                    for db_label in kmeans_labels_list:
                        # dbindex = metrics.davies_bouldin_score(sample_feature.detach().cpu().numpy(), db_label.detach().cpu().numpy())
                        dbindex = utils.get_np_mean_dbindex(sample_feature.detach().cpu().numpy(), db_label.detach().cpu().numpy())
                        # print(dbindex)
                    dbindex = metrics.davies_bouldin_score(sample_feature.detach().cpu().numpy(), GT_label)
                    print(dbindex)
                    dbindex = utils.get_np_mean_dbindex(sample_feature.detach().cpu().numpy(), GT_label)
                    print(dbindex)
                    results['GT_dbindex'].append(dbindex)
                    plot_num = 1024 * 3
                    
                    if args.kmeans_plot:
                        break

                    del kmeans_feature
                    del sample_feature
                    torch.cuda.empty_cache()
                else:
                    results['GT_dbindex'].append(dbindex)
            else:
                if (args.piermaro_restart_epoch + epoch) % 10 == 1:
                    kmeans_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
                    kmeans_feature = []
                    GT_label = []
                    net.eval()
                    with torch.no_grad():
                        for pos_1, pos_2, pos_org, target in kmeans_loader:
                            pos_org, target = pos_org.cuda(), target.cuda()
                            feature, out = net(pos_org)
                            if args.use_out_kmeans:
                                kmeans_feature.append(out)
                            else:
                                kmeans_feature.append(feature)
                            GT_label.append(target.detach().cpu().numpy())
                    
                    kmeans_feature = torch.cat(kmeans_feature, dim=0)
                    GT_label = np.concatenate(GT_label, axis=0)

                    dbindex = metrics.davies_bouldin_score(kmeans_feature.detach().cpu().numpy(), GT_label)
                    print(dbindex)
                    dbindex = utils.get_np_mean_dbindex(kmeans_feature.detach().cpu().numpy(), GT_label)
                    print(dbindex)
                    results['GT_dbindex'].append(dbindex)
                    last_dbindex = dbindex

                    del kmeans_feature
                    torch.cuda.empty_cache()
                else:
                    results['GT_dbindex'].append(last_dbindex)
                    # print(results['GT_dbindex'])

            if args.curriculum == 'DBindex_cluster_GT':
                kmeans_loader = DataLoader(const_train_data, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)
                sample_feature = []
                GT_label = []

                with torch.no_grad():
                    for pos_1, pos_2, pos_org, target in kmeans_loader:
                        pos_org, target = pos_org.cuda(), target.cuda()
                        feature, out = net(pos_org)
                        if args.use_out_kmeans:
                            sample_feature.append(out)
                        else:
                            sample_feature.append(feature)
                        GT_label.append(target.detach().cpu().numpy())
                
                sample_feature = torch.cat(sample_feature, dim=0)
                GT_label = np.concatenate(GT_label, axis=0)

                torch.cuda.empty_cache()
                sample_feature_centroids = utils.get_center(sample_feature.detach().cpu().numpy(), GT_label, args.num_clusters, args.curriculum)
                kmean_result = {'centroids':sample_feature_centroids}
                    
            train_bar = tqdm(data_loader)
            for pos_1, pos_2, pos_org, target in train_bar:
                if args.piermaro_restart_epoch + epoch >= args.start_dbindex_loss_epoch:
                    if args.check_every_step:
                        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, epoch, epochs)
                    this_loss, this_batch_size = train_batch_kmeans(net, pos_1, pos_2, pos_org, target, train_optimizer, args.ifm_epsilon, args.weight_dbindex_loss, kmean_result)
                else:
                    this_loss, this_batch_size = train_batch(net, pos_1, pos_2, target, train_optimizer, args.ifm_epsilon, 0)
                if args.curriculum in ['DBindex_cluster_momentum_kmeans', 'DBindex_cluster_momentum_kmeans_wholeset', 'DBindex_cluster_momentum_kmeans_repeat_v2', 'DBindex_cluster_GT']:
                    net._momentum_update_key_encoder()

                total_num += this_batch_size
                total_loss += this_loss
                train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        train_loss =  total_loss / total_num

        results['train_loss'].append(train_loss)
        test_acc_1, test_acc_5 = test(net, memory_loader, test_loader, epoch, epochs)
        # print(test_acc_1, test_acc_5)
        # input()
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        if train_loss < best_train_loss:
            best_train_loss_acc = test_acc_1
            best_train_loss = train_loss
            if not args.no_save:
                state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
                torch.save(state, 'results/{}_model.pth'.format(save_name_pre))
        if test_acc_1 > best_test_acc:
            best_test_acc = test_acc_1
            best_test_acc_loss = train_loss
            if not args.no_save:
                state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
                torch.save(state, 'results/{}_best_test_acc_model.pth'.format(save_name_pre))

        results['best_test_acc'].append(best_test_acc)
        results['best_test_acc_loss'].append(best_test_acc_loss)
        results['best_train_loss'].append(best_train_loss)
        results['best_train_loss_acc'].append(best_train_loss_acc)

        if not args.no_save:
            # save statistics
            data_frame = pd.DataFrame(data=results, index=range(1, args.piermaro_restart_epoch + epoch + 1))
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

    if not args.no_save:
        state = {'epoch': args.piermaro_restart_epoch + epochs, 'state_dict': net.state_dict(), 'optimizer': train_optimizer.state_dict()}
        torch.save(state, 'results/{}_piermaro_model.pth'.format(save_name_pre))
        for key in results.keys():
            length = len(results[key])
            results[key] = results[key][length-10:length]
        data_frame = pd.DataFrame(data=results, index=range(args.piermaro_restart_epoch + epochs - 9, args.piermaro_restart_epoch + epochs + 1))
        data_frame.to_csv('results/{}_statistics_final_10_line.csv'.format(save_name_pre), index_label='epoch')
        utils.plot_loss('results/{}_statistics'.format(save_name_pre))
        

if __name__ == '__main__':
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    if args.data_name != 'whole_cifar10' and args.train_mode not in ['normal', 'train_dbindex_loss']:
        raise("Please use whole_cifar10")
    if args.dataset == 'cifar10':
        train_data = utils.CIFAR10Triple(root='data', train=True, transform=utils.train_transform, download=True, data_name=args.data_name)
        const_train_data = utils.CIFAR10Triple(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    elif args.dataset == 'cifar100':
        train_data = utils.CIFAR100Triple(root='data', train=True, transform=utils.train_transform, download=True, data_name=args.data_name)
        const_train_data = utils.CIFAR100Triple(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    else:
        raise('Wrong dataset!')
    if args.train_mode in ["normal", "train_dbindex_loss"]:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=flag_shuffle_train_data, num_workers=4, pin_memory=True, drop_last=args.train_data_drop_last)
    else:
        train_loader = ByIndexDataLoader(train_data)
    if args.dataset == 'cifar10':
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    elif args.dataset == 'cifar100':
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.ToTensor_transform, download=True, data_name=args.data_name)
    else:
        raise('Wrong dataset!')
    memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    if args.curriculum not in ['DBindex_cluster_momentum_kmeans', 'DBindex_cluster_momentum_kmeans_wholeset', 'DBindex_cluster_momentum_kmeans_repeat_v2', 'DBindex_cluster_momentum_kmeans_repeat_v2_mean_dbindex', 'DBindex_cluster_momentum_kmeans_repeat_v2_weighted_cluster', 'DBindex_cluster_GT']:
        model = Model(feature_dim, arch=args.arch).cuda()
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        flops, params = clever_format([flops, params])
        print('# Model Params: {} FLOPs: {}'.format(params, flops))
    else:
        model = momentum_Model(feature_dim, arch=args.arch, m=args.m).cuda()
        flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        flops, params = clever_format([flops, params])
        print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    if args.load_model:
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        if 'epoch' in checkpoints and 'state_dict' in checkpoints and 'optimizer' in checkpoints:
            try:
                model.load_state_dict(checkpoints['state_dict'])
                optimizer.load_state_dict(checkpoints['optimizer'])
                print('load_momentum_model done')
            except:
                model.model.load_state_dict(checkpoints['state_dict'])
                model.key_model.load_state_dict(checkpoints['state_dict'])
                optimizer.load_state_dict(checkpoints['optimizer'])
                print('load_single_model done')
                raise('check why comes to load optimizre here')
        else:
            try:
                model.load_state_dict(checkpoints)
                # logger.info("File %s loaded!" % (load_model_path))
            except:
                model.model.load_state_dict(checkpoints)
                model.key_model.load_state_dict(checkpoints)
                # logger.info("File %s loaded!" % (load_model_path))

    if args.pretrain_model_path != '': # This is to use pre-trained model to get DBindex instead of warm start model.
        pretrain_model = Model(feature_dim, arch=args.arch).cuda()
        flops, params = profile(pretrain_model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
        flops, params = clever_format([flops, params])
        print('# Pretrain-Model Params: {} FLOPs: {}'.format(params, flops))
        pretrain_model_path = './results/{}.pth'.format(args.pretrain_model_path)
        checkpoints = torch.load(pretrain_model_path, map_location=device)
        if 'epoch' in checkpoints and 'state_dict' in checkpoints and 'optimizer' in checkpoints:
            pretrain_model.load_state_dict(checkpoints['state_dict'])
        else:
            pretrain_model.load_state_dict(checkpoints)
    else:
        pretrain_model = None

    if args.load_piermaro_model:
        load_model_path = './results/{}.pth'.format(args.load_piermaro_model_path)
        checkpoints = torch.load(load_model_path, map_location=device)
        if 'epoch' in checkpoints and 'state_dict' in checkpoints and 'optimizer' in checkpoints:
            try:
                model.load_state_dict(checkpoints['state_dict'])
                optimizer.load_state_dict(checkpoints['optimizer'])
            except:
                model.model.load_state_dict(checkpoints['state_dict'])
                model.key_model.load_state_dict(checkpoints['state_dict'])
                optimizer.load_state_dict(checkpoints['optimizer'])
        else:
            try:
                model.load_state_dict(checkpoints)
            except:
                model.model.load_state_dict(checkpoints)
                model.key_model.load_state_dict(checkpoints)

        del checkpoints
        torch.cuda.empty_cache()

    if args.piermaro_whole_epoch == '':
        save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(args.train_mode, args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, k, batch_size, epochs)
    else:
        if args.load_piermaro_model:
            save_name_pre = args.load_piermaro_model_path
            save_name_pre = save_name_pre.replace("_piermaro_model", "").replace("_model", "")
        else:
            save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(args.train_mode, args.job_id, datetime.datetime.now().strftime("%Y%m%d%H%M%S"), temperature, k, batch_size, args.piermaro_whole_epoch)

    c = np.max(train_data.targets) + 1

    if args.train_mode == "normal":
        train(model, train_loader, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre)

    elif args.train_mode == "just_plot":
        just_plot(model, None, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre)

    elif args.train_mode == "train_dbindex_loss":
        train_dbindex_loss_pytorch_loader(model, train_loader, optimizer, memory_loader, test_loader, temperature, k, batch_size, epochs, save_name_pre, train_data, const_train_data)
