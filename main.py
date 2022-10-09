import argparse

parser = argparse.ArgumentParser(description='Train SimCLR')
parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
parser.add_argument('--epochs', default=1000, type=int, help='Number of sweeps over the dataset to train')
parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--method', default='vanilla', type=str)
parser.add_argument('--neg', default='gt_same_diff_label', type=str)
parser.add_argument('--local', default='', type=str)
parser.add_argument('--job_id', default='local', type=str)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--load_model_path', default='', type=str)
parser.add_argument('--plot_feature', action='store_true', default=False)
parser.add_argument('--plot_save_name', default='', type=str)
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--seed', default=0, type=int)

# args parse
args = parser.parse_args()

import os
if args.local != '':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.local

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from model import Model

import numpy as np
import random
import time

if torch.cuda.is_available():
    # torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def vanilla(epochs, model, optimizer, train_loader, memory_loader, test_loader):
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc@1': [],}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['train_loss'].append(train_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['best_acc@1'].append(best_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))


def decoupled(epochs, model, optimizer, train_loader, memory_loader, test_loader):
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc@1': [],}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_decoupled_batch(model, train_loader, optimizer, epoch)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['train_loss'].append(train_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['best_acc@1'].append(best_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))

# train for one epoch to learn unique features
def train_decoupled_batch(net, data_loader, train_optimizer, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        pos_den_mask1 = torch.cat([torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device), torch.eye(pos_1.shape[0], device=pos_1.device)], dim=0)
        pos_den_mask2 = torch.cat([torch.eye(pos_1.shape[0], device=pos_1.device), torch.zeros((pos_1.shape[0], pos_1.shape[0]), device=pos_1.device)], dim=0)
        pos_den_mask = torch.cat([pos_den_mask1, pos_den_mask2], dim=1)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device) - pos_den_mask).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

def cluster(epochs, model, optimizer, train_loader, memory_loader, test_loader, neg_mode):
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': [], 'best_acc@1': [],}
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train_cluster_batch(model, train_loader, optimizer, epoch, neg_mode)
        test_acc_1, test_acc_5 = test(model, memory_loader, test_loader, epoch)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            if not args.no_save:
                torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))
        results['train_loss'].append(train_loss)
        results['test_acc@1'].append(test_acc_1)
        results['test_acc@5'].append(test_acc_5)
        results['best_acc@1'].append(best_acc)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        if not args.no_save:
            data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
    if not args.no_save:
        torch.save(model.state_dict(), 'results/{}_final_model.pth'.format(save_name_pre))

# train for one epoch to learn unique features
def train_cluster_batch(net, data_loader, train_optimizer, epoch, neg_mode):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    last_batch_out = None
    last_batch_target = None
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True), target.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        batch_size = out_1.shape[0]
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        sim_pos1_neg1 = sim_matrix[:batch_size, :batch_size]
        sim_pos1_neg2 = sim_matrix[:batch_size, batch_size:]
        sim_pos2_neg1 = sim_matrix[batch_size:, :batch_size]
        sim_pos2_neg2 = sim_matrix[batch_size:, batch_size:]
        mask_pos1_neg1 = sim_pos1_neg1 > sim_pos1_neg2
        mask_pos1_neg2 = mask_pos1_neg1.logical_not()
        mask_pos2_neg1 = sim_pos2_neg1 > sim_pos2_neg2
        mask_pos2_neg2 = mask_pos2_neg1.logical_not()
        if 'gt_test5':
            mask_pos1_neg1 = torch.ones_like(mask_pos1_neg1).bool()
            mask_pos1_neg2 = torch.zeros_like(mask_pos1_neg2).bool()
            mask_pos2_neg1 = torch.zeros_like(mask_pos2_neg1).bool()
            mask_pos2_neg2 = torch.ones_like(mask_pos2_neg2).bool()

        mask_pos1 = torch.cat([mask_pos1_neg1, mask_pos1_neg2], dim=1)
        mask_pos2 = torch.cat([mask_pos2_neg1, mask_pos2_neg2], dim=1)
        mask_closer_neg = torch.cat([mask_pos1, mask_pos2], dim=0)

        label_mat = target.repeat([batch_size, 1]).t()
        mask_same_label = label_mat == label_mat.t()
        mask_same_label = mask_same_label.repeat([2,2])

        if last_batch_out != None:
            sim_matrix2 = torch.exp(torch.mm(out, last_batch_out.t().contiguous()) / temperature)
            sim_pos1_neg1 = sim_matrix2[:batch_size, :batch_size]
            sim_pos1_neg2 = sim_matrix2[:batch_size, batch_size:]
            sim_pos2_neg1 = sim_matrix2[batch_size:, :batch_size]
            sim_pos2_neg2 = sim_matrix2[batch_size:, batch_size:]
            mask_pos1_neg1 = sim_pos1_neg1 > sim_pos1_neg2
            mask_pos1_neg2 = mask_pos1_neg1.logical_not()
            mask_pos2_neg1 = sim_pos2_neg1 > sim_pos2_neg2
            mask_pos2_neg2 = mask_pos2_neg1.logical_not()

            mask_pos1 = torch.cat([mask_pos1_neg1, mask_pos1_neg2], dim=1)
            mask_pos2 = torch.cat([mask_pos2_neg1, mask_pos2_neg2], dim=1)
            mask_closer_neg2 = torch.cat([mask_pos1, mask_pos2], dim=0)

            label_mat2 = last_batch_target.repeat([batch_size, 1]).t()
            mask_same_label2 = label_mat == label_mat2.t()
            mask_same_label2 = mask_same_label2.repeat([2,2])

            mask_diff_label_neg2 = torch.logical_and(mask_closer_neg2, mask_same_label2.logical_not())

            mask_neg2 = mask_diff_label_neg2
            mask2 = mask_neg2.int()

            sim_matrix2 = sim_matrix2 * mask2

        ## check whether it is right!!!!!

        # print(mask_closer_neg, mask_same_label)
        if neg_mode == 'gt_same_label':
            mask_same_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label)
            mask_diff_label_neg = mask_same_label.logical_not()
        elif neg_mode == 'gt_same_diff_label':
            mask_same_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label)
            mask_diff_label_neg = torch.logical_and(mask_closer_neg, mask_same_label.logical_not())
        elif neg_mode == 'gt_diff_label':
            mask_same_label_neg = mask_same_label
            mask_diff_label_neg = torch.logical_and(mask_closer_neg, mask_same_label.logical_not())
        elif neg_mode == 'gt_test1':
            mask_same_label_neg = torch.logical_and(mask_closer_neg, mask_same_label)
            mask_diff_label_neg = mask_same_label.logical_not()
        elif neg_mode == 'gt_test2':
            mask_same_label_neg = torch.logical_and(mask_closer_neg, mask_same_label)
            mask_diff_label_neg = torch.logical_and(mask_closer_neg, mask_same_label.logical_not())
        elif neg_mode == 'gt_test3':
            mask_same_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label)
            mask_diff_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label.logical_not())
        elif neg_mode == 'gt_test4':
            mask_same_label_neg = mask_same_label
            mask_diff_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label.logical_not()) # This can support the conclusion about clustering
        elif neg_mode == 'gt_test5':
            mask_same_label_neg = torch.logical_and(mask_closer_neg.logical_not(), mask_same_label)
            mask_diff_label_neg = torch.logical_and(mask_closer_neg, mask_same_label.logical_not())
        
        mask_neg = torch.logical_or(mask_same_label_neg, mask_diff_label_neg)
        # input()

        mask_remove_self = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        mask = torch.logical_and(mask_remove_self, mask_neg).int()
        # [2*B, 2*B-1]
        # sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        sim_matrix = sim_matrix * mask

        if last_batch_out != None:
            sim_matrix = torch.cat([sim_matrix, sim_matrix2], dim=-1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

        last_batch_out = out.detach()
        last_batch_target = target.detach()

    return total_loss / total_num

# train for one epoch to learn unique features
def train(net, data_loader, train_optimizer, epoch):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        # [2*B, D]
        out = torch.cat([out_1, out_2], dim=0)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_data_loader, test_data_loader, epoch):
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


if __name__ == '__main__':
    feature_dim, temperature, k = args.feature_dim, args.temperature, args.k
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    if args.dataset == 'cifar10':
        train_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        memory_data = utils.CIFAR10Pair(root='data', train=True, transform=utils.test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR10Pair(root='data', train=False, transform=utils.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    elif args.dataset == 'cifar100':
        train_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.train_transform, download=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        memory_data = utils.CIFAR100Pair(root='data', train=True, transform=utils.test_transform, download=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_data = utils.CIFAR100Pair(root='data', train=False, transform=utils.test_transform, download=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # model setup and optimizer config
    model = Model(feature_dim).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    c = len(memory_data.classes)

    if args.load_model:
        # unlearnable_cleantrain_41501264_1_20211204151414_0.5_512_1000_final_model
        load_model_path = './results/{}.pth'.format(args.load_model_path)
        # load_model_path = '/mnt/home/renjie3/Documents/unlearnable/CL_cluster/diffaug_SimCLR/results/vanilla_local_128_0.5_200_1024_1000_model.pth'
        checkpoints = torch.load(load_model_path, map_location=device)
        filter_name_checkpoints = {}
        for key in checkpoints:
            filter_name_checkpoints[key.replace('module.', '')] = checkpoints[key]
        model.load_state_dict(filter_name_checkpoints)

    # training loop
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(args.method, args.job_id, feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    # # model setup and optimizer config
    # model2 = Model(feature_dim).cuda()
    # flops, params = profile(model2, inputs=(torch.randn(1, 3, 32, 32).cuda(),))
    # flops, params = clever_format([flops, params])
    # print('# Model Params: {} FLOPs: {}'.format(params, flops))
    # optimizer = optim.Adam(model2.parameters(), lr=1e-3, weight_decay=1e-6)
    # c = len(memory_data.classes)

    # if args.load_model:
    #     load_model_path = './results/vanilla_local_128_0.5_200_512_1_final_model.pth'
    #     checkpoints = torch.load(load_model_path, map_location=device)
    #     filter_name_checkpoints = {}
    #     for key in checkpoints:
    #         filter_name_checkpoints[key.replace('module.', '')] = checkpoints[key]
    #     model2.load_state_dict(filter_name_checkpoints)

    # training loop
    save_name_pre = '{}_{}_{}_{}_{}_{}_{}'.format(args.method, args.job_id, feature_dim, temperature, k, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    if args.plot_feature:
        utils.test_ssl_visualization(model, test_data, args.plot_save_name, None)
        input('plot done')

    if args.method == 'vanilla':
        vanilla(epochs, model, optimizer, train_loader, memory_loader, test_loader)
    elif args.method == 'decoupled':
        decoupled(epochs, model, optimizer, train_loader, memory_loader, test_loader)
    elif args.method == 'cluster':
        cluster(epochs, model, optimizer, train_loader, memory_loader, test_loader, args.neg)
