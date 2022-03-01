from utils import train_diff_transform, ToTensor_transform
import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import time
from inst_suppress_utils import get_batch_idx_group
import torch.nn as nn
import kornia.augmentation as Kaug

# def train_diff_transform_prob(p_recrop=0.0, p_hflip=0.0, p_cj=0.0, p_gray=0.0):
#     # simclr: 1.0 0.5 0.8 0.2
#     return nn.Sequential(
#             Kaug.RandomResizedCrop([32,32], p=p_recrop),
#             Kaug.RandomHorizontalFlip(p=p_hflip),
#             Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=p_cj),
#             Kaug.RandomGrayscale(p=p_gray)
#         )

def train_diff_transform_prob(p_recrop=0.0, p_hflip=0.0, p_cj=0.0, p_gray=0.0, s_cj=1):
    # simclr: 1.0 0.5 0.8 0.2
    return nn.Sequential(
            Kaug.RandomResizedCrop([32,32], p=p_recrop),
            Kaug.RandomHorizontalFlip(p=p_hflip),
            Kaug.ColorJitter(0.4*s_cj, 0.4*s_cj, 0.4*s_cj, 0.1*s_cj, p=p_cj),
            Kaug.RandomGrayscale(p=p_gray)
        )

def get_aug(net, data_loader, batch_size, use_out, augmentation_prob, cj_strength, save_file_name='temp.txt'):
    net.eval()

    my_transform = train_diff_transform_prob(*augmentation_prob, cj_strength)

    DB_inst_list = []
    DB_cluster_list = []
    center_DB_cluster_list = []

    for _ in range(10):

        arange_batch_idx = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=batch_size, shuffle=False, drop_last=True)
        batch_num = len(arange_batch_idx)
        feature_bank = []
        label_bank = []
        inst_label_bank = []
        center_feature_bank = []
        center_label_bank = []
        print("Curriculum sample_from_mass: extracting feature..")
        net.eval()
        for i in range(batch_num):
            batch_feature = []
            pos_1, target = data_loader.get_batch(arange_batch_idx[i])
            pos_1 = pos_1.cuda(non_blocking=True)
            for _ in range(10):
                pos_1_aug = my_transform(pos_1)
                feature, out = net(pos_1_aug)
                if not use_out:
                    feature_bank.append(feature.cpu().detach().numpy())
                    batch_feature.append(feature.cpu().detach().numpy())
                else:
                    feature_bank.append(out.cpu().detach().numpy())
                    batch_feature.append(out.cpu().detach().numpy())
                label_bank.append(target.cpu().detach().numpy())
                inst_label_bank.append(arange_batch_idx[i])
            batch_feature = np.stack(batch_feature, axis=0)
            # print(batch_feature.shape)
            batch_feature = np.mean(batch_feature, axis=0)
            # print(batch_feature.shape)
            # input()
            center_feature_bank.append(batch_feature)
            center_label_bank.append(target.cpu().detach().numpy())

        feature_bank = np.concatenate(feature_bank, axis=0)
        center_feature_bank = np.concatenate(center_feature_bank, axis=0)
        label_bank = np.concatenate(label_bank, axis=0)
        center_label_bank = np.concatenate(center_label_bank, axis=0)
        inst_label_bank = np.concatenate(inst_label_bank, axis=0)

        print("Curriculum sample_from_mass: extracting feature DONE.")

        DB_inst = metrics.davies_bouldin_score(feature_bank, inst_label_bank)
        DB_cluster = metrics.davies_bouldin_score(feature_bank, label_bank)
        center_DB_cluster = metrics.davies_bouldin_score(center_feature_bank, center_label_bank)
        print(DB_inst)
        print(DB_cluster)
        print(center_DB_cluster)

        DB_inst_list.append(DB_inst)
        DB_cluster_list.append(DB_cluster)
        center_DB_cluster_list.append(center_DB_cluster)

    DB_inst_mean = np.mean(DB_inst_list)
    DB_cluter_mean = np.mean(DB_cluster_list)
    center_DB_cluter_mean = np.mean(center_DB_cluster_list)
    DB_inst_std = np.std(DB_inst_list)
    DB_cluter_std = np.std(DB_cluster_list)
    center_DB_cluter_std = np.std(center_DB_cluster_list)

    print("{:.2f}±{:.2f}".format(DB_inst_mean, DB_inst_std))
    print("{:.2f}±{:.2f}".format(DB_cluter_mean, DB_cluter_std))
    print("{:.2f}±{:.2f}".format(center_DB_cluter_mean, center_DB_cluter_std))

    print(DB_inst_mean / DB_cluter_mean)
    print(DB_inst_mean / center_DB_cluter_mean)

    f = open("./{}".format(save_file_name), "a")
    f.write("{}\t{}\t{}\t{}\t{}\n".format(DB_inst_mean, DB_cluter_mean, center_DB_cluter_mean, DB_inst_mean / DB_cluter_mean, DB_inst_mean / center_DB_cluter_mean))
    f.close()

def cluster_augmentation(net, sample, target, batch_noise, index_str, epsilon, adv_step, step_size, augmentation_prob):
    
    if np.sum(augmentation_prob) != 0:
        my_transform = train_diff_transform_prob(*augmentation_prob)
    else:
        my_transform = train_diff_transform

    for _ in range(adv_step):
        feature, out = net(sample)
    
        net.train()
        pos_1, pos_2 = sample.cuda(non_blocking=True), sample.cuda(non_blocking=True)
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
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return loss.item() * this_batch_size, this_batch_size


# 0.75
# 4.817785491999378
# 9.797968756478
# 4.835217477932049
# 9.6492752
# 4.81343396676924
# 9.6618827

# 0.8
# 4.844401757461425
# 9.675185762412
# 4.844766064834024
# 9.65820710320444


# pretrain

# 0.5
# 1.8685616348078615
# 2.5974140764

# 0.7
# 1.8962703672614158
# 2.6067267

# 0.8 
# 1.9169538731205098
# 2.61022154597

# 0.9
# 1.93890983
# 2.61733329

# 1.0
# 1.951487520609953
# 2.6220965
