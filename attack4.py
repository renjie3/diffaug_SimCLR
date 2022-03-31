
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from utils2 import train_diff_transform, ToTensor_transform
import kornia.augmentation as Kaug

from sklearn import metrics
from kmeans_pytorch import kmeans

import time

def train_diff_transform_prob(p_recrop=0.0, p_hflip=0.0, p_cj=0.0, p_gray=0.0, s_cj=1):
    # simclr: 1.0 0.5 0.8 0.2
    return nn.Sequential(
            Kaug.RandomResizedCrop([32,32], p=p_recrop),
            Kaug.RandomHorizontalFlip(p=p_hflip),
            Kaug.ColorJitter(0.4*s_cj, 0.4*s_cj, 0.4*s_cj, 0.1*s_cj, p=p_cj),
            Kaug.RandomGrayscale(p=p_gray)
        )

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

class PGD():
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, epsilon, alpha, min_val, max_val, max_iters, augmentation_prob, loss_type, _type='linf'):
        self.model = model
        # self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type
        # The loss of perturbation
        self.loss_type = loss_type

        if np.sum(augmentation_prob) != 0:
            self.transform = train_diff_transform_prob(*augmentation_prob)
        else:
            self.transform = train_diff_transform

    def project(self, x, original_x, epsilon, _type='linf'):

        if _type == 'linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon

            x = torch.max(torch.min(x, max_x), min_x)

        elif _type == 'l2':
            dist = (x - original_x)
            dist = dist.view(x.shape[0], -1)
            dist_norm = torch.norm(dist, dim=1, keepdim=True)
            mask = (dist_norm > epsilon).unsqueeze(2).unsqueeze(3)
            # dist = F.normalize(dist, p=2, dim=1)
            dist = dist / dist_norm
            dist *= epsilon
            dist = dist.view(x.shape)
            x = (original_x + dist) * mask.float() + x * (1 - mask.float())

        else:
            raise NotImplementedError

        return x
        
    def perturb(self, original_images, labels, temperature, reverse, repeat_num, reduction4loss='mean', random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = tensor2cuda(rand_perturb)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True 

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon

        self.model.eval()

        timer = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # self.get_DBindex(x, torch.arange(x.shape[0]).cuda())
        # self.get_DBindex(x, labels)

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                if self.loss_type == '':
                    pos_1, pos_2 = self.transform(x), self.transform(x)
                    feature_1, out_1 = self.model(pos_1)
                    feature_2, out_2 = self.model(pos_2)
                    # [2*B, D]
                    out = torch.cat([out_1, out_2], dim=0)
                    # [2*B, 2*B]
                    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
                    this_batch_size = pos_1.shape[0]
                    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * this_batch_size, device=sim_matrix.device)).bool()
                    # [2*B, 2*B-1]
                    sim_matrix = sim_matrix.masked_select(mask).view(2 * this_batch_size, -1)

                    # compute loss
                    pos_sim = torch.exp((torch.sum(out_1 * out_2, dim=-1)) / temperature)
                    # [2*B]
                    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                    loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                elif self.loss_type == 'DBindex_high2low':
                    # 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 
                    # # print(np.max(train_targets))
                    sample = []
                    # repeat_num = 5
                    for _ in range(repeat_num):
                        # time0 = time.time()
                        aug_sample = self.transform(x)
                        # time1 = time.time()
                        # timer[0] += (time1 - time0)
                        feature, out = self.model(aug_sample)
                        # time2 = time.time()
                        # timer[1] += (time2 - time1)
                        sample.append(feature)
                    # time2 = time.time()
                    sample = torch.cat(sample, dim=0).double()
                    inst_label = torch.arange(x.shape[0]).cuda()
                    inst_label = inst_label.repeat((repeat_num, ))
                    class_center = []
                    sort_class = []
                    intra_class_dis = []
                    c = x.shape[0]
                    # time3 = time.time()
                    # timer[2] += (time3 - time2)
                    for i in range(c):
                        idx_i = torch.where(inst_label == i)[0]
                        class_i = sample[idx_i, :]
                        class_i_center = class_i.mean(dim=0)
                        class_center.append(class_i_center)
                        intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
                    # time4 = time.time()
                    # timer[3] += (time4 - time3)
                    class_center = torch.stack(class_center, dim=0)
                    
                    class_dis = torch.cdist(class_center, class_center, p=2)
                    # time5 = time.time()
                    # timer[4] += (time5 - time4)
                    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
                    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

                    # time6 = time.time()
                    # timer[5] += (time6 - time5)

                    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
                    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
                    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
                    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

                    # time7 = time.time()
                    # timer[6] += (time7 - time6)

                    loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

                    # time8 = time.time()
                    # timer[7] += (time8 - time7)
                    # max_np = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].detach().cpu().numpy()
                    # sample_np = sample.detach().cpu().numpy()
                    # inst_label_np = inst_label.detach().cpu().numpy()

                elif self.loss_type == 'DBindex_cluster_GT':
                    # 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 
                    # # print(np.max(train_targets))
                    sample = []
                    # repeat_num = 5
                    for _ in range(repeat_num):
                        aug_sample = self.transform(x)
                        feature, out = self.model(aug_sample)
                        sample.append(feature)
                    sample = torch.cat(sample, dim=0).double()
                    cluster_label = labels.repeat((repeat_num, ))
                    class_center = []
                    sort_class = []
                    intra_class_dis = []
                    c = torch.max(cluster_label) + 1
                    for i in range(c):
                        idx_i = torch.where(cluster_label == i)[0]
                        class_i = sample[idx_i, :]
                        class_i_center = class_i.mean(dim=0)
                        class_center.append(class_i_center)
                        intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
                    class_center = torch.stack(class_center, dim=0)
                    
                    class_dis = torch.cdist(class_center, class_center, p=2)
                    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
                    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

                    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
                    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
                    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
                    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

                    loss = - torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

                elif self.loss_type == 'DBindex_ratio_inst_cluster_GT':
                    # 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 
                    # # print(np.max(train_targets))
                    sample = []
                    # repeat_num = 5
                    for _ in range(repeat_num):
                        aug_sample = self.transform(x)
                        feature, out = self.model(aug_sample)
                        sample.append(feature)
                    sample = torch.cat(sample, dim=0).double()
                    cluster_label = labels.repeat((repeat_num, ))
                    class_center = []
                    sort_class = []
                    intra_class_dis = []
                    c = torch.max(cluster_label) + 1
                    for i in range(c):
                        idx_i = torch.where(cluster_label == i)[0]
                        class_i = sample[idx_i, :]
                        class_i_center = class_i.mean(dim=0)
                        class_center.append(class_i_center)
                        intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
                    class_center = torch.stack(class_center, dim=0)
                    
                    class_dis = torch.cdist(class_center, class_center, p=2)
                    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
                    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

                    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
                    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
                    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
                    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

                    cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

                    inst_label = torch.arange(x.shape[0]).cuda()
                    inst_label = inst_label.repeat((repeat_num, ))
                    class_center = []
                    sort_class = []
                    intra_class_dis = []
                    c = x.shape[0]
                    for i in range(c):
                        idx_i = torch.where(inst_label == i)[0]
                        class_i = sample[idx_i, :]
                        class_i_center = class_i.mean(dim=0)
                        class_center.append(class_i_center)
                        intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
                    class_center = torch.stack(class_center, dim=0)
                    
                    class_dis = torch.cdist(class_center, class_center, p=2)
                    mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
                    class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

                    intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
                    trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
                    intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
                    intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

                    inst_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

                    loss = inst_DB_loss / cluster_DB_loss

                if reduction4loss == 'none':
                    grad_outputs = tensor2cuda(torch.ones(loss.shape))
                    
                else:
                    grad_outputs = None

                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, 
                        only_inputs=True)[0]

                if reverse:
                    x.data -= self.alpha * torch.sign(grads.data)
                else:
                    x.data += self.alpha * torch.sign(grads.data)

                # the adversaries' pixel value should within max_x and min_x due 
                # to the l_infinity / l2 restriction
                x = self.project(x, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                x.clamp_(self.min_val, self.max_val)

        # self.get_DBindex(x, torch.arange(x.shape[0]).cuda())
        # self.get_DBindex(x, labels)

        # input()

        return x

    def get_DBindex(self, x, label, repeat_num=5):
        sample = []
        self.model.eval()
        for _ in range(repeat_num):
            aug_sample = self.transform(x)
            feature, out = self.model(aug_sample)
            sample.append(feature)
        sample = torch.cat(sample, dim=0).detach().cpu().numpy()
        new_label = label.repeat((repeat_num, )).detach().cpu().numpy()
        print(metrics.davies_bouldin_score(sample, new_label))


def get_dbindex_loss(net, x, labels, loss_type, reverse, my_transform, num_clusters, repeat_num, use_out_dbindex, use_sim, kmean_result, use_wholeset_centroid, use_mean_dbindex, flag_select_confidence, confidence_thre, keep_gradient_on_center, inter_dis_type, dbindex_type):

    # repeat_num = 5
    if loss_type in ['DBindex_cluster_momentum_kmeans', 'DBindex_cluster_momentum_kmeans_wholeset']:
        if repeat_num != 1:
            momentum_encoder_sample = []
            for i in range(repeat_num):
                aug_sample = my_transform(x)
                feature, out = net.momentum_encoder(aug_sample)
                if use_out_dbindex:
                    momentum_encoder_sample.append(out)
                else:
                    momentum_encoder_sample.append(feature)
            momentum_encoder_sample = torch.cat(momentum_encoder_sample, dim=0).double()
        else:
            feature, out = net.momentum_encoder(x)
            if use_out_dbindex:
                # input('check here')
                momentum_encoder_sample = out.double()
            else:
                momentum_encoder_sample = feature.double()

    # DBindex_cluster_momentum_kmeans_repeat_v2
    if loss_type in ['DBindex_cluster_momentum_kmeans_repeat_v2', 'DBindex_cluster_momentum_kmeans_repeat_v2_weighted_cluster', 'DBindex_cluster_momentum_kmeans_repeat_v2_mean_dbindex']:
        feature, out = net.momentum_encoder(x) # here it requires checck
        if use_out_dbindex:
            momentum_encoder_sample = out.double()
        else:
            momentum_encoder_sample = feature.double()
    
    if repeat_num != 1:
        sample = []
        for i in range(repeat_num):
            aug_sample = my_transform(x)
            feature, out = net(aug_sample.detach())
            if use_out_dbindex:
                sample.append(out)
            else:
                sample.append(feature)
        sample = torch.cat(sample, dim=0).double()
    else:
        feature, out = net(x)
        if use_out_dbindex:
            # input('check here')
            sample = out.double()
        else:
            sample = feature.double()

    if loss_type == 'DBindex_cluster_momentum_kmeans_wholeset':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        n_clueter_num = len(num_clusters)
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels = labels[:, num_cluster_idx]
            high_conf_label = labels[:, n_clueter_num + num_cluster_idx]
            cluster_label = kmeans_labels.repeat((repeat_num, ))
            high_conf_label = high_conf_label.repeat((repeat_num, ))
            point_dis_to_center_list = []
            if not use_sim:
                class_center = []
                class_center_wholeset = []
                intra_class_dis = []
                c = torch.max(cluster_label) + 1 # The class larger than c is not included in this batch
                for i in range(c):
                    idx_i = torch.where(cluster_label == i)[0]
                    class_i = sample[idx_i, :]
                    class_high_conf_label = high_conf_label[idx_i]
                    # class_i = sample[idx_i, :][class_high_conf_label == 1]


                    # class_i_center = kmean_result['centroids'][num_cluster_idx][i].detach()
                    # if keep_gradient_on_center:
                    #     class_i_center = class_i.mean(dim=0)
                    # else:
                    #     class_i_center = kmean_result['centroids'][num_cluster_idx][i].detach()
                    # class_i_center = class_i.mean(dim=0)
                    class_i_center = kmean_result['centroids'][num_cluster_idx][i].detach()

                    class_i_center = nn.functional.normalize(class_i_center, p=2, dim=0)
                    
                    # torch.set_printoptions(profile="full")
                    # print(torch.norm(kmean_result['centroids'][num_cluster_idx], dim=1, p=2))
                    # input()
                    if class_i.shape[0] == 0:
                        continue
                    # class_center.append(class_i_center)
                    class_center_wholeset.append(kmean_result['centroids'][num_cluster_idx][i].detach())
                    class_center.append(nn.functional.normalize(class_i.mean(dim=0), p=2, dim=0))
                    # class_center.append(class_i.mean(dim=0))
                    # print(class_i.shape)
                    # print(torch.norm(class_i, dim=1, p=2))
                    # input()
                    point_dis_to_center = torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))
                    # TODO: Similar to this
                    # >>> import torch
                    # >>> a = torch.tensor([0, 1, 3, 3.5, 4.5, 8])
                    # >>> b = torch.tensor([0, 1, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0,0,0, ])
                    # >>> a[b]
                    if flag_select_confidence:
                        point_dis_to_center = point_dis_to_center[class_high_conf_label == 1]
                        if point_dis_to_center.shape[0] == 0:
                            class_center.pop()
                            class_center_wholeset.pop()
                            continue
                            # input('yes')
                    # print(torch.max(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
                    point_dis_to_center_list.append(point_dis_to_center)
                    intra_class_dis.append(torch.mean(point_dis_to_center)) # TODO: here this should not be mean. I think it should be a classwise coefficient.
                    # intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)*0, dim = 1))))
                if len(class_center) <= 1:
                    continue
                class_center = torch.stack(class_center, dim=0)
                class_center_wholeset = torch.stack(class_center_wholeset, dim=0)
                # input('no')

                c = len(intra_class_dis)

                if inter_dis_type == 'wholeset':
                    class_dis = torch.cdist(class_center_wholeset.double(), class_center_wholeset.double(), p=2) # TODO: this can be done for only one time in the whole set
                elif inter_dis_type == 'half':
                    class_dis = torch.cdist(class_center.double(), class_center_wholeset.double(), p=2) # TODO: this can be done for only one time in the whole set
                elif inter_dis_type == 'batch':
                    class_dis = torch.cdist(class_center.double(), class_center.double(), p=2) # TODO: this can be done for only one time in the whole set
                
                
                # print(class_dis)

                mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
                class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)
                # print(class_dis)
                # input()

                intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
                trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
                intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
                intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

                if use_mean_dbindex:
                    # cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()
                    # cluster_DB_loss = (intra_class_dis_pair_sum).mean() + (1 / class_dis).mean()
                    if dbindex_type == 'half':
                        if num_cluster_idx == 0:
                            cluster_DB_loss = (intra_class_dis_pair_sum / (class_dis + 0.0001)).mean()
                        else:
                            cluster_DB_loss = (intra_class_dis_pair_sum).mean()
                    elif dbindex_type == 'intra_inter':
                        cluster_DB_loss = (intra_class_dis_pair_sum / (class_dis + 0.0001)).mean()
                    elif dbindex_type == 'inter':
                        cluster_DB_loss = (1 / (class_dis + 0.0001)).mean()
                    # cluster_DB_loss = - class_dis.mean()
                    # cluster_DB_loss = - (1 / intra_class_dis_pair_sum).mean()
                else:
                    cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()
                # print(cluster_DB_loss)
                # print(cluster_DB_loss_mean)
                # print(intra_class_dis_pair_sum / class_dis)
                # input()
                # print("cluster_DB_loss", cluster_DB_loss.item())
                # print(cluster_DB_loss)
                # input()
                loss -= cluster_DB_loss
            else:
                class_center = []
                intra_class_sim = []
                c = torch.max(cluster_label) + 1
                for i in range(c):
                    idx_i = torch.where(cluster_label == i)[0]
                    class_i = sample[idx_i, :]
                    class_i_center = class_i.mean(dim=0)
                    if idx_i.shape[0] == 0:
                        continue
                    class_center.append(class_i_center)
                    intra_class_sim.append(torch.mean(torch.mm(class_i, class_i_center.unsqueeze(1))))
                    # intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)*0, dim = 1))))
                class_center = torch.stack(class_center, dim=0)

                c = len(intra_class_sim)
                
                class_sim = torch.mm(class_center, class_center.t().contiguous())

                mask = (torch.ones_like(class_sim) - torch.eye(class_sim.shape[0], device=class_sim.device)).bool()
                class_sim = class_sim.masked_select(mask).view(class_sim.shape[0], -1)

                intra_class_sim = torch.tensor(intra_class_sim).unsqueeze(1).repeat((1, c)).cuda()
                trans_intra_class_sim = torch.transpose(intra_class_sim, 0, 1)
                intra_class_sim_pair_sum = intra_class_sim + trans_intra_class_sim
                intra_class_sim_pair_sum = intra_class_sim_pair_sum.masked_select(mask).view(intra_class_sim_pair_sum.shape[0], -1)

                cluster_DB_loss = torch.min(class_sim / intra_class_sim_pair_sum, dim=1)[0].mean()
                loss -= cluster_DB_loss

        # print(loss.item())
        # input()
    
    elif loss_type == 'DBindex_cluster_GT':
        # print(torch.norm(kmean_result['centroids'][0], dim=1, p=2))
        # input()
        cluster_label = labels.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = torch.max(cluster_label) + 1
        for i in range(c):
            idx_i = torch.where(cluster_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            # class_i_center = kmean_result['centroids'][0][i].detach()
            if idx_i.shape[0] == 0:
                continue
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        loss = - torch.mean(intra_class_dis_pair_sum / class_dis)

    elif loss_type == 'DBindex_cluster_momentum_kmeans_repeat_v2':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels, cluster_centers = kmeans(
                X=momentum_encoder_sample, num_clusters=num_clusters[num_cluster_idx], distance='euclidean', device=sample.device, tqdm_flag=False
            )
            cluster_label = kmeans_labels.repeat((repeat_num, ))
            class_center = []
            sort_class = []
            intra_class_dis = []
            c = torch.max(cluster_label) + 1
            for i in range(c):
                idx_i = torch.where(cluster_label == i)[0]
                class_i = sample[idx_i, :]
                class_i_center = class_i.mean(dim=0)
                class_center.append(class_i_center)
                intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
            class_center = torch.stack(class_center, dim=0)
            
            class_dis = torch.cdist(class_center, class_center, p=2)
            mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
            class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

            intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
            trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
            intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
            intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

            cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()
            loss -= cluster_DB_loss

    elif loss_type == 'DBindex_high2low':
        # 'DBindex_high2low', 'DBindex_cluster_GT', 'DBindex_ratio_inst_cluster_GT', 
        inst_label = torch.arange(x.shape[0]).cuda()
        inst_label = inst_label.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = x.shape[0]
        for i in range(c):
            idx_i = torch.where(inst_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

    elif loss_type == 'DBindex_ratio_inst_cluster_GT':
        cluster_label = labels.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = torch.max(cluster_label) + 1
        for i in range(c):
            idx_i = torch.where(cluster_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

        inst_label = torch.arange(x.shape[0]).cuda()
        inst_label = inst_label.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = x.shape[0]
        for i in range(c):
            idx_i = torch.where(inst_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        inst_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

        loss = inst_DB_loss / cluster_DB_loss

    elif loss_type == 'DBindex_product_inst_cluster_GT':
        cluster_label = labels.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = torch.max(cluster_label) + 1
        for i in range(c):
            idx_i = torch.where(cluster_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

        inst_label = torch.arange(x.shape[0]).cuda()
        inst_label = inst_label.repeat((repeat_num, ))
        class_center = []
        sort_class = []
        intra_class_dis = []
        c = x.shape[0]
        for i in range(c):
            idx_i = torch.where(inst_label == i)[0]
            class_i = sample[idx_i, :]
            class_i_center = class_i.mean(dim=0)
            class_center.append(class_i_center)
            intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
        class_center = torch.stack(class_center, dim=0)
        
        class_dis = torch.cdist(class_center, class_center, p=2)
        mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
        class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

        intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
        trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
        intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
        intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

        inst_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()

        loss = inst_DB_loss * cluster_DB_loss

    elif loss_type == 'DBindex_cluster_kmeans':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels, cluster_centers = kmeans(
                X=sample, num_clusters=num_clusters[num_cluster_idx], distance='euclidean', device=sample.device, tqdm_flag=False
            )
            cluster_label = kmeans_labels #.repeat((repeat_num, ))
            class_center = []
            sort_class = []
            intra_class_dis = []
            c = torch.max(cluster_label) + 1
            for i in range(c):
                idx_i = torch.where(cluster_label == i)[0]
                class_i = sample[idx_i, :]
                class_i_center = class_i.mean(dim=0)
                class_center.append(class_i_center)
                intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
            class_center = torch.stack(class_center, dim=0)
            
            class_dis = torch.cdist(class_center, class_center, p=2)
            mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
            class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

            intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
            trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
            intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
            intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

            cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()
            loss -= cluster_DB_loss

    elif loss_type == 'DBindex_cluster_momentum_kmeans':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels, cluster_centers = kmeans(
                X=momentum_encoder_sample, num_clusters=num_clusters[num_cluster_idx], distance='euclidean', device=sample.device, tqdm_flag=False
            )
            cluster_label = kmeans_labels #.repeat((repeat_num, ))
            class_center = []
            sort_class = []
            intra_class_dis = []
            c = torch.max(cluster_label) + 1
            for i in range(c):
                idx_i = torch.where(cluster_label == i)[0]
                class_i = sample[idx_i, :]
                class_i_center = class_i.mean(dim=0)
                class_center.append(class_i_center)
                intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
            class_center = torch.stack(class_center, dim=0)
            
            class_dis = torch.cdist(class_center, class_center, p=2)
            mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
            class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

            intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
            trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
            intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
            intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

            cluster_DB_loss = torch.max(intra_class_dis_pair_sum / class_dis, dim=1)[0].mean()
            loss -= cluster_DB_loss

    elif loss_type == 'DBindex_cluster_momentum_kmeans_repeat_v2_weighted_cluster':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels, cluster_centers = kmeans(
                X=momentum_encoder_sample, num_clusters=num_clusters[num_cluster_idx], distance='euclidean', device=sample.device, tqdm_flag=False
            )
            cluster_label = kmeans_labels.repeat((repeat_num, ))
            class_center = []
            sort_class = []
            intra_class_dis = []
            c = torch.max(cluster_label) + 1
            for i in range(c):
                idx_i = torch.where(cluster_label == i)[0]
                class_i = sample[idx_i, :]
                class_i_center = class_i.mean(dim=0)
                class_center.append(class_i_center)
                intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
            class_center = torch.stack(class_center, dim=0)
            
            class_dis = torch.cdist(class_center, class_center, p=2)
            mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
            class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

            intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
            trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
            intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
            intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

            cluster_DB_index = (intra_class_dis_pair_sum / class_dis)
            cluster_DB_loss = (cluster_DB_index * cluster_DB_index).mean()
            # print(cluster_DB_loss.shape)
            # input()
            loss -= cluster_DB_loss

    elif loss_type == 'DBindex_cluster_momentum_kmeans_repeat_v2_mean_dbindex':
        if len(num_clusters) <= 1 and np.sum(num_clusters) == 0:
            num_clusters = [4, 5, 7, 10, 15, 20]
        loss = 0
        for num_cluster_idx in range(len(num_clusters)):
            kmeans_labels, cluster_centers = kmeans(
                X=momentum_encoder_sample, num_clusters=num_clusters[num_cluster_idx], distance='euclidean', device=sample.device, tqdm_flag=False
            )
            cluster_label = kmeans_labels.repeat((repeat_num, ))
            class_center = []
            sort_class = []
            intra_class_dis = []
            c = torch.max(cluster_label) + 1
            for i in range(c):
                idx_i = torch.where(cluster_label == i)[0]
                class_i = sample[idx_i, :]
                class_i_center = class_i.mean(dim=0)
                class_center.append(class_i_center)
                intra_class_dis.append(torch.mean(torch.sqrt(torch.sum((class_i-class_i_center)**2, dim = 1))))
            class_center = torch.stack(class_center, dim=0)
            
            class_dis = torch.cdist(class_center, class_center, p=2)
            mask = (torch.ones_like(class_dis) - torch.eye(class_dis.shape[0], device=class_dis.device)).bool()
            class_dis = class_dis.masked_select(mask).view(class_dis.shape[0], -1)

            intra_class_dis = torch.tensor(intra_class_dis).unsqueeze(1).repeat((1, c)).cuda()
            trans_intra_class_dis = torch.transpose(intra_class_dis, 0, 1)
            intra_class_dis_pair_sum = intra_class_dis + trans_intra_class_dis
            intra_class_dis_pair_sum = intra_class_dis_pair_sum.masked_select(mask).view(intra_class_dis_pair_sum.shape[0], -1)

            cluster_DB_loss = (intra_class_dis_pair_sum / class_dis).mean()
            loss -= cluster_DB_loss

    if not reverse:
        return loss
    else:
        return -loss
