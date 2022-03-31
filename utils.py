from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
import kornia.augmentation as Kaug
import torch.nn as nn
import torch
import os
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn import manifold, datasets
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
import math
from matplotlib.colors import ListedColormap

from tqdm import tqdm

import faiss

# from inst_suppress_utils import get_batch_idx_group
from kmeans_pytorch import kmeans

ToTensor_transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_batch_idx_group(total_num, batch_size=512, shuffle=False, drop_last=False):
    if shuffle:
        perm_id = np.random.permutation(total_num)
    else:
        perm_id = np.arange(total_num)
    if drop_last:
        batch_num = total_num // batch_size
    else:
        batch_num = (total_num - 1) // batch_size + 1
    batch_idx_list = []
    for i in range(batch_num):
        batch_idx_list.append(perm_id[i*batch_size:min(total_num, (i+1)*batch_size)])
    return batch_idx_list


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_name: str = "cifar10_1024_4class"
    ) -> None:

        super(CIFAR10Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if data_name != 'whole_cifar10':
            sampled_filepath = os.path.join(root, "sampled_cifar10", "{}.pkl".format(data_name))
            with open(sampled_filepath, "rb") as f:
                sampled_data = pickle.load(f)
            if train:
                self.data = sampled_data["train_data"]
                self.targets = np.array(sampled_data["train_targets"])
            else:
                self.data = sampled_data["test_data"]
                self.targets = np.array(sampled_data["test_targets"])
        
        else:
            self.targets = np.array(self.targets)

        # print("class_to_idx", self.class_to_idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR10Triple(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_name: str = "cifar10_1024_4class"
    ) -> None:

        super(CIFAR10Triple, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        if data_name != 'whole_cifar10':
            sampled_filepath = os.path.join(root, "sampled_cifar10", "{}.pkl".format(data_name))
            with open(sampled_filepath, "rb") as f:
                sampled_data = pickle.load(f)
            if train:
                self.data = sampled_data["train_data"]
                self.targets = np.array(sampled_data["train_targets"])
            else:
                self.data = sampled_data["test_data"]
                self.targets = np.array(sampled_data["test_targets"])
        
        else:
            self.targets = np.array(self.targets)

        # print("class_to_idx", self.class_to_idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_org = ToTensor_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, pos_org, target


class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_name: str = "whole_cifar100"
    ) -> None:

        super(CIFAR100Pair, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = np.array(self.targets)

        # print("class_to_idx", self.class_to_idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR100Triple(CIFAR100):
    """CIFAR100 Dataset.
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        data_name: str = "whole_cifar100"
    ) -> None:

        super(CIFAR100Triple, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.targets = np.array(self.targets)

        # print("class_to_idx", self.class_to_idx)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_org = ToTensor_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, pos_org, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
])

train_diff_transform = nn.Sequential(
    Kaug.RandomResizedCrop([32,32]),
    Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    Kaug.RandomGrayscale(p=0.2)
)

train_diff_transform2 = nn.Sequential(
    # Kaug.RandomResizedCrop([32,32]),
    # Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1),
    # Kaug.RandomGrayscale(p=0.2)
)

train_diff_transform3 = nn.Sequential(
    Kaug.RandomResizedCrop([32,32]),
    # Kaug.RandomHorizontalFlip(p=0.5),
    Kaug.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
    # Kaug.RandomGrayscale(p=0.2)
)

def train_diff_transform_prob(p_recrop=0.0, p_hflip=0.0, p_cj=0.0, p_gray=0.0, s_cj=1):
    # simclr: 1.0 0.5 0.8 0.2
    return nn.Sequential(
            Kaug.RandomResizedCrop([32,32], p=p_recrop),
            Kaug.RandomHorizontalFlip(p=p_hflip),
            Kaug.ColorJitter(0.4*s_cj, 0.4*s_cj, 0.4*s_cj, 0.1*s_cj, p=p_cj),
            Kaug.RandomGrayscale(p=p_gray)
        )

def plot_loss(file_prename):
    pd_reader = pd.read_csv(file_prename+".csv")

    epoch = pd_reader.values[:,0]
    loss = pd_reader.values[:,1]
    acc = pd_reader.values[:,2]
    acc_top5 = pd_reader.values[:,3]
    best_test_acc = pd_reader.values[:,4]
    best_test_acc_loss = pd_reader.values[:,5]
    best_train_loss_acc = pd_reader.values[:,6]
    best_train_loss = pd_reader.values[:,7]

    fig, ax=plt.subplots(1,1,figsize=(9,6))
    ax1 = ax.twinx()

    p2 = ax.plot(epoch, loss,'r-', label = 'loss')
    ax.legend()
    p3 = ax1.plot(epoch, acc, 'b-', label = 'test_acc')
    ax1.legend()

    #显示图例
    # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
    # plt.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax1.set_ylabel('acc')
    plt.title('Training loss on generating model and clean test acc')
    plt.savefig(file_prename + ".png")
    plt.close()

    fig, ax=plt.subplots(1,1,figsize=(9,6))

    p2 = ax.plot(epoch, best_train_loss_acc,'r-', label = 'loss')
    ax.legend()
    p3 = ax.plot(epoch, best_test_acc, 'b-', label = 'test_acc')
    ax.legend()

    #显示图例
    # p3 = pl.plot(epoch,acc, 'b-', label = 'test_acc')
    # plt.legend()
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    plt.title('Best test acc and acc when lowest train loss')
    plt.savefig(file_prename + "_2_acc.png")
    plt.close()

def plot_mass_candidate(net, new_batch_idx_list, data_loader, save_name_pre, batch_size):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    # feature_bank = np.concatenate(feature_bank, axis=0)
    batch_feature_bank = []
    batch_plot_labels = []
    for batch_idx in new_batch_idx_list:
        pos_1, targets = data_loader.get_batch(batch_idx)
        pos_1, targets = pos_1.cuda(), targets.cuda()
        feature, out = net(pos_1)
        batch_feature_bank.append(feature.detach().cpu().numpy())
        batch_plot_labels.append(targets.detach().cpu().numpy())
    feature_bank = np.concatenate(batch_feature_bank, axis=0)
    plot_labels = np.concatenate(batch_plot_labels, axis=0)

    c = np.max(plot_labels) + 1
    
    feature_tsne_input = feature_bank
    plot_labels_colar = plot_labels
    feature_tsne_output = tsne.fit_transform(feature_tsne_input)
    
    coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
    coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

    cm = plt.cm.get_cmap('gist_rainbow')
    z = np.arange(c)
    my_cmap = cm(z)
    my_cmap = ListedColormap(my_cmap)

    marker = ['o', 'x', 'v', 'd']
    color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'chartreuse', 'cyan', 'sage', 'coral', 'gold', 'plum', 'sienna', 'teal']

    for i,batch_id in enumerate(new_batch_idx_list):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("\n max:{} min:{}".format(coord_max, coord_min))

        x_pos_1 = feature_tsne_output[:, 0]
        y_pos_1 = feature_tsne_output[:, 1]
        plot_labels_colar = plot_labels

        # linewidths

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar, cmap=cm)

        x_pos_1 = feature_tsne_output[i*batch_size:(1+i)*batch_size, 0]
        y_pos_1 = feature_tsne_output[i*batch_size:(1+i)*batch_size, 1]
        plot_labels_colar = plot_labels[i*batch_size:(1+i)*batch_size]

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=20, marker='o', c=plot_labels_colar, cmap=cm, linewidths=3, edgecolors='k')

        plt.xlim((coord_min, coord_max))
        plt.ylim((coord_min, coord_max))
        if not os.path.exists('./plot_mass_candidate/{}'.format(save_name_pre)):
            os.mkdir('./plot_mass_candidate/{}'.format(save_name_pre))
        plt.savefig('./plot_mass_candidate/{}/{}.png'.format(save_name_pre, i))
        plt.close()

def plot_kmeans(feature_bank, GT_label, save_name_pre, kmeans_labels_list):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    # # feature_bank = np.concatenate(feature_bank, axis=0) get_batch_idx_group
    # batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=128, shuffle=False, drop_last=False)
    # batch_feature_bank = []
    # batch_plot_labels = []
    
    # for batch_idx in batch_idx_list:
    #     pos_1, targets = data_loader.get_batch(batch_idx)
    #     pos_1, targets = pos_1.cuda(), targets.cuda()
    #     feature, out = net(pos_1)
    #     batch_feature_bank.append(feature)
    #     # batch_feature_bank.append(feature.detach().cpu().numpy())
    #     batch_plot_labels.append(targets.detach().cpu().numpy())
    # feature_bank = torch.cat(batch_feature_bank, dim=0)
    # plot_labels = np.concatenate(batch_plot_labels, axis=0)

    labels_list = [GT_label] + kmeans_labels_list

    # for i, n_cluster in enumerate(num_clusters):
    #     kmeans_labels, cluster_centers = kmeans(X=feature_bank, num_clusters=n_cluster, distance='euclidean', device=feature_bank.device, tqdm_flag=False)
    #     kmeans_labels_list.append(kmeans_labels.detach().cpu().numpy())

    for i, labels in enumerate(labels_list):
        
        feature_tsne_input = feature_bank.detach().cpu().numpy()
        if i == 0:
            if torch.is_tensor(labels):
                plot_labels_colar = labels.detach().cpu().numpy()
            else:
                plot_labels_colar = labels
        else:
            plot_labels_colar = labels.detach().cpu().numpy()
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        c = np.max(plot_labels_colar) + 1
        
        coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
        coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

        cm = plt.cm.get_cmap('gist_rainbow')
        z = np.arange(c)
        my_cmap = cm(z)
        my_cmap = ListedColormap(my_cmap)

        marker = ['o', 'x', 'v', 'd']
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'chartreuse', 'cyan', 'sage', 'coral', 'gold', 'plum', 'sienna', 'teal']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("\n max:{} min:{}".format(coord_max, coord_min))

        x_pos_1 = feature_tsne_output[:, 0]
        y_pos_1 = feature_tsne_output[:, 1]
        # plot_labels_colar = labels.detach().cpu().numpy()

        # linewidths

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar, cmap=cm)

        plt.xlim((coord_min, coord_max))
        plt.ylim((coord_min, coord_max))
        if not os.path.exists('./plot_kmeans/{}'.format(save_name_pre)):
            os.mkdir('./plot_kmeans/{}'.format(save_name_pre))
        if i == 0:
            plt.savefig('./plot_kmeans/{}/{}_{}.png'.format(save_name_pre, 'GT', save_name_pre))
        else:
            plt.savefig('./plot_kmeans/{}/{}_{}.png'.format(save_name_pre, c, save_name_pre))
        plt.close()

def plot_kmeans_train_test(feature_bank, GT_label, save_name_pre, kmeans_labels_list, n_train):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    # # feature_bank = np.concatenate(feature_bank, axis=0) get_batch_idx_group
    # batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=128, shuffle=False, drop_last=False)
    # batch_feature_bank = []
    # batch_plot_labels = []
    
    # for batch_idx in batch_idx_list:
    #     pos_1, targets = data_loader.get_batch(batch_idx)
    #     pos_1, targets = pos_1.cuda(), targets.cuda()
    #     feature, out = net(pos_1)
    #     batch_feature_bank.append(feature)
    #     # batch_feature_bank.append(feature.detach().cpu().numpy())
    #     batch_plot_labels.append(targets.detach().cpu().numpy())
    # feature_bank = torch.cat(batch_feature_bank, dim=0)
    # plot_labels = np.concatenate(batch_plot_labels, axis=0)

    labels_list = [GT_label] + kmeans_labels_list

    # for i, n_cluster in enumerate(num_clusters):
    #     kmeans_labels, cluster_centers = kmeans(X=feature_bank, num_clusters=n_cluster, distance='euclidean', device=feature_bank.device, tqdm_flag=False)
    #     kmeans_labels_list.append(kmeans_labels.detach().cpu().numpy())

    for i, labels in enumerate(labels_list):
        
        feature_tsne_input = feature_bank.detach().cpu().numpy()
        if i == 0:
            plot_labels_colar = labels
        else:
            plot_labels_colar = labels.detach().cpu().numpy()
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        c = np.max(plot_labels_colar) + 1
        
        coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
        coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

        cm = plt.cm.get_cmap('gist_rainbow')
        z = np.arange(c)
        my_cmap = cm(z)
        my_cmap = ListedColormap(my_cmap)

        marker = ['o', 'x', 'v', 'd']
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'chartreuse', 'cyan', 'sage', 'coral', 'gold', 'plum', 'sienna', 'teal']

        fig = plt.figure(figsize=(16, 8))
        # ax = fig.add_subplot(1, 2, 1)
        plt.subplot(1, 2, 1)
        plt.title("\n max:{} min:{}".format(coord_max, coord_min))

        x_pos_1 = feature_tsne_output[:n_train, 0]
        y_pos_1 = feature_tsne_output[:n_train, 1]

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar[:n_train], cmap=cm)

        plt.xlim((coord_min, coord_max))
        plt.ylim((coord_min, coord_max))

        # ax = fig.add_subplot(1, 2, 2)
        plt.subplot(1, 2, 2)
        plt.title("\n max:{} min:{}".format(coord_max, coord_min))

        x_pos_1 = feature_tsne_output[n_train:, 0]
        y_pos_1 = feature_tsne_output[n_train:, 1]

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar[n_train:], cmap=cm)

        plt.xlim((coord_min, coord_max))
        plt.ylim((coord_min, coord_max))

        if not os.path.exists('./plot_kmeans/{}'.format(save_name_pre)):
            os.mkdir('./plot_kmeans/{}'.format(save_name_pre))
        if i == 0:
            plt.savefig('./plot_kmeans/{}/{}_{}.png'.format(save_name_pre, 'GT', save_name_pre))
        else:
            plt.savefig('./plot_kmeans/{}/{}_{}.png'.format(save_name_pre, c, save_name_pre))
        plt.close()

def plot_kmeans_epoch(feature_bank, GT_label, save_name_pre, kmeans_labels_list, epoch, plot_num):

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

    # # feature_bank = np.concatenate(feature_bank, axis=0) get_batch_idx_group
    # batch_idx_list = get_batch_idx_group(data_loader.data_source.data.shape[0], batch_size=128, shuffle=False, drop_last=False)
    # batch_feature_bank = []
    # batch_plot_labels = []
    
    # for batch_idx in batch_idx_list:
    #     pos_1, targets = data_loader.get_batch(batch_idx)
    #     pos_1, targets = pos_1.cuda(), targets.cuda()
    #     feature, out = net(pos_1)
    #     batch_feature_bank.append(feature)
    #     # batch_feature_bank.append(feature.detach().cpu().numpy())
    #     batch_plot_labels.append(targets.detach().cpu().numpy())
    # feature_bank = torch.cat(batch_feature_bank, dim=0)
    # plot_labels = np.concatenate(batch_plot_labels, axis=0)

    labels_list = [GT_label] + kmeans_labels_list

    # for i, n_cluster in enumerate(num_clusters):
    #     kmeans_labels, cluster_centers = kmeans(X=feature_bank, num_clusters=n_cluster, distance='euclidean', device=feature_bank.device, tqdm_flag=False)
    #     kmeans_labels_list.append(kmeans_labels.detach().cpu().numpy())

    for i, labels in enumerate(labels_list):
        
        feature_tsne_input = feature_bank[:plot_num].detach().cpu().numpy()
        if i == 0:
            plot_labels_colar = labels[:plot_num]
        else:
            plot_labels_colar = labels.detach().cpu().numpy()[:plot_num]
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        c = np.max(plot_labels_colar) + 1
        
        coord_min = math.floor(np.min(feature_tsne_output) / 1) * 1
        coord_max = math.ceil(np.max(feature_tsne_output) / 1) * 1

        cm = plt.cm.get_cmap('gist_rainbow')
        z = np.arange(c)
        my_cmap = cm(z)
        my_cmap = ListedColormap(my_cmap)

        marker = ['o', 'x', 'v', 'd']
        color_map = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'chartreuse', 'cyan', 'sage', 'coral', 'gold', 'plum', 'sienna', 'teal']

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("\n max:{} min:{}".format(coord_max, coord_min))

        x_pos_1 = feature_tsne_output[:, 0]
        y_pos_1 = feature_tsne_output[:, 1]
        # plot_labels_colar = labels.detach().cpu().numpy()

        # linewidths

        aug1 = plt.scatter(x_pos_1, y_pos_1, s=15, marker='o', c=plot_labels_colar, cmap=cm)

        plt.xlim((coord_min, coord_max))
        plt.ylim((coord_min, coord_max))
        if not os.path.exists('./plot_kmeans/{}'.format(save_name_pre)):
            os.mkdir('./plot_kmeans/{}'.format(save_name_pre))
        if not os.path.exists('./plot_kmeans/{}/{}'.format(save_name_pre, epoch)):
            os.mkdir('./plot_kmeans/{}/{}'.format(save_name_pre, epoch))
        if i == 0:
            plt.savefig('./plot_kmeans/{}/{}/{}_{}.png'.format(save_name_pre, epoch, 'GT', save_name_pre))
        else:
            plt.savefig('./plot_kmeans/{}/{}/{}_{}.png'.format(save_name_pre, epoch, c, save_name_pre))
        plt.close()

def test_instance_sim(net, memory_data_loader, test_data_loader, augmentation_prob, cj_strength):
    net.eval()
    total_top1, total_top5, total_num, feature_bank1 = 0.0, 0.0, 0, []
    feature_bank1, feature_bank2, feature_bank, sim_list = [], [], [], []
    c = 4
    if np.sum(augmentation_prob) == 0:
        my_transform_func = train_diff_transform
    else:
        my_transform_func = train_diff_transform_prob(*augmentation_prob, cj_strength)
        
    with torch.no_grad():
        
        feature_bank = []
        posiness = []
        for i in range(3):
            data_iter = iter(test_data_loader)
            end_of_iteration = "END_OF_ITERATION"
            total_top1, total_top5, total_num = 0.0, 0.0, 0.0
            feature_bank1, feature_bank2, = [], []

            for pos_samples_1, pos_samples_2, labels in tqdm(test_data_loader, desc='Feature extracting'):

                pos_samples_1, pos_samples_2, labels = pos_samples_1.cuda(non_blocking=True), pos_samples_2.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                target = torch.arange(0, pos_samples_1.shape[0]).cuda(non_blocking=True)

                net.eval()
                pos_samples_1 = my_transform_func(pos_samples_1)
                pos_samples_2 = my_transform_func(pos_samples_2)
                feature1, out1 = net(pos_samples_1)
                feature2, out2 = net(pos_samples_2)
                feature_bank1.append(feature1)
                feature_bank2.append(feature2)
                
            feature1 = torch.cat(feature_bank1, dim=0).contiguous()
            feature2 = torch.cat(feature_bank2, dim=0).contiguous()
    
            target = torch.arange(0, feature1.shape[0]).cuda(non_blocking=True)
            
            # compute cos similarity between each two groups of augmented samples ---> [B, B]
            sim_matrix = torch.mm(feature1, feature2.t())
            pos_sim = torch.sum(feature1 * feature2, dim=-1)
            
            mask2 = (torch.ones_like(sim_matrix) - torch.eye(feature1.shape[0], device=sim_matrix.device)).bool()
            # [B, B-1]
            neg_sim_matrix2 = sim_matrix.masked_select(mask2).view(feature1.shape[0], -1)
            sim_weight, sim_indices = neg_sim_matrix2.topk(k=1, dim=-1)
            posiness.append(pos_sim - sim_weight.squeeze(1))
            
            sim_indice_1 = sim_matrix.argsort(dim=0, descending=True) #[B, B]
            sim_indice_2 = sim_matrix.argsort(dim=1, descending=True) #[B, B]
            # print(sim_indice_1[0, :30])
            # print(sim_indice_2[:30, 0])

            total_top1 += torch.sum((sim_indice_1[:1, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top1 += torch.sum((sim_indice_2[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_1[:5, :].t() == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum((sim_indice_2[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += feature1.shape[0] * 2
        
        # print(total_top1 / total_num * 100, total_top5 / total_num * 100, )
        posiness = torch.stack(posiness, dim = 1).mean(dim=1)
        easy_weight_50, easy_50 = posiness.topk(k=50, dim=0)
        # print(easy_weight_50)
        # input(easy_50)

    return total_top1 / total_num * 100, total_top5 / total_num * 100

def run_kmeans(x, num_clusters, device, temperature):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    
    for seed, num_cluster in enumerate(num_clusters):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

        clus.train(x, index)   

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]
        
        # get cluster centroids
        # centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
        Dcluster = [[] for c in range(k)]          
        for im,i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
        density = np.zeros(k)
        for i,dist in enumerate(Dcluster):
            if len(dist)>1:
                d = (np.asarray(dist)**0.5).mean()
                density[i] = d     
                
        # #if cluster only has one point, use the max to estimate its concentration        
        # dmax = density.max()
        # for i,dist in enumerate(Dcluster):
        #     if len(dist)<=1:
        #         density[i] = dmax

        # density = density.clip(0, np.percentile(density,90)) #clamp extreme values for stability
        # density = temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
        # centroids = torch.Tensor(centroids).cuda()
        # centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()               
        density = torch.Tensor(density).cuda()
        
        # results['centroids'].append(centroids)
        # results['density'].append(density)
        results['im2cluster'].append(im2cluster)    
        
    return results

def get_np_mean_dbindex(train_data, train_targets):
    # print(np.max(train_targets))
    c = np.max(train_targets) + 1
    class_center = []
    intra_class_dis = []

    # index_range = range(20,30)

    for i in range(c):
        idx_i = np.where(train_targets == i)[0]
        class_i = train_data[idx_i,:]
        class_i_center = train_data[idx_i].mean(axis=0)[:]
        class_center.append(class_i_center)
        intra_class_dis.append(np.mean(np.sqrt(np.sum((class_i-class_i_center)**2, axis=1))))


    class_center = np.stack(class_center, axis=0)
    class_sim = class_center @ class_center.transpose()
    class_dis = np.zeros(shape=class_sim.shape)

    for i in range(class_center.shape[0]):
        for j in range(class_center.shape[0]):
            dis = np.sqrt(np.sum((class_center[i] - class_center[j]) ** 2))
            class_dis[i,j] = dis

    # print(class_dis)

    DBindex = []
    for i in range(class_center.shape[0]):
        index = []
        for j in range(class_center.shape[0]):
            if j != i:
                index.append((intra_class_dis[i] + intra_class_dis[j]) / class_dis[i,j])
        DBindex.append(np.mean(index))

    # print(DBindex)
    DBindex = np.mean(DBindex)
    return DBindex

def get_center(train_data, kmean_result, num_clusters, curriculum):
    # print(np.max(train_targets))
    if curriculum == 'DBindex_cluster_GT':
        train_targets = kmean_result
        centroids = []
        c = np.max(train_targets) + 1
        class_center = []

        for i in range(c):
            idx_i = np.where(train_targets == i)[0]
            # class_i = train_data[idx_i,:]
            class_i_center = train_data[idx_i].mean(axis=0)[:]
            class_center.append(class_i_center)

        class_center = np.stack(class_center, axis=0)
        class_center = torch.tensor(class_center).cuda()
        class_center = nn.functional.normalize(class_center, p=2, dim=1)
        centroids.append(class_center)
    else:
        centroids = []
        for num_cluster_idx in range(len(num_clusters)):
            train_targets = kmean_result['im2cluster'][num_cluster_idx].detach().cpu().numpy()

            c = np.max(train_targets) + 1
            class_center = []

            for i in range(c):
                idx_i = np.where(train_targets == i)[0]
                # class_i = train_data[idx_i,:]
                class_i_center = train_data[idx_i].mean(axis=0)[:]
                class_center.append(class_i_center)

            class_center = np.stack(class_center, axis=0)
            centroids.append(torch.tensor(class_center).cuda())

    return centroids

def get_reorder_point_2_center_diss(sample, kmean_result, num_clusters, start_epoch, epoch, epochs, piermaro_whole_epoch, final_percent):
    
    if piermaro_whole_epoch == '':
        whole_epoch = epochs
    else:
        whole_epoch = int(piermaro_whole_epoch)
    if final_percent == 0.5:
        percent_high_conf = 0.5 * float(epoch - start_epoch) / float(whole_epoch - start_epoch)
        # input('check')
    else:
        percent_high_conf = final_percent
    high_conf_label_list = []
    for num_cluster_idx in range(len(num_clusters)):
        cluster_label = kmean_result['im2cluster'][num_cluster_idx]
        cluster_label = cluster_label.detach().cpu().numpy()
        high_conf_label = np.zeros(cluster_label.shape)
        class_center = []
        intra_class_dis = []
        point_dis_to_center_list = []
        c = np.max(cluster_label) + 1
        for i in range(c):
            idx_i_wholeset = np.where(cluster_label == i)[0]
            class_i = sample[idx_i_wholeset, :]
            class_i_center = kmean_result['centroids'][num_cluster_idx][i].detach().cpu().numpy()
            if idx_i_wholeset.shape[0] == 0:
                continue
            class_center.append(class_i_center)
            point_dis_to_center = np.sqrt(np.sum((class_i-class_i_center)**2, axis = 1))
            if point_dis_to_center.shape[0] == 0:
                class_center.pop()
                continue

            num_high_conf = int(percent_high_conf * point_dis_to_center.shape[0])
            
            idx_high_conf_class = np.argpartition(point_dis_to_center, num_high_conf)[:num_high_conf]
            idx_low_conf_class = np.argpartition(point_dis_to_center, num_high_conf)[num_high_conf:]
            # print(idx_high_conf_class.shape)
            # print(point_dis_to_center.shape)
            # print(point_dis_to_center[idx_low_conf_class].mean())
            # print(point_dis_to_center[idx_high_conf_class].mean())
            # print(np.min(point_dis_to_center[idx_low_conf_class]))
            # print(np.max(point_dis_to_center[idx_high_conf_class]))
            # input()
            idx_high_conf_class_of_wholeset = idx_i_wholeset[idx_high_conf_class]
            high_conf_label[idx_high_conf_class_of_wholeset] = 1
        
        high_conf_label_list.append(torch.LongTensor(high_conf_label).cuda())

    return high_conf_label_list
