from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import kornia.augmentation as Kaug
import torch.nn as nn
import os
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


ToTensor_transform = transforms.Compose([
    transforms.ToTensor(),
])


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
        sampled_filepath = os.path.join(root, "sampled_cifar10", "{}.pkl".format(data_name))
        with open(sampled_filepath, "rb") as f:
            sampled_data = pickle.load(f)
        if train:
            self.data = sampled_data["train_data"]
            self.targets = np.array(sampled_data["train_targets"])
        else:
            self.data = sampled_data["test_data"]
            self.targets = np.array(sampled_data["test_targets"])

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