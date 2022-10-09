from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


class CIFAR100Pair(CIFAR100):
    """CIFAR100 Dataset.
    """

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
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

def test_ssl_visualization(net, test_data_visualization, save_name, model2=None):
    net.eval()
    c = 10
    feature_bank = []
    feature_bank2 = []
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    with torch.no_grad():
        test_data_visualization_loader = DataLoader(test_data_visualization, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
        # generate feature bank
        for data, _, target in tqdm(test_data_visualization_loader, desc='Feature extracting'):
            feature, out = net(data.cuda(non_blocking=True))
            feature_bank.append(feature)
            if model2 != None:
                feature, out = model2(data.cuda(non_blocking=True))
                feature_bank2.append(feature)
            if len(feature_bank) >= 2:
                break
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        if model2 != None:
            feature_bank2 = torch.cat(feature_bank2, dim=0).contiguous()
            feature_bank = torch.cat([feature_bank, feature_bank2], dim=0).contiguous()
        # [N]
        feature_labels = torch.tensor(test_data_visualization_loader.dataset.targets, device=feature_bank.device)[:1024]
        feature_tsne_input = feature_bank.cpu().numpy()
        labels_tsne_color = feature_labels.cpu().numpy()
        if model2 != None:
            labels_tsne_color = np.concatenate([labels_tsne_color, labels_tsne_color], axis=0)
        feature_tsne_output = tsne.fit_transform(feature_tsne_input)
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        plt.title("clean data with original label")
        cm = plt.cm.get_cmap('gist_rainbow', c)
        plt.scatter(feature_tsne_output[:1024, 0], feature_tsne_output[:1024, 1], s=10, c=labels_tsne_color[:1024], cmap=cm)
        plt.scatter(feature_tsne_output[1024:, 0], feature_tsne_output[1024:, 1], s=10, c=labels_tsne_color[1024:], cmap=cm, marker='+')
        ax.xaxis.set_major_formatter(NullFormatter())  # 设置标签显示格式为空
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.savefig('./visualization/cleandata_orglabel_{}.png'.format(save_name))

    return 
