import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, resnet18


class Model(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18'):
        super(Model, self).__init__()

        self.f = []
        if arch == 'resnet18':
            backbone = resnet18()
            encoder_dim = 512
        elif arch == 'resnet50':
            backbone = resnet50()
            encoder_dim = 2048
        else:
            raise NotImplementedError

        for name, module in backbone.named_children():
            if name == 'conv1' and cifar_head == True:
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(encoder_dim, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        # print(F.normalize(feature, dim=-1))
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        # return feature, F.normalize(out, dim=-1)

class momentum_Model(nn.Module):
    def __init__(self, feature_dim=128, cifar_head=True, arch='resnet18', m=0.999):
        super(momentum_Model, self).__init__()
        self.model = Model(feature_dim, cifar_head, arch)
        self.key_model = Model(feature_dim, cifar_head, arch)
        self.m = m

        for param_q, param_k in zip(self.model.parameters(), self.key_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
    
    def forward(self, x):
        return self.model(x)

    def momentum_encoder(self, x):
        # input("momentum_encoder")
        return self.key_model(x)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model.parameters(), self.key_model.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        # input("_momentum_update_key_encoder")

    @torch.no_grad()
    def restore_k_with_q(self):
        for param_q, param_k in zip(self.model.parameters(), self.key_model.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        
        # input("restore_k_with_q")
        