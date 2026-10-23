import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.autograd import Function
from torch.utils.data import DataLoader
import time
from utils import split_domain, partition_into_three, PackedDataset


class FeatureExtractor(nn.Module):
    """
    FeatureExtractor

    """

    def __init__(self, fea_len, hidden_1, hidden_2):
        super(FeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(fea_len, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class DomainDisentangler(nn.Module):
    """
    domian feature decoupler
    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class ClassDisentangler(nn.Module):
    """
    class feature decoupler
    """

    def __init__(self, shallow_fea, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(shallow_fea, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        # x = F.relu(x)
        x = F.leaky_relu(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class CommonDiscriminator(nn.Module):
    """
    discriminator
    """

    def __init__(self, hidden_1, hidden_2, class_num):
        super(CommonDiscriminator, self).__init__()
        self.fc1 = nn.Linear(hidden_1, hidden_2)
        self.fc2 = nn.Linear(hidden_2, class_num)
        self.dropout1 = nn.Dropout(p=0.25)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        # x = F.relu(x)
        x=F.leaky_relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

    def get_parameters(self) -> List[Dict]:
        params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
        ]
        return params

class ReverseLayerF(Function):
    """
    Gradient reversal layer, for dann
    such as：
        disc_in1 = Adver_network.ReverseLayerF.apply(z1, self.args.alpha1)    # alpha1=0.1
        disc_out1 = self.ddiscriminator(disc_in1)
        disc_loss = F.cross_entropy(disc_out1, all_d1, reduction='mean')

    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None




