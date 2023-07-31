import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os.path as osp
from sklearn.neighbors import KNeighborsClassifier


class FocalLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(FocalLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.softmax(inputs)

        tmp = log_probs[range(log_probs.shape[0]), targets]  # batch
        if self.use_gpu:
            targets = targets.cuda()
        # targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-((1 - tmp) ** 2) * torch.log(tmp)).mean(0).sum()
        else:
            loss = (-((1 - tmp) ** 2) * torch.log(tmp)).sum(1)
        return loss


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1, targets.unsqueeze(1).cpu(), 1
        )
        if self.use_gpu:
            targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        if self.size_average:
            loss = (-targets * log_probs).mean(0).sum()
        else:
            loss = (-targets * log_probs).sum(1)
        return loss


def cal_acc_(loader, netF, netC):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            output_f = netF.forward(inputs)  # a^t
            outputs = netC(output_f)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(
        all_label.size()[0]
    )
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy, mean_ent

