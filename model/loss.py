import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, reduction=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(
            1,
            targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
        return loss


def cross_entropy(soft_targets, pred, reduction='none'):
    if reduction == 'none':
        return torch.sum(- soft_targets * torch.log(pred), 1)
    elif reduction == 'mean':
        return torch.mean(torch.sum(- soft_targets * torch.log(pred), 1))


def compute_dist(true_score, pred_score, type='l2'):
    "Input: score/output from Softmax Layer"
    if type == 'l2':
        return torch.norm(true_score - pred_score, dim=-1)
    elif type == 'ce':
        return cross_entropy(true_score, pred_score)
    elif type == 'sce':  # symmetric cross entropy
        return cross_entropy(true_score, pred_score) + cross_entropy(pred_score, true_score)


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta

    def forward(self, labels, pred, reduction='mean', rce_weight=None):
        # CCE
        ce = cross_entropy(labels, pred, reduction=reduction)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels = torch.clamp(labels, min=1e-4, max=1.0)
        rce = cross_entropy(pred, labels, reduction=reduction)

        # Loss
        if rce_weight is not None:
            loss = self.alpha * ce + rce_weight * rce 
        else:
            loss = self.alpha * ce + self.beta * rce  
        return loss


def compute_loss(targets, pred, type='ce', reduction='mean', weight=None, cls_weight=None, rce_weight=None, soft_flag=False):
    batch_size, num_cls = pred.size()
    if targets.ndim < 2:
        ones = torch.eye(num_cls, device=pred.device)
        targets_2d = torch.index_select(ones, dim=0, index=targets)
        targets_1d = targets
        if soft_flag: # no soft label to use
            print('Warning: no soft_label provided')
    else:
        targets_2d = targets
        targets_1d = targets_2d.argmax(dim=1)
        if not soft_flag:  # use hard label
            ones = torch.eye(num_cls, device=pred.device)
            targets_2d = torch.index_select(ones, dim=0, index=targets_1d)

    if type == 'ce':
        loss = cross_entropy(targets_2d, pred, reduction=reduction)
    elif type == 'sce':
        loss_criterion = SCELoss()
        loss = loss_criterion(targets_2d, pred, reduction=reduction, rce_weight = rce_weight)
    elif type == 'dot' or type == 'dot_d':
        loss = torch.sum(- targets_2d * pred, 1)

    if weight is not None:
        loss = loss * weight
    if cls_weight is not None: 
        elementwise_weight = cls_weight[targets_1d]
        loss = loss * elementwise_weight
        
    loss = loss.sum() / batch_size
    return loss