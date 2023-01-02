import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList
import random, pdb, math, copy
from loss import CrossEntropyLabelSmooth
from sklearn.metrics import confusion_matrix
from train_src import cal_acc, data_load
from utils import *
#========================================================================================
parser = argparse.ArgumentParser(description='Neighbors')
parser.add_argument('--gpu_id', type=str, nargs='?', default='9', help="device id to run")
parser.add_argument('--s', type=int, default=0, help="source")
parser.add_argument('--t', type=int, default=1, help="target")
parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
parser.add_argument('--batch_size', type=int, default=16, help="batch_size")
parser.add_argument('--worker', type=int, default=0, help="number of workers")
parser.add_argument('--dset', type=str,default='visda-2017')
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
parser.add_argument('--net', type=str, default='resnet50', help="resnet50, resnet101")
parser.add_argument('--seed', type=int, default=2020, help="random seed")
parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--epsilon', type=float, default=1e-5)
parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
parser.add_argument('--smooth', type=float, default=0.1)
parser.add_argument('--output', type=str, default='weight/source/')
parser.add_argument('--da', type=str, default='uda')
parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
args = parser.parse_args([])

if args.dset == 'office-home':
    names = ['Art', 'Clipart', 'Product', 'RealWorld']
    args.class_num = 65
if args.dset == 'visda-2017':
    names = ['train', 'validation']
    args.class_num = 12

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

folder = '../dataset/'
args.s_dset_path = folder + args.dset + '/' + names[args.s] + '/image_list.txt'
args.test_dset_path = folder + args.dset + '/' + names[args.t] + '/image_list.txt'

args.output_dir_src = osp.join(args.output, args.da, args.dset, names[args.s][0].upper())
args.name_src = names[args.s][0].upper()
if not osp.exists(args.output_dir_src):
    os.system('mkdir -p ' + args.output_dir_src)
if not osp.exists(args.output_dir_src):
    os.mkdir(args.output_dir_src)

#=========================================
dset_loaders = data_load(args)    # ['source_tr', 'source_te', 'test']

## set base network
netF = network.ResBase(res_name=args.net)
netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck)
netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck)

param_group = []
learning_rate = args.lr
for k, v in netF.named_parameters():
    param_group += [{'params': v, 'lr': learning_rate*0.1}]
for k, v in netB.named_parameters():
    param_group += [{'params': v, 'lr': learning_rate}]
for k, v in netC.named_parameters():
    param_group += [{'params': v, 'lr': learning_rate}]
optimizer = optim.SGD(param_group)
optimizer = op_copy(optimizer)

acc_init = 0
max_iter = args.max_epoch * len(dset_loaders["source_tr"])
interval_iter = max_iter // 10
iter_num = 0

netF.train()
netB.train()
netC.train()
