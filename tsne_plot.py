# encoding:utf-8
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import network, moco
from dataset.data_list import ImageList
from dataset.visda_data import data_load, image_train, moco_transform, mm_transform
from model.model_util import bn_adapt, bn_adapt1, label_propagation, extract_feature_labels, extract_features
from model.loss import compute_loss
from dataset.data_transform import TransformSW
from utils import cal_acc, print_args, log, set_log_path
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# ================== t-SNE ==================

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


before = np.load('before.npz')
feats = before['feats']
labels = before['labels']

after = np.load('after.npz')
feats = after['feats']
labels = after['labels']


tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(feats)
tsne.kl_divergence_


fig = plt.scatter(x=x_tsne[:, 0], y=x_tsne[:, 1], c=labels, cmap='jet', marker='.', alpha=0.1)
fig.update_layout(
    title="t-SNE visualization of Custom Classification dataset",
    xaxis_title="First t-SNE",
    yaxis_title="Second t-SNE",
)
fig.show()


x2_tsne = np.load('after_tsne.npz')
x2_tsne = x2_tsne['tsne']
fig = plt.scatter(x=x2_tsne[:, 0], y=x2_tsne[:, 1], c=labels, cmap='jet', marker='.', alpha=0.1)


x1_tsne = np.load('before_tsne.npz')
x1_tsne = x1_tsne['tsne']
fig = plt.scatter(x=x1_tsne[:, 0], y=x1_tsne[:, 1], c=labels, cmap='jet', marker='.', alpha=0.1)




# ================== config ==================

parser = argparse.ArgumentParser(description='Neighbors')
parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
parser.add_argument('--s', type=int, default=0, help="source")
parser.add_argument('--t', type=int, default=1, help="target")
parser.add_argument('--max_epoch', type=int, default=30, help="max iterations")
parser.add_argument('--interval', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
parser.add_argument('--worker', type=int, default=2, help="number of workers")
parser.add_argument('--dset', type=str, default='visda-2017')
parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
parser.add_argument('--net', type=str, default='resnet101')
parser.add_argument('--seed', type=int, default=2021, help="random seed")

parser.add_argument('--bottleneck', type=int, default=256)
parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
parser.add_argument('--bn_adapt', type=int, default=0, help='Whether to first finetune mu and std in BN layers')

parser.add_argument('--lp_type', type=float, default=0,
                    help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
parser.add_argument('--T_decay', type=float, default=0.8,
                    help='Temperature decay of creating pseudo-label in feature extraction')
parser.add_argument('--feat_type', type=str, default='cls', choices=['cls', 'teacher', 'student'])
parser.add_argument('--nce_wt', type=float, default=1.0, help='weight for nce loss')
parser.add_argument('--nce_wt_decay', type=float, default=0.0, help='0.0:no decay, larger value faster decay')

parser.add_argument('--loss_type', type=str, default='dot', help='Loss function', choices=['ce', 'sce', 'dot', 'dot_d'])
parser.add_argument('--loss_wt', type=str, default='en5', help='CE/SCE loss weight: e|p|n, c|n (classwise weight), 0-9')
parser.add_argument('--plabel_soft', action='store_false', help='Whether to use soft/hard pseudo label')
parser.add_argument("--beta", type=float, default=5.0)
parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument('--data_aug', type=str, default='0.2,0.5', help='delimited list input')
parser.add_argument('--w_type', type=str, default='poly', help='how to calculate weight of adjacency matrix',
                    choices=['poly', 'exp'])
parser.add_argument('--gamma', type=float, default=1.0)

parser.add_argument('--lp_ma', type=float, default=0.0, help='label used for LP is based on MA or not')

parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')
parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')

parser.add_argument('--output', type=str, default='result/')
parser.add_argument('--exp_name', type=str, default='unim_en5_dot')
parser.add_argument('--data_trans', type=str, default='moco')
args = parser.parse_args([])

if args.data_aug != 'null':
    args.data_aug = [float(v) for v in args.data_aug.split(',')]
else:
    args.data_aug = None
if args.loss_type == 'dot' or args.loss_type == 'dot_d':
    args.plabel_soft = True

if args.dset == 'visda-2017':
    names = ['train', 'validation']
    args.class_num = 12

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

args.s=0
args.t=1

folder = '../dataset/'
args.s_dset_path = folder + args.dset + '/' + names[args.s] + '/image_list.txt'
args.t_dset_path = folder + args.dset + '/' + names[args.t] + '/image_list.txt'
args.test_dset_path = folder + args.dset + '/' + names[args.t] + '/image_list.txt'

args.output_dir_src = osp.join(args.output, args.dset, 'source', names[args.s][0].upper())
args.output_dir = osp.join(args.output, args.dset, args.exp_name, names[args.s][0].upper() + names[args.t][0].upper())
args.name = names[args.s][0].upper() + names[args.t][0].upper()

if not osp.exists(args.output_dir):
    os.makedirs(args.output_dir)
set_log_path(args.output_dir)
log('save log to path {}'.format(args.output_dir_src))
log(print_args(args))

# ================== config ==================
dset_loaders = data_load(args, ss_load='moco')

# set base network
netF = network.ResBase(res_name=args.net)
netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,bottleneck_dim=args.bottleneck)
netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck)
modelpath = args.output_dir_src + '/source_F.pt'
netF.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
modelpath = args.output_dir_src + '/source_B.pt'
netB.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
modelpath = args.output_dir_src + '/source_C.pt'
netC.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))

# ========== performance of original model ==========

pred_labels, feats, labels, pred_probs = extract_feature_labels(dset_loaders["test"], netF, netB, netC, args, log)

mean_acc, classwise_acc, acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
log("Source model accuracy on target domain: {:.2f}%".format(mean_acc * 100) + '\nClasswise accuracy: {}'.format(
    classwise_acc))

FT_MAX_ACC, FT_MAX_MEAN_ACC = acc, mean_acc
LP_MAX_ACC, LP_MAX_MEAN_ACC = acc, mean_acc

if args.bn_adapt >= 0:  # -1 No BN adapt
    log("Adapt Batch Norm parameters")
    if args.bn_adapt == 0:
        netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)
    else:
        mom = 0.1 if args.bn_adapt == 1 else None
        netF, netB = bn_adapt1(netF, netB, dset_loaders["target"], mom=mom)

# ========== performance of original model ==========

a = torch.tensor([[1.2, 1.3], [1.5, 1.6]])
torch.save(a, 'a.pt')

a = np.array([1, 2.0])
b = np.array([3.5, 3.6])
np.savez('123.npz', a=a, b=b)

data = np.load('123.npz')
data['a']

np.save('a.pt', a)
with open('a.pt', 'rb') as f:
    new_a = np.load(f)



