# encoding:utf-8
import pdb
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import network
from data_list import ImageList
from train_tar_util import obtain_ncc_label, bn_adapt, label_propagation, extract_features
from loss import compute_dist, compute_loss
from dataset.data_transform import TransformSW
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args, log, set_log_path, pad_string


def trim_str(s, l):
    ret = s[:l] if len(s) >= l else s
    return ret


visda_classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse' 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)

    data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()
    dsets["target"] = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(txt_test, transform=image_test(), root=os.path.dirname(args.test_dset_path), ret_idx=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)

    return dset_loaders


def reset_data_load(dset_loaders, pred_prob, args):
    """
    modify the target data loader to return both image and pseudo label
    """
    txt_tar = open(args.t_dset_path).readlines()
    data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()
    dsets_target = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True, pprob=pred_prob, ret_plabel=True, args=args)
    dset_loaders["target"] = DataLoader(dsets_target, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)


def analysis_target(args):
    dset_loaders = data_load(args)

    # set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # performance of original model
    acc_tar, cls_acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain {} \n classwise accuracy {} \n".format(acc_tar, cls_acc))

    MAX_TEXT_ACC = acc_tar

    for epoch in range(1, args.max_epoch+1):
        if args.ncc:
            pred_labels, feats, labels, pred_probs = obtain_ncc_label(dset_loaders["test"], netF, netB, netC, args, log)
        else:
            pred_labels, feats, labels, pred_probs = extract_features(dset_loaders["test"], netF, netB, netC, args, log)

        pred_labels, pred_probs = label_propagation(pred_labels, feats, labels, args, log, alpha=0.99, max_iter=20)
        reset_data_load(dset_loaders, pred_probs, args)

        acc_tar = finetune_model(netF, netB, netC, dset_loaders)

        if acc_tar > MAX_TEXT_ACC:
            MAX_TEXT_ACC = acc_tar
            today = date.today()
            torch.save(netF.state_dict(),
                       osp.join(args.output_dir, "target_F_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netB.state_dict(),
                       osp.join(args.output_dir, "target_B_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netC.state_dict(),
                       osp.join(args.output_dir, "target_C_" + today.strftime("%Y%m%d") + ".pt"))


def finetune_model(netF, netB, netC, dset_loaders):

    param_group = [{'params': netF.parameters(), 'lr': args.lr * 0.1},
                   {'params': netB.parameters(), 'lr': args.lr * 1}]
    param_group_c = [{'params': netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # ======================== start training / adaptation
    netF.train()
    netB.train()
    netC.train()

    for iter_num, batch_data in enumerate(dset_loaders["target"]):
        input_tar, _, tar_idx, plabel, weight = batch_data

        if input_tar.size(0) == 1:
            continue

        input_tar = input_tar.cuda()
        plabel = plabel.cuda()
        weight = weight.cuda()

        feat_tar = netB(netF(input_tar))
        logit_tar = netC(feat_tar)
        prob_tar = nn.Softmax(dim=1)(logit_tar)

        if args.loss_wt:
            loss = compute_loss(plabel, prob_tar, type=args.loss_type, weight=weight)
        else:
            loss = compute_loss(plabel, prob_tar, type=args.loss_type)

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        # how about LR

        netF.eval()
        netB.eval()
        netC.eval()
        if args.dset == 'visda-2017':
            acc_tar, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, flag=True)
            log('Task: {}; Accuracy on target = {:.2f}%'.format(args.name, acc_tar) + '\n' + 'T: ' + acc_list)

        return acc_tar



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--T', type=float, default=0.5, help='Temperature for creating pseudo-label')
    parser.add_argument('--loss_type', type=str, default='sce', help='Loss function for target domain adaptation')
    parser.add_argument('--loss_wt', action='store_false', help='Whether to use weighted CE/SCE loss')
    parser.add_argument('--use_ncc', action='store_false', help='Whether to apply NCC in the feature extraction process')


    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')

    parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='Clust_LB')
    parser.add_argument('--data_trans', type=str, default='W')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

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

        analysis_target(args)
