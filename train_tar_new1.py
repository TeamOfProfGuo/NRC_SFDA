# encoding:utf-8
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import network
from dataset.data_list import ImageList
from dataset.visda_data import data_load, image_train
from model.model_util import obtain_ncc_label, bn_adapt, label_propagation, extract_feature_labels
from model.loss import compute_loss
from dataset.data_transform import TransformSW
from utils import cal_acc, print_args, log, set_log_path
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


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

    param_group = [{'params': netF.parameters(), 'lr': args.lr * 0.1},
                   {'params': netB.parameters(), 'lr': args.lr * 1},
                   {'params': netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)


    # performance of original model
    mean_acc, classwise_acc, acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain: {:.2f}%".format(mean_acc*100) + '\nClasswise accuracy: {}'.format(classwise_acc))

    MAX_TEXT_ACC = mean_acc
    if args.bn_adapt: 
        log("Adapt Batch Norm parameters")
        netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)

    for epoch in range(1, args.max_epoch+1):
        log('==> Start epoch {}'.format(epoch))
        pred_labels, feats, labels, pred_probs = extract_feature_labels(dset_loaders["test"], netF, netB, netC, args, log, epoch)

        pred_labels, pred_probs = label_propagation(pred_probs, feats, labels, args, log, alpha=0.99, max_iter=20)
        reset_data_load(dset_loaders, pred_probs, args)

        acc_tar = finetune_one_epoch(netF, netB, netC, dset_loaders, optimizer, epoch=epoch)


        # how about LR
        scheduler.step()
        log('Current lr is netF: {:.6f}, netB: {:.6f}, netC: {:.6f}'.format(
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr']))

        if acc_tar > MAX_TEXT_ACC:
            MAX_TEXT_ACC = acc_tar
            today = date.today()
            torch.save(netF.state_dict(),
                       osp.join(args.output_dir, "target_F_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netB.state_dict(),
                       osp.join(args.output_dir, "target_B_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netC.state_dict(),
                       osp.join(args.output_dir, "target_C_" + today.strftime("%Y%m%d") + ".pt"))


def finetune_one_epoch(netF, netB, netC, dset_loaders, optimizer, epoch):

    # ======================== start training / adaptation
    netF.train()
    netB.train()
    netC.train()
    max_iter = args.max_epoch * len(dset_loaders["target"])

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
            loss = compute_loss(plabel, prob_tar, type=args.loss_type, weight=weight, soft_flag=args.plabel_soft)
        else:
            loss = compute_loss(plabel, prob_tar, type=args.loss_type, soft_flag=args.plabel_soft)

        if args.loss_type == 'dot_d':
            mask = torch.ones((prob_tar.shape[0], prob_tar.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag  # square matrix with only diagonal matrix = 0

            copy = prob_tar.T  # .detach().clone()# [c, batch]
            dot_neg = prob_tar @ copy  # batch x batch
            dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
            neg_pred = torch.mean(dot_neg)

            curr_iter = iter_num + len(dset_loaders["target"]) * (epoch-1)
            alpha = (1 + 10 * curr_iter / max_iter) ** (-args.beta) * 1.0
            loss += neg_pred * alpha

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    netF.eval()
    netB.eval()
    netC.eval()
    if args.dset == 'visda-2017':
        mean_acc, classwise_acc, acc = cal_acc(dset_loaders['test'], netF, netB, netC, flag=True)
        log('After fine-tuning, Acc: {:.2f}%, Mean Acc: {:.2f}%,'.format(acc*100, mean_acc*100) + '\n' + 'Classwise accuracy: ' + classwise_acc)

    return mean_acc



if __name__ == "__main__":
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
    parser.add_argument('--loss_type', type=str, default='sce', help='Loss function', choices=['ce', 'sce', 'dot', 'dot_d'])
    parser.add_argument('--loss_wt', action='store_false', help='Whether to use weighted CE/SCE loss')
    parser.add_argument('--plabel_soft', action='store_true', help='Whether to use soft/hard pseudo label')
    parser.add_argument("--beta", type=float, default=5.0)

    parser.add_argument('--bn_adapt', action='store_false', help='Whether to first finetune mu and std in BN layers')
    parser.add_argument('--lp_type', type=float, default=0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=0.8, help='Temperature decay for creating pseudo-label')

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'cosine1' 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')

    parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='LP_cosine')
    parser.add_argument('--data_trans', type=str, default='W')
    args = parser.parse_args()

    if args.loss_type == 'dot' or args.loss_type == 'dot_d':
        args.plabel_soft = True

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
