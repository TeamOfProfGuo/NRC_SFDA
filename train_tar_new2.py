# encoding:utf-8
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import network, moco
from dataset.data_list import ImageList
from dataset.visda_data import data_load, image_train, moco_transform
from model.model_util import obtain_ncc_label, bn_adapt, label_propagation, extract_features
from model.loss import compute_loss
from dataset.data_transform import TransformSW
from utils import cal_acc, print_args, log, set_log_path
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def reset_data_load(dset_loaders, pred_prob, args, moco_load=False):
    """
    modify the target data loader to return both image and pseudo label
    """
    txt_tar = open(args.t_dset_path).readlines()
    if moco_load:
        data_trans = moco_transform
    else:
        data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()
    dsets = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True, pprob=pred_prob, ret_plabel=True, args=args)
    dloader = DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    if moco_load:
        dset_loaders['target_moco'] = dloader
    else:
        dset_loaders['target'] = dloader


def analysis_target(args):
    dset_loaders = data_load(args, moco_load=True)

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
    mean_acc, classwise_acc, acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain: {:.2f}%".format(mean_acc*100) + '\nClasswise accuracy: {}'.format(classwise_acc))

    MAX_TEXT_ACC = mean_acc
    if args.bn_adapt:
        log("Adapt Batch Norm parameters")
        netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)

    # ========== Define Model with Contrastive Branch ============
    model = moco.MoCo(netF, netB, netC, dim=128, K=4096, m=0.999, T=0.07, mlp=True)
    model = model.cuda()

    param_group = [{'params': model.netF.parameters(), 'lr': args.lr * 0.5},
                   {'params': model.projection_layer.parameters(), 'lr': args.lr * 1},
                   {'params': model.netB.parameters(), 'lr': args.lr * 1},
                   {'params': model.netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    # ======================= start training =======================
    for epoch in range(1, args.max_epoch+1):
        log('==> Start epoch {}'.format(epoch))
        if args.use_ncc:
            pred_labels, feats, labels, pred_probs = obtain_ncc_label(dset_loaders["test"], model.netF, model.netB, model.netC, args, log)
        else:
            pred_labels, feats, labels, pred_probs = extract_features(dset_loaders["test"], model.netF, model.netB, model.netC, args, log, epoch)

        pred_labels, pred_probs = label_propagation(pred_probs, feats, labels, args, log, alpha=0.99, max_iter=20)

        # modify data loader: (1) add pseudo label to moco data loader
        reset_data_load(dset_loaders, pred_probs, args, moco_load=True)

        acc_tar = finetune_one_epoch(model, dset_loaders, optimizer)


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


def finetune_one_epoch(model, dset_loaders, optimizer):

    # ======================== start training / adaptation
    model.train()

    for iter_num, batch_data in enumerate(dset_loaders["target_moco"]):
        img_tar, _, tar_idx, plabel, weight = batch_data

        if img_tar[0].size(0) == 1:
            continue

        img_tar[0] = img_tar[0].cuda()
        img_tar[1] = img_tar[1].cuda()
        plabel = plabel.cuda()
        weight = weight.cuda()

        logit_tar = model(img_tar[0])
        prob_tar = nn.Softmax(dim=1)(logit_tar)
        if args.loss_wt:
            ce_loss = compute_loss(plabel, prob_tar, type=args.loss_type, weight=weight)
        else:
            ce_loss = compute_loss(plabel, prob_tar, type=args.loss_type)

        if img_tar[0].size(0) == args.batch_size:
            output, target = model.moco_forward(im_q=img_tar[0], im_k=img_tar[1])
            nce_loss = nn.CrossEntropyLoss()(output, target)
            loss = ce_loss + args.nce_wt * nce_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    if args.dset == 'visda-2017':
        mean_acc, classwise_acc, acc = cal_acc(dset_loaders['test'], model.netF, model.netB, model.netC, flag=True)
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
    parser.add_argument('--loss_type', type=str, default='sce', help='Loss function for target domain adaptation')
    parser.add_argument('--loss_wt', action='store_false', help='Whether to use weighted CE/SCE loss')
    parser.add_argument('--use_ncc', action='store_true', help='Whether to apply NCC in the feature extraction process')
    parser.add_argument('--bn_adapt', action='store_false', help='Whether to first finetune mu and std in BN layers')
    parser.add_argument('--lp_type', type=float, default=0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=0.8, help='Temperature decay for creating pseudo-label')
    parser.add_argument('--nce_wt', type=float, default=1.0, help='weight for nce loss')

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')

    parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='moco')
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
