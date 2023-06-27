# encoding:utf-8
import pdb
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random

from model import network, moco, dino
from model.loss import compute_loss
from dataset.visda_data import data_load, reset_data_load
from model.model_util import bn_adapt, label_propagation, extract_feature_labels
from utils import cal_acc, print_args, log, set_log_path, get_params_groups, cosine_scheduler, clip_gradients


def analysis_target(args):
    dset_loaders = data_load(args, ss_load='dino')

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

    # ========= Define Model with Dino Branch ===========
    model = dino.DiNo(netF, netB, netC, args, out_dim=args.out_dim, mlp_layers=2, use_bn=True, norm_last_layer=True)
    model = model.cuda()

    # ============ preparing optimizer ... ============
    params_groups = get_params_groups(model.netF, base_lr=args.lr * 0.5, weight_decay=1e-3) + \
                    get_params_groups(model.netB, base_lr=args.lr * 1.0, weight_decay=1e-3) + \
                    get_params_groups(model.netC, base_lr=args.lr * 1.0, weight_decay=1e-3) + \
                    get_params_groups(model.student[1], base_lr=args.lr * 1.0, weight_decay=1e-3) + \
                    get_params_groups(model.student[2], base_lr=args.lr * 1.0, weight_decay=1e-3)

    optimizer = torch.optim.SGD(params_groups, momentum=0.9, lr=args.lr)
    lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)
    # lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch * len(dset_loaders['target_ss']))

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = cosine_scheduler(args.momentum_teacher, 1, args.max_epoch, len(dset_loaders['target_ss']))
    log(f"Loss, optimizer and schedulers ready.")

    # =============== DiNo Loss ... ================
    dino_loss = dino.DINOLoss(
        args.out_dim,
        args.local_crops_num + 1,  # total number of crops = 2 global crops + local_crops_number
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=0,
        nepochs=args.max_epoch,
    ).cuda()

    # ======================= start training =======================
    for epoch in range(0, args.max_epoch):
        log('==> Start epoch {}'.format(epoch))

        pred_labels, feats, labels, pred_probs = extract_feature_labels(dset_loaders["test"], model.netF, model.netB, model.netC, args, log, epoch)

        pred_labels, pred_probs = label_propagation(pred_probs, feats, labels, args, log, alpha=0.99, max_iter=20)

        # modify data loader: (1) add pseudo label to moco data loader
        reset_data_load(dset_loaders, pred_probs, args, ss_load='dino')

        #pdb.set_trace()
        acc_tar = finetune_one_epoch(model, dset_loaders, optimizer, lr_schedule, momentum_schedule, dino_loss, epoch=epoch)

        log('Current lr is netF: {:.5f}, netB: {:.5f}, netC: {:.5f}, dino: {:.5f}'.format(
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'], optimizer.param_groups[3]['lr']))

        if acc_tar > MAX_TEXT_ACC:
            MAX_TEXT_ACC = acc_tar
            today = date.today()
            torch.save(netF.state_dict(),
                       osp.join(args.output_dir, "target_F_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netB.state_dict(),
                       osp.join(args.output_dir, "target_B_" + today.strftime("%Y%m%d") + ".pt"))
            torch.save(netC.state_dict(),
                       osp.join(args.output_dir, "target_C_" + today.strftime("%Y%m%d") + ".pt"))


def finetune_one_epoch(model, dset_loaders, optimizer, lr_schedule, momentum_schedule, dino_loss, epoch):

    # ======================== start training / adaptation
    model.train()

    ss_loss_epoch, ce_loss_epoch = 0.0, 0.0
    for iter_num, batch_data in enumerate(dset_loaders["target_ss"]):
        img_tar, _, tar_idx, plabel, weight = batch_data

        if img_tar[0].size(0) == 1:
            continue

        img_tar = [im.cuda(non_blocking=True) for im in img_tar]
        plabel = plabel.cuda()
        weight = weight.cuda()

        # teacher and student forward passes + compute dino loss
        teacher_output = model.dino_forward(img_tar[0], is_teacher=True)  # only the 1 global views pass through the teacher
        student_output = model.dino_forward(img_tar, is_teacher=False)
        ss_loss = dino_loss(student_output, teacher_output, epoch=epoch)

        logit_tar = model(img_tar[0])
        prob_tar = nn.Softmax(dim=1)(logit_tar)
        if args.loss_wt:
            ce_loss = compute_loss(plabel, prob_tar, type=args.loss_type, weight=weight)
        else:
            ce_loss = compute_loss(plabel, prob_tar, type=args.loss_type)
            
        ss_loss_epoch += ss_loss.item()
        ce_loss_epoch += ce_loss.item()

        loss = ce_loss + args.nce_wt * ss_loss
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad:
            param_norms = clip_gradients(model.student, args.clip_grad)
        optimizer.step()

        # update the teacher network
        it = iter_num + len(dset_loaders['target_ss']) * epoch
        m = momentum_schedule[it]
        model.ema_update_teacher(m)

    lr_schedule.step()

    model.eval()
    ss_loss_epoch /= len(dset_loaders["target_ss"])
    ce_loss_epoch /= len(dset_loaders["target_ss"])
    if args.dset == 'visda-2017':
        mean_acc, classwise_acc, acc = cal_acc(dset_loaders['test'], model.netF, model.netB, model.netC, flag=True)
        log('After fine-tuning, Acc: {:.2f}%, Mean Acc: {:.2f}%, ce_loss: {:.4f}, ss_loss: {:.4f}'.format(
            acc*100, mean_acc*100, ce_loss_epoch, ss_loss_epoch) 
            + '\n' + 'Classwise accuracy: ' + classwise_acc)

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
    parser.add_argument('--bn_adapt', action='store_false', help='Whether to first finetune mu and std in BN layers')
    parser.add_argument('--lp_type', type=float, default=0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=0.8, help='Temperature decay for creating pseudo-label')

    parser.add_argument('--hidden_dim', type=int, default=1024, help='hidden dim for the projection head in dino')
    parser.add_argument('--bottleneck_dim', type=int, default=256, help='bottleneck dim for the projection head in dino')
    parser.add_argument('--out_dim', type=int, default=8192, help='output dim of dino')
    parser.add_argument('--nce_wt', type=float, default=0.1, help='weight for nce loss')
    parser.add_argument('--local_crops_num', type=int, default=2, help='num of small local views')
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA param for teacher update. 
            We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="Max param gradient norm if using gradient clipping. 0 for disabling.")

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')

    parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='dino_wt1')
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
