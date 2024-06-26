
import sys
sys.path.append('./')

import os.path as osp
from datetime import date
import torch.optim as optim
import random
import argparse
from utils import *
from model import moco, network
from model.loss import compute_loss
from torch.utils.data import DataLoader
from dataset.data_list import ImageList
from dataset.oh_data import office_load, moco_transform
from model.model_util import bn_adapt, label_propagation, extract_feature_labels, extract_features


def map_name(shot_name):
    map_dict = {'a': 'Art', 'c': 'Clipart', 'p': 'Product', 'r': 'Real_World'}
    return map_dict[shot_name]


def reset_data_load(dset_loaders, pred_prob, args, ss_load=None):
    """
    modify the target data loader to return both image and pseudo label
    """
    tt = args.dset.split('2')[1]
    t = map_name(tt)
    tar_list = 'dataset/data_list/office-home/{}.txt'.format(t)
    tar_list = open(tar_list).readlines()

    ss_target = ImageList(tar_list, transform=moco_transform, root='../dataset/', ret_idx=True,
                          pprob=pred_prob, ret_plabel=True, args=args)
    dloader = DataLoader(ss_target, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)

    if ss_load == 'moco':
        dset_loaders['target_ss'] = dloader
    else:
        dset_loaders['target'] = dloader

    label_inique, label_cnt = np.unique(ss_target.plabel, return_counts=True)
    log('Pseudo label count: ' +
        ', '.join(['{} : {}'.format(k, v) for k, v in zip(label_inique, label_cnt)]))


def train_target(args):
    dset_loaders = office_load(args, ret_idx=True)

    # ==== set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    # ========== performance of original model ==========
    mean_acc, classwise_acc, acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain: {:.2f}%".format(mean_acc * 100) +
        '\nClasswise accuracy: {}'.format(classwise_acc))

    MAX_TEXT_ACC = mean_acc
    if args.bn_adapt:
        log("Adapt Batch Norm parameters")
        netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)

    # ========== Define Model with Contrastive Branch ============
    model = moco.MoCo(netF, netB, netC, dim=128, K=4096, m=0.999, T=0.07, mlp=True)
    model = model.cuda()

    param_group = [{'params': model.netF.parameters(), 'lr': args.lr * 0.5},
                   {'params': model.netB.parameters(), 'lr': args.lr * 1},
                   {'params': model.netC.parameters(), 'lr': args.lr * 1},
                   {'params': model.projection_layer.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    # ======================= start training =======================
    for epoch in range(1, args.max_epoch + 1):
        log('==> Start epoch {}'.format(epoch))

        # ====== extract features ======
        pred_labels, feats, labels, pred_probs = extract_feature_labels(dset_loaders["test"],
                                                                        model.netF, model.netB, model.netC,
                                                                        args, log, epoch)
        if args.feat_type == 'cls':
            pass
        elif args.feat_type == 'student':
            feats = extract_features(dset_loaders["test"], model.encoder_q, args)
        elif args.feat_type == 'teacher':
            feats = extract_features(dset_loaders["test"], model.encoder_k, args)

        Z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()  # intermediate values
        z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()  # temporal outputs
        if (args.lp_ma > 0.0) and (args.lp_ma < 1.0):  # if lp_ma=0 or lp_ma=1, then no moving avg
            Z = args.lp_ma * Z + (1. - args.lp_ma) * pred_probs
            z = Z * (1. / (1. - args.lp_ma ** epoch))
            pred_probs = z

        # ====== label propagation ======
        pred_labels, pred_probs = label_propagation(pred_probs, feats, labels, args, log, alpha=0.99, max_iter=20)

        # modify data loader: (1) add pseudo label to moco data loader
        reset_data_load(dset_loaders, pred_probs, args, ss_load='moco')

        acc_tar = finetune_one_epoch(model, dset_loaders, optimizer, epoch)

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


def finetune_one_epoch(model, dset_loaders, optimizer, epoch=None):
    # ======================== start training / adaptation
    model.train()

    if args.loss_wt[1] == 'c':  # classwise weight
        plabel_inique, plabel_cnt = np.unique(dset_loaders["target_ss"].dataset.plabel, return_counts=True)
        sorted_idx = np.argsort(plabel_inique)
        plabel_cnt = plabel_cnt.astype(np.float)
        cls_weight = 1 / plabel_cnt[sorted_idx]
        cls_weight *= np.mean(plabel_cnt)
        log('cls_weight: ' + ','.join(['{:.2f}'.format(wt) for wt in cls_weight]))
        cls_weight = torch.tensor(cls_weight).cuda()

    for iter_num, batch_data in enumerate(dset_loaders["target_ss"]):
        img_tar, _, tar_idx, plabel, weight = batch_data

        if img_tar[0].size(0) == 1:
            continue

        img_tar[0] = img_tar[0].cuda()
        img_tar[1] = img_tar[1].cuda()
        plabel = plabel.cuda()
        weight = weight.cuda()

        logit_tar0 = model(img_tar[0])
        prob_tar0 = nn.Softmax(dim=1)(logit_tar0)
        logit_tar1 = model(img_tar[1])
        prob_tar1 = nn.Softmax(dim=1)(logit_tar1)  # [B, K]

        if args.loss_wt[0] == 'e':  # entropy weight
            pass
        elif args.loss_wt[0] == 'p':  # confidence weight
            prob_dist = torch.abs(prob_tar1.detach() - prob_tar0.detach()).sum(dim=1)  # [B]
            confidence_weight = 1 - torch.nn.functional.sigmoid(prob_dist)
            weight = confidence_weight

        if args.loss_wt[1] == 'c':
            pass
        else:
            cls_weight = None

        ce0_wt, ce1_wt = float(args.loss_wt[2]) / 10, 1 - float(args.loss_wt[2]) / 10

        ce_loss0 = compute_loss(plabel, prob_tar0, type=args.loss_type, weight=weight, cls_weight=cls_weight)
        ce_loss1 = compute_loss(plabel, prob_tar1, type=args.loss_type, weight=weight, cls_weight=cls_weight)
        ce_loss = 2.0 * ce0_wt * ce_loss0 + 2.0 * ce1_wt * ce_loss1

        model._momentum_update_teacher()

        # if iter_num == 0 and epoch == 1:
        #     log('pred0 {}, pred1 {}'.format(prob_tar0[0].cpu().detach().numpy(), prob_tar1[0].cpu().detach().numpy()))
        #     log('{} weight {}'.format('entropy' if args.loss_wt[0]=='e' else 'confidence',
        #                               weight[0:5].cpu().numpy()))

        if img_tar[0].size(0) == args.batch_size:
            output, target = model.moco_forward(im_q=img_tar[0], im_k=img_tar[1])
            nce_loss = nn.CrossEntropyLoss()(output, target)
            nce_wt = args.nce_wt * (1 + (epoch - 1) / args.max_epoch) ** (-args.nce_wt_decay)
            loss = ce_loss + nce_wt * nce_loss
        else:
            loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    mean_acc, classwise_acc, acc, cm = cal_acc(dset_loaders['test'], model.netF, model.netB, model.netC,
                                                flag=True, ret_cm=True)
    log('After fine-tuning, Acc: {:.2f}%, Mean Acc: {:.2f}%,'.format(acc * 100, mean_acc * 100) +
        '\n' + 'Classwise accuracy: ' + classwise_acc)

    # if epoch == 1 or epoch == 5:
    #     log('confusion matrix')
    #     for line in cm:
    #         log(' '.join(str(e) for e in line))

    return mean_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation on office-home dataset')
    parser.add_argument('--home', action='store_false')
    parser.add_argument('--gpu_id',  type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=30, help="maximum epoch") # set to 50 on office-31
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--interval', type=int, default=2)
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='a2r')
    parser.add_argument('--choice', type=str, default='shot')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, resnet101")
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])

    parser.add_argument('--bn_adapt', action='store_false', help='Whether to first finetune mu and std in BN layers')
    parser.add_argument('--feat_type', type=str, default='cls', choices=['cls', 'teacher', 'student'])

    parser.add_argument('--loss_type', type=str, default='sce', help='Loss function for target domain adaptation')
    parser.add_argument('--loss_wt', type=str, default='pn5', help='CE/SCE loss weight: e|p|n, c|n, 0-9')
    parser.add_argument('--nce_wt', type=float, default=1.0, help='weight for nce loss')
    parser.add_argument('--nce_wt_decay', type=float, default=0.0, help='0.0:no decay, larger value faster decay')

    parser.add_argument('--lp_ma', type=float, default=0.0, help='label used for LP is based on MA or not')
    parser.add_argument('--lp_type', type=float, default=0.0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=0.8, help='Temperature decay for creating pseudo-label')

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')
    parser.add_argument('--k', type=int, default=3, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='moco_nce5_pn5_k3')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    current_folder = "./"
    args.output_dir = osp.join(current_folder, args.output,
                               'seed' + str(args.seed), args.dset)

    args.output_dir_src = osp.join('result/home/source/seed2021/', args.dset[0])
    args.output_dir = osp.join(args.output, 'home', args.exp_name, args.dset)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    train_target(args)
