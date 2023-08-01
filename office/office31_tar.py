import argparse
import os, sys

sys.path.append("./")

import os.path as osp
from datetime import date
import torch.optim as optim
from torchvision import transforms
from model import network, moco
from model.loss import compute_loss
from dataset.data_list import ImageList
from dataset.office_data import office_load, moco_transform, image_target, mn_transform, get_AutoAug, mw_transform, get_RandAug
from model.model_util import bn_adapt, label_propagation, extract_feature_labels, extract_features
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import pickle
from utils import *
from torch import autograd


def map_name(shot_name):
    map_dict = {'a': "amazon", "d": "dslr", "w": "webcam"}
    return map_dict[shot_name]


def reset_data_load(dset_loaders, pred_prob, args,):
    """
        modify the target data loader to return both image and pseudo label
        """
    tt = args.dset.split("2")[1]
    t = map_name(tt)
    tar_list = "./dataset/data_list/office/{}_list.txt".format(t)
    tar_list = open(tar_list).readlines()

    if args.data_trans == 'moco':    # moco|mn|mw|ai|ac|ra()
        data_trans = moco_transform(min_scales=args.data_aug)
    elif args.data_trans == 'mn':
        data_trans = mn_transform(min_scales=args.data_aug)
    elif args.data_trans[0] == 'a':
        data_trans = get_AutoAug(args)
    elif args.data_trans == 'ra':
        data_trans = get_RandAug(args)
    elif args.data_trans == 'mw':
        data_trans = mw_transform()
    else:
        data_trans = image_target()

    data_target = ImageList(tar_list, transform=data_trans, root='../dataset/', ret_idx=True, pprob=pred_prob, ret_plabel=True, args=args)
    dloader = DataLoader(data_target, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders['target'] = dloader


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
    mean_acc, classwise_acc, acc = cal_acc(dset_loaders["test"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain: {:.2f}%".format(mean_acc * 100) +
        '\nClasswise accuracy: {}'.format(classwise_acc))

    FT_MAX_ACC, FT_MAX_MEAN_ACC = acc, mean_acc
    LP_MAX_ACC, LP_MAX_MEAN_ACC = acc, mean_acc

    if args.bn_adapt:
        log("Adapt Batch Norm parameters")
        netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)

    # ========== Define Model with Contrastive Branch ============
    model = moco.UniModel(netF, netB, netC)
    model = model.cuda()

    param_group = [{'params': model.netF.parameters(), 'lr': args.lr * 0.5},
                   {'params': model.netB.parameters(), 'lr': args.lr * 1},
                   {'params': model.netC.parameters(), 'lr': args.lr * 1},]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    # ======================= start training =======================
    for epoch in range(1, args.max_epoch + 1):
        log('==> Start epoch {}'.format(epoch))

        # ====== extract features ======
        pred_labels, feats, labels, pred_probs = extract_feature_labels(dset_loaders["test"],
                                                                        model.netF, model.netB, model.netC,
                                                                        args, log, epoch)

        Z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()  # intermediate values
        z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()  # temporal outputs
        if (args.lp_ma > 0.0) and (args.lp_ma < 1.0):  # if lp_ma=0 or lp_ma=1, then no moving avg
            Z = args.lp_ma * Z + (1. - args.lp_ma) * pred_probs
            z = Z * (1. / (1. - args.lp_ma ** epoch))
            pred_probs = z

        # ====== label propagation ======
        pred_labels, pred_probs, mean_acc, acc = label_propagation(pred_probs, feats, labels, args, log, alpha=0.99,
                                                                   max_iter=20, ret_acc=True)
        if acc > LP_MAX_ACC:
            LP_MAX_ACC = acc
            LP_MAX_MEAN_ACC = mean_acc

        # modify data loader: (1) add pseudo label to moco data loader
        reset_data_load(dset_loaders, pred_probs, args,)

        mean_acc, acc = finetune_one_epoch(model, dset_loaders, optimizer, epoch)

        # how about LR
        # scheduler.step()
        log('Current lr is netF: {:.6f}, netB: {:.6f}, netC: {:.6f}'.format(
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr']))

        if acc > FT_MAX_ACC:
            FT_MAX_ACC = acc
            FT_MAX_MEAN_ACC = mean_acc

            # today = date.today()
            # torch.save(netF.state_dict(),
            #            osp.join(args.output_dir, "target_F_" + today.strftime("%Y%m%d") + ".pt"))
            # torch.save(netB.state_dict(),
            #            osp.join(args.output_dir, "target_B_" + today.strftime("%Y%m%d") + ".pt"))
            # torch.save(netC.state_dict(),
            #            osp.join(args.output_dir, "target_C_" + today.strftime("%Y%m%d") + ".pt"))
        log('------LP_MAX_ACC={:.2f}%, LP_MAX_MEAN_ACC={:.2f}%, FT_MAX_ACC={:.2f}%, FT_MAX_MEAN_ACC={:.2f}% '.format(
            LP_MAX_ACC * 100, LP_MAX_MEAN_ACC * 100, FT_MAX_ACC * 100, FT_MAX_MEAN_ACC * 100))


def finetune_one_epoch(model, dset_loaders, optimizer, epoch=None):
    # ======================== start training / adaptation
    model.train()

    if args.loss_wt[1] == 'c':  # classwise weight
        plabel_inique, plabel_cnt = np.unique(dset_loaders["target"].dataset.plabel, return_counts=True)
        sorted_idx = np.argsort(plabel_inique)
        plabel_cnt = plabel_cnt.astype(np.float)
        cls_weight = 1 / plabel_cnt[sorted_idx]
        cls_weight *= np.mean(plabel_cnt)
        log('cls_weight: ' + ','.join(['{:.2f}'.format(wt) for wt in cls_weight]))
        cls_weight = torch.tensor(cls_weight).cuda()

    for iter_num, batch_data in enumerate(dset_loaders["target"]):
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
        elif args.loss_wt[1] == 'n':
            weight = None

        if args.loss_wt[1] == 'c':
            pass
        else:
            cls_weight = None

        ce0_wt, ce1_wt = float(args.loss_wt[2]) / 10, 1 - float(args.loss_wt[2]) / 10

        ce_loss0 = compute_loss(plabel, prob_tar0, type=args.loss_type, weight=weight, cls_weight=cls_weight, soft_flag=args.plabel_soft)
        ce_loss1 = compute_loss(plabel, prob_tar1, type=args.loss_type, weight=weight, cls_weight=cls_weight, soft_flag=args.plabel_soft)
        ce_loss = 2.0 * ce0_wt * ce_loss0 + 2.0 * ce1_wt * ce_loss1

        # model._momentum_update_teacher()

        # if iter_num == 0 and epoch == 1:
        #     log('pred0 {}, pred1 {}'.format(prob_tar0[0].cpu().detach().numpy(), prob_tar1[0].cpu().detach().numpy()))
        #     log('{} weight {}'.format('entropy' if args.loss_wt[0]=='e' else 'confidence',
        #                               weight[0:5].cpu().numpy()))

        optimizer.zero_grad()
        ce_loss.backward()
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

    return mean_acc, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation on office-31 dataset")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=50, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=0, help="number of workers")
    parser.add_argument("--dset", type=str, default="a2d")
    parser.add_argument("--office31", action="store_false", default=True)
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")

    parser.add_argument('--net', type=str, default='resnet50', help="resnet50, resnet101")
    parser.add_argument("--class_num", type=int, default=31)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--bn_adapt', action='store_false', help='Whether to first finetune mu and std in BN layers')

    parser.add_argument('--loss_type', type=str, default='dot', help='Loss function for target domain adaptation', choices=['ce', 'sce', 'dot', 'dot_d'])
    parser.add_argument('--loss_wt', type=str, default='en5', help='CE/SCE loss weight: e|p|n, c|n, 0-9')
    parser.add_argument('--plabel_soft', action='store_false', help='Whether to use soft/hard pseudo label')  #
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument('--data_aug', type=str, default='null', help='delimited list input')  # 0.2,0.5
    parser.add_argument('--data_trans', type=str, default='moco')

    parser.add_argument('--lp_ma', type=float, default=0.0, help='label used for LP is based on MA or not')
    parser.add_argument('--lp_type', type=float, default=1.0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=1.0, help='Temperature decay for creating pseudo-label')
    parser.add_argument('--w_type', type=str, default='poly', help='how to calculate weight of adjacency matrix', choices=['poly','exp'])

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')
    parser.add_argument('--k', type=int, default=3, help='number of neighbors for label propagation')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='moco_nce5_pn5_k3')
    args = parser.parse_args()

    if args.data_aug != 'null':
        args.data_aug = [float(v) for v in args.data_aug.split(',')]
    else:
        args.data_aug = None
    if args.loss_type == 'dot' or args.loss_type == 'dot_d':
        args.plabel_soft = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    args.output_dir_src = osp.join('result/office31/source/seed2021/', args.dset[0])
    args.output_dir = osp.join(args.output, 'office31', args.exp_name, args.dset)

    if os.path.exists(args.output_dir):                             # if output_dir already exists, reset it
        print('remove the existing folder {}'.format(args.output_dir))
        shutil.rmtree(args.output_dir)
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    train_target(args)
