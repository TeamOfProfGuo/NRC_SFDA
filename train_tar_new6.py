# encoding:utf-8
import pdb
import pickle
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
from dataset.visda_data import data_load, image_train, moco_transform, mm_transform, mn_transform, mr_transform
from model.model_util import bn_adapt, bn_adapt1, label_propagation, extract_feature_labels, keep_top_n, get_affinity, local_cluster
from model.loss import compute_loss
from dataset.data_transform import TransformSW
from utils import cal_acc, print_args, log, set_log_path
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def reset_data_load(dset_loaders, pred_prob, args, moco_load=False):
    """
    modify the target data loader to return both image and pseudo label
    """
    txt_tar = open(args.t_dset_path).readlines()
    if args.data_trans == 'moco':
        data_trans = moco_transform(min_scales=args.data_aug)
    elif args.data_trans == 'sw':
        data_trans = TransformSW(mean, std)
    elif args.data_trans == 'mm':
        data_trans = mm_transform(min_scales=args.data_aug)
    elif args.data_trans == 'mn':
        data_trans = mn_transform(min_scales=args.data_aug)
    elif args.data_trans == 'mr':
        data_trans = mr_transform()
    else:
        data_trans = image_train()
    dsets = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True, pprob=pred_prob, ret_plabel=True, args=args)
    dloader = DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    if moco_load:
        dset_loaders['target_moco'] = dloader
    else:
        dset_loaders['target'] = dloader

    # label_inique, label_cnt = np.unique(dsets.plabel, return_counts=True)
    # log('Pseudo label count: ' +
    #     ', '.join([ '{} : {}'.format(k, v) for k, v in zip(label_inique, label_cnt) ]) )

def analysis_target(args):
    dset_loaders = data_load(args, ss_load='moco')

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

    # ========== performance of original model ==========
    mean_acc, classwise_acc, acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain: {:.2f}%".format(mean_acc*100) + '\nClasswise accuracy: {}'.format(classwise_acc))

    FT_MAX_ACC, FT_MAX_MEAN_ACC = acc, mean_acc
    LP_MAX_ACC, LP_MAX_MEAN_ACC = acc, mean_acc

    if args.bn_adapt >= 0:  # -1 No BN adapt
        log("Adapt Batch Norm parameters")
        if args.bn_adapt == 0: 
            netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)
        else: 
            mom = 0.1 if args.bn_adapt==1 else None 
            netF, netB = bn_adapt1(netF, netB, dset_loaders["target"], mom=mom)
            

    # ========== Define Model with Contrastive Branch ============
    model = moco.UniModel(netF, netB, netC)
    model = model.cuda()

    param_group = [{'params': model.netF.parameters(), 'lr': args.lr * 0.5},
                   {'params': model.netB.parameters(), 'lr': args.lr * 1},
                   {'params': model.netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.25)

    # ======================= start training =======================
    for epoch in range(1, args.max_epoch+1):
        log('==> Start epoch {}'.format(epoch))

        # ============ feature extraction ============
        model.eval()
        pred_labels, feats, labels, pred_probs_raw = extract_feature_labels(dset_loaders["test"], model.netF, model.netB, model.netC, args, log, epoch)

        if epoch==1 and (args.fuse_af>=0 or args.feat_type=='o'):
            feats_ori = copy.deepcopy(feats)
            W_ori = get_affinity(feats_ori, args)

        
        Z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()       # intermediate values
        z = torch.zeros(len(dset_loaders['target'].dataset), args.class_num).float().numpy()       # temporal outputs
        if (args.lp_ma > 0.0) and (args.lp_ma<1.0):  # if lp_ma=0 or lp_ma=1, then no moving avg
            Z = args.lp_ma * Z + (1. - args.lp_ma) * pred_probs_raw
            z = Z * (1. / (1. - args.lp_ma ** epoch))
            pred_probs_raw = z

        # ============ label propagation ============
        if args.fuse_af < 0:
            W0 = None
        elif args.fuse_af == 0:
            W0 = W_ori if epoch >= 2 else None
        elif args.fuse_af >= 1:
            if epoch == 0:
                W0 = None
            elif (epoch >= 1) and (epoch <= args.fuse_af + 1):
                W0 = W_ori
            else: 
                if (epoch - args.fuse_af) % 2 == 0: 
                    fname = osp.join(args.output_dir, 'w{}.pickle'.format((epoch - args.fuse_af) + 1))
                    with open(fname, 'rb') as f:
                        W0 = pickle.load(f)
                    log('load W0 from {}'.format(fname))
                else: 
                    log('use weight W0 from previous epoch')


        pred_labels, pred_probs, mean_acc, acc, W_new = label_propagation(pred_probs_raw, feats, labels, args, log,
                                                                          alpha=0.99, max_iter=20, ret_acc=True, W0=W0,
                                                                          ret_W=True)
        if args.debug:  # for ablation analysis
            W_new_k = keep_top_n(W_new, args.kk)
            _ = local_cluster(pred_probs_raw, W_new_k, labels, log)
           
            if epoch <= 20: 
                fname1 = osp.join(args.output_dir, 'pred_prob_ep{}.pickle'.format(epoch))
                with open(fname1, 'wb') as f: 
                    pickle.dump(pred_probs,f)
            
            if epoch == 1:
                fname2 = osp.join(args.output_dir, 'label_ep{}.pickle'.format(epoch))
                with open(fname2, 'wb') as f: 
                    pickle.dump(labels,f)
        
        
        if args.fuse_af>=0:
            if epoch%2 == 1:
                fname = osp.join(args.output_dir, 'w{}.pickle'.format(epoch))
                with open(fname, 'wb') as f:
                    pickle.dump(W_new, f)

        if mean_acc > LP_MAX_MEAN_ACC:
            LP_MAX_ACC = acc
            LP_MAX_MEAN_ACC = mean_acc

        # modify data loader: (1) add pseudo label to moco data loader
        reset_data_load(dset_loaders, pred_probs, args, moco_load=True)

        # ============ model finetuning ============
        mean_acc, acc = finetune_one_epoch(model, dset_loaders, optimizer, epoch)

        # how about LR
        scheduler.step()
        log('Current lr is netF: {:.6f}, netB: {:.6f}, netC: {:.6f}'.format(
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr']))

        if mean_acc > FT_MAX_MEAN_ACC:
            FT_MAX_MEAN_ACC = mean_acc
            FT_MAX_ACC = acc
            today = date.today()
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

    if args.loss_wt[1] == 'c':
        plabel_inique, plabel_cnt = np.unique(dset_loaders["target_moco"].dataset.plabel, return_counts=True)
        sorted_idx = np.argsort(plabel_inique)
        plabel_cnt = plabel_cnt.astype(np.float)
        cls_weight = 1/plabel_cnt[sorted_idx]
        cls_weight *= np.mean(plabel_cnt)
        log('cls_weight: ' + ','.join(['{:.2f}'.format(wt) for wt in cls_weight]))
        cls_weight = torch.tensor(cls_weight).cuda()

    for iter_num, batch_data in enumerate(dset_loaders["target_moco"]):
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
        elif args.loss_wt[0] == 'p':
            prob_dist = torch.abs(prob_tar1.detach() - prob_tar0.detach()).sum(dim=1) # [B]
            confidence_weight = 1 - torch.nn.functional.sigmoid(prob_dist)
            weight = confidence_weight
        elif args.loss_wt[0] == 'n':
            weight = None

        if args.loss_wt[1] == 'c':
            pass
        else:
            cls_weight = None

        ce0_wt, ce1_wt = float(args.loss_wt[2])/10, 1-float(args.loss_wt[2])/10

        ce_loss0 = compute_loss(plabel, prob_tar0, type=args.loss_type, weight=weight, cls_weight=cls_weight, soft_flag=args.plabel_soft)
        ce_loss1 = compute_loss(plabel, prob_tar1, type=args.loss_type, weight=weight, cls_weight=cls_weight, soft_flag=args.plabel_soft)
        ce_loss = 2.0 * ce0_wt * ce_loss0 + 2.0 * ce1_wt * ce_loss1

        if args.div_wt > 0.0:
            msoftmax0 = prob_tar0.mean(dim=0)
            msoftmax1 = prob_tar1.mean(dim=0)
            mentropy_loss = torch.sum(msoftmax0 * torch.log(msoftmax0 + 1e-8)) +\
                            torch.sum(msoftmax1 * torch.log(msoftmax1 + 1e-8))
            ce_loss += mentropy_loss * args.div_wt


        # if iter_num == 0 and epoch == 1:
        #     log('pred0 {}, pred1 {}'.format(prob_tar0[0].cpu().detach().numpy(), prob_tar1[0].cpu().detach().numpy()))
        #     log('{} weight {}'.format('entropy' if args.loss_wt[0]=='e' else 'confidence',
        #                               weight[0:5].cpu().numpy()))

        # if img_tar[0].size(0) == args.batch_size:
        #     output, target = model.moco_forward(im_q=img_tar[0], im_k=img_tar[1])  # query, key
        #     nce_loss = nn.CrossEntropyLoss()(output, target)
        #     nce_wt = args.nce_wt * (1+(epoch-1)/args.max_epoch) ** (-args.nce_wt_decay)
        #     loss = ce_loss + nce_wt * nce_loss
        # else:
        loss = ce_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    if args.dset == 'visda-2017':
        mean_acc, classwise_acc, acc, cm = cal_acc(dset_loaders['test'], model.netF, model.netB, model.netC, flag=True, ret_cm=True)
        log('After fine-tuning, Acc: {:.2f}%, Mean Acc: {:.2f}%,'.format(acc*100, mean_acc*100) + '\n' + 'Classwise accuracy: ' + classwise_acc)

        if epoch == 1 or epoch ==5:
            log('confusion matrix')
            for line in cm:
                log(' '.join(str(e) for e in line))

    return mean_acc, acc



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
    parser.add_argument('--bn_adapt', type=int, default=0, help='Whether to first finetune mu and std in BN layers')
    

    parser.add_argument('--lp_type', type=float, default=0, help="Label propagation use hard label or soft label, 0:hard label, >0: temperature")
    parser.add_argument('--T_decay', type=float, default=0.0, help='Temperature decay of creating pseudo-label in feature extraction')
    parser.add_argument('--feat_type', type=str, default='cls', choices=['cls', 'teacher', 'student', 't', 's', 'o'])
    parser.add_argument('--nce_wt', type=float, default=1.0, help='weight for nce loss')
    parser.add_argument('--nce_wt_decay', type=float, default=0.0, help='0.0:no decay, larger value faster decay')

    parser.add_argument('--loss_type', type=str, default='dot', help='Loss function', choices=['ce', 'sce', 'dot', 'dot_d'])
    parser.add_argument('--loss_wt', type=str, default='en5', help='CE/SCE loss weight: e|p|n, c|n (classwise weight), 0-9')
    parser.add_argument('--plabel_soft', action='store_false', help='Whether to use soft/hard pseudo label')
    parser.add_argument("--beta", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument('--data_aug', type=str, default='0.2,0.5', help='delimited list input')
    parser.add_argument('--w_type', type=str, default='poly', help='how to calculate weight of adjacency matrix', choices=['poly','exp'])
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--div_wt', type=float, default=0.0, help='weight for divergence')

    parser.add_argument('--lp_ma', type=float, default=0.0, help='label used for LP is based on MA or not')

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')
    parser.add_argument('--k', type=int, default=5, help='number of neighbors for label propagation')
    parser.add_argument('--kk', type=int, default=5, help='number of neighbors for label propagation')
    parser.add_argument('--fuse_af', type=int, default=0, help='fuse affinity')
    parser.add_argument('--fuse_type', type=str, default='c', help='how to fuse affinity')  # c|m

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='unim_en5_dot')
    parser.add_argument('--data_trans', type=str, default='moco')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    if args.data_aug != 'null':
        args.data_aug = [float(v) for v in args.data_aug.split(',')]
    else:
        args.data_aug = None
    if args.loss_type == 'dot' or args.loss_type == 'dot_d':
        args.plabel_soft = True
    if (args.fuse_af >= 0) and (args.k <= args.kk):
        args.k = args.kk*3

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
