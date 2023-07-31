import argparse
import os, sys, copy
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import model.network as network
from torch.utils.data import DataLoader
import random, pdb, math, copy
from tqdm import tqdm
import pickle
from office.utils import *
from torch import autograd
import shutil
from dataset.office_data import office_load
from utils import cal_acc, ensure_path, set_log_path, log


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_source(args, log):
    dset_loaders = office_load(args)

    # === set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = [{'params': netF.parameters(), 'lr': args.lr},
                   {'params': netB.parameters(), 'lr': args.lr * 10},
                   {'params': netC.parameters(), 'lr': args.lr * 10}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    acc_init = 0
    for epoch in range(args.max_epoch):
        netF.train()
        netB.train()
        netC.train()
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source, labels_source) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

            output = netF(inputs_source)
            output = netB(output)
            output = netC(output)

            loss = CrossEntropyLabelSmooth(
                num_classes=args.class_num, epsilon=args.smooth
            )(output, labels_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        netF.eval()
        netB.eval()
        netC.eval()
        acc_t, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
        acc_s, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
        log_str = "Task: {}, Iter:{}/{}; Src Acc={:.2f}%, Tar Acc={:.2f}%".format(
            args.dset, epoch + 1, args.max_epoch, acc_s * 100, acc_t * 100
        )
        log(log_str)

        if acc_t >= acc_init:
            acc_init = acc_t
            best_netF = copy.deepcopy(netF.state_dict())
            best_netB = copy.deepcopy(netB.state_dict())
            best_netC = copy.deepcopy(netC.state_dict())
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


def test_target(args, log):
    dset_loaders = office_load(args)

    # === set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(torch.load(args.output_dir + '/source_F.pt'))
    netB.load_state_dict(torch.load(args.output_dir + '/source_B.pt'))
    netC.load_state_dict(torch.load(args.output_dir + '/source_C.pt'))

    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders["test"], netF, netB, netC)
    log("Task: {}, Accuracy = {:.2f}%".format(args.dset, acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain Adaptation on office dataset")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--s", type=int, default=0, help="source")
    parser.add_argument("--t", type=int, default=1, help="target")
    parser.add_argument("--max_epoch", type=int, default=20, help="maximum epoch")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size")
    parser.add_argument("--worker", type=int, default=4, help="number of workers")
    parser.add_argument("--dset", type=str, default="a2c")

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=2021, help="random seed")
    parser.add_argument("--class_num", type=int, default=31)
    parser.add_argument("--par", type=float, default=0.1)

    parser.add_argument('--net', type=str, default='resnet50', help="resnet50, resnet101")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--layer", type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument("--classifier", type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument("--smooth", type=float, default=0.1)

    parser.add_argument("--output", type=str, default="office31_weight")
    parser.add_argument("--office31", action="store_true", default=True)

    args = parser.parse_args()
    # args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    args.output_dir = osp.join('result/office31/source', 'seed' + str(args.seed), args.dset)
    ensure_path(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    train_source(args, log)

    if args.office31:
        task = ["a", "d", "w"]
    task_s = args.dset.split("2")[0]
    task.remove(task_s)
    task_all = [task_s + "2" + i for i in task]
    for task_sameS in task_all:
        args.dset = task_sameS
        test_target(args, log)
