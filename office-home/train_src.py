import random
import argparse
import os.path as osp
import torch.optim as optim
from utils import *
from model import network
from model.loss import CrossEntropyLabelSmooth
from dataset.oh_data import office_load

def train_source(args, log):
    dset_loaders = office_load(args)

    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    param_group = [{'params': netF.parameters(), 'lr': args.lr},
                   {'params': netB.parameters(), 'lr': args.lr * 10},
                   {'params': netC.parameters(),'lr': args.lr * 10}]
    optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=5e-4, nesterov=True)

    acc_init = 0.0
    for epoch in range(args.max_epoch):
        netF.train()
        netB.train()
        netC.train()
        iter_source = iter(dset_loaders["source_tr"])
        for batch_idx, (inputs_source, labels_source) in enumerate(iter_source):
            if inputs_source.size(0) == 1:
                continue
            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

            output = netB(netF(inputs_source))
            output = netC(output)
            
            loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(output, labels_source)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        netF.eval()
        netC.eval()
        acc_s_tr, _ = cal_acc(dset_loaders['source_te'], netF, netB, netC)
        log('Source: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, epoch + 1, args.max_epoch, acc_s_tr * 100))

        if acc_s_tr >= acc_init:
            acc_init = acc_s_tr
            best_netF = netF.state_dict()
            best_netB = netB.state_dict()
            best_netC = netC.state_dict()
    torch.save(best_netF, osp.join(args.output_dir, "source_F.pt"))
    torch.save(best_netB, osp.join(args.output_dir, "source_B.pt"))
    torch.save(best_netC, osp.join(args.output_dir, "source_C.pt"))


def test_target(args, log):
    dset_loaders = office_load(args)
    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type='bn', feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netF.load_state_dict(torch.load(args.output_dir + '/source_F.pt'))
    netB.load_state_dict(torch.load(args.output_dir + '/source_B.pt'))
    netC.load_state_dict(torch.load(args.output_dir + '/source_C.pt'))

    netF.eval()
    netB.eval()
    netC.eval()

    acc, _ = cal_acc(dset_loaders['test'], netF, netB, netC)
    log('Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc * 100))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain Adaptation on office-home dataset')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=20, help="maximum epoch")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='c2a')

    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, resnet101")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--layer', type=str, default="wn", help='type-classification layer', choices=["linear", "wn"])
    parser.add_argument('--class_num', type=int, default=65)
    parser.add_argument('--par', type=float, default=0.1)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--home', action='store_false')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    task = ['c', 'a', 'p', 'r']
    task_s = args.dset[0]
    task.remove(task_s)
    task_all = [task_s + '2' + i for i in task]

    args.output_dir = osp.join('result/home/source', 'seed' + str(args.seed), task_s)
    ensure_path(args.output_dir)
    set_log_path(args.output_dir)
    log('save log to path {}'.format(args.output_dir))
    log(print_args(args))

    if not osp.exists(osp.join(args.output_dir + '/source_F.pt')):
        train_source(args, log)
        for task in task_all:
            args.dset = task
            test_target(args, log)
