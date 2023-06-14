import os
import sys
import pdb
import math
import time
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models import network, moco
from loss.loss import compute_loss
from dataset.data_list import ImageList
from dataset.data_transform import TransformSW
from dataset.visda_data import data_load, image_train, moco_transform
from models.utils import obtain_ncc_label, bn_adapt, label_propagation, extract_features

from utils.logger import Logger as Log
from utils.util import load_cfg_from_cfg_file, merge_cfg_from_list, AverageMeter, ensure_path, Entropy, adjust_learning_rate


class MidFeatureNetF(nn.Module):
    def __init__(self, netF):
        super(MidFeatureNetF, self).__init__()
        self.init_block = nn.Sequential(
            netF.conv1,
            netF.bn1,
            netF.relu,
            netF.maxpool,
        )

        self.layer1 = netF.layer1
        self.layer2 = netF.layer2
        self.layer3 = netF.layer3
        self.layer4 = netF.layer4

        self.avg_pool = netF.avgpool

        self.in_features = netF.in_features
    
    def forward(self, x):
        feat_lst = []
        feats = self.init_block(x)
        feats = self.layer1(feats)
        feats = self.layer2(feats)
        feat_lst.append(feats)
        feats = self.layer3(feats)
        feat_lst.append(feats)
        feats = self.layer4(feats)
        feat_lst.append(feats)

        return feat_lst


class Aggregator(nn.Module):
    def __init__(self, in_features=(512, 1024, 2048), out_feature=512, strides=(1, 2, 2)):
        super(Aggregator, self).__init__()

        self.stages = []
        assert len(in_features) == len(strides), "out_features length does not equal to strides size!"
        self.stages = nn.ModuleList([self._make_stage(in_features[i], out_feature, size=7, stride=strides[i]) for i in range(len(in_features))])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_feature*len(in_features), 2048, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_feature, size, stride):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # can use conv layer to adjust feature size
        conv = nn.Conv2d(features, out_feature, kernel_size=1, bias=False)
        bn = nn.Sequential(
            nn.BatchNorm2d(out_feature),
            nn.ReLU()
        )
        return nn.Sequential(prior, conv, bn)

    def forward(self, feat_lst):
        # feature list contains multi-stage features
        feats = [self.stages[i](feat_lst[i]) for i in range(len(feat_lst))]
        
        feats = self.bottleneck(torch.cat(feats, 1))

        return feats


class AggNetF(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, netF, in_features=(512, 1024, 2048), out_feature=512, strides=(1, 2, 2)):
        super(AggNetF, self).__init__()

        self.mid_feat_extract = MidFeatureNetF(netF)

        self.in_features = netF.in_features
        
        self.aggregator = Aggregator(in_features=in_features, out_feature=out_feature, strides=strides)

        self.avg_pool = netF.avgpool

    def forward(self, x):
        # get the mid-stage features
        feat_lst = self.mid_feat_extract(x)
        # aggreagate the mid-stage features
        feats = self.aggregator(feat_lst)

        # pass the bottleneck and average pooling to get the final feature
        feats = self.avg_pool(feats).squeeze()

        return feats


class Trainer(object):
    def __init__(self, args):
        self.args = args
        
        self.device = torch.device(args.device)
        # self.loss_scaler = NativeScaler()
        self.train_losses = AverageMeter()

        self._init_base_network()
        # set dataloader
        self._set_dataloader()

        self._set_contrastive()


    def _init_base_network(self):
        # build model
        # feature extractor
        self.netF = network.ResBase(res_name=self.args.net).to(self.device)
        # bottleneck
        self.netB = network.feat_bootleneck(type=self.args.classifier, feature_dim=self.netF.in_features, bottleneck_dim=self.args.bottleneck).to(self.device)
        # classifer
        self.netC = network.feat_classifier(type=self.args.layer, class_num=self.args.class_num, bottleneck_dim=self.args.bottleneck).to(self.device)

        # load net weight
        self.netF.load_state_dict(torch.load(self.args.weight_dir + '/source_F.pt'))
        self.netB.load_state_dict(torch.load(self.args.weight_dir + '/source_B.pt'))
        self.netC.load_state_dict(torch.load(self.args.weight_dir + '/source_C.pt'))


    def _set_dataloader(self):
        self.dataloaders = data_load(self.args, moco_load=True)


    def _set_contrastive(self, agg=False):
        # ========== Define Model with Contrastive Branch ============
        self.model = moco.MoCo(self.netF, self.netB, self.netC, dim=128, K=4096, m=0.999, T=0.07, mlp=True).to(self.device)

        param_group = [
            {'params': self.model.netF.parameters(), 'lr': self.args.lr, 'lr_scale': 0.5},
            {'params': self.model.projection_layer.parameters(), 'lr': self.args.lr, 'lr_scale': 1},
            {'params': self.model.netB.parameters(), 'lr': self.args.lr, 'lr_scale': 1},
            {'params': self.model.netC.parameters(), 'lr': self.args.lr, 'lr_scale': 1}
        ]

        self.optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.25)


    def finetune_one_epoch(self, epoch, get_acc=True):
        # self.model.train()
        start_time = time.time()
        batch_time = time.time()

        for iter_num, batch_data in enumerate(self.dataloaders["target_moco"]):
            # scheuler
            adjust_learning_rate(self.optimizer, iter_num / len(self.dataloaders["target_moco"]) + epoch, self.args)

            img_tar, _, tar_idx, plabel, weight = batch_data

            if img_tar[0].size(0) == 1:
                continue

            img_tar[0] = img_tar[0].cuda()
            img_tar[1] = img_tar[1].cuda()
            plabel = plabel.cuda()
            weight = weight.cuda()

            logit_tar = self.model(img_tar[0])
            prob_tar = nn.Softmax(dim=1)(logit_tar)
            if args.loss_wt:
                ce_loss = compute_loss(plabel, prob_tar, type=self.args.loss_type, reduction='none', weight=weight, cls_weight=False)
            else:
                ce_loss = compute_loss(plabel, prob_tar, type=self.args.loss_type)

            if img_tar[0].size(0) == self.args.batch_size:
                output, target, feat = self.model.moco_forward(im_q=img_tar[0], im_k=img_tar[1])

                nce_loss = nn.CrossEntropyLoss()(output, target)
                loss = ce_loss + self.args.nce_wt * nce_loss
            else:
                loss = ce_loss

            # store the features
            if iter_num == 0:
                all_feats = feat
            else:
                all_feats = torch.cat((all_feats, feat), dim=0)

            # update the record of the train phase
            self.train_losses.update(loss.item(), self.args.batch_size)  # running avg of loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # print the log info & reset the states.
            if iter_num % (len(self.dataloaders["target_moco"]) // 4) == 0:
                Log.info('Ep {0} Iter {1} | Total {total} iters | loss {loss.val:.4f} (avg {loss.avg:.4f}) | '
                        'lr {3} | time {batch_time:.2f}s/{2}iters'.format(
                    epoch, iter_num, 200,
                    f"{self.optimizer.param_groups[0]['lr']:.7f}", 
                    batch_time=(time.time()-batch_time), loss=self.train_losses, total=len(self.dataloaders["target_moco"])))
                batch_time = time.time()

                self.train_losses.reset()

        # validation
        if get_acc:
            self.model.eval()
            if self.args.dataset == 'visda-2017':
                mean_acc, classwise_acc, acc = self.cal_acc(self.dataloaders['test'], self.model.netF, self.model.netB, self.model.netC, flag=True)
                Log.info('After fine-tuning | Acc: {:.2f}% | Mean Acc: {:.2f}% | '.format(acc*100, mean_acc*100) + '\n' + 'Classwise accuracy: ' + classwise_acc)

        # Log info of the time
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        msg = f"Epoch {epoch} done | Training time {total_time_str}"
        msg += '\n\n'
        Log.info(msg)

        if get_acc:
            return mean_acc, all_feats


    def train(self):
        # performance of original model
        Log.info("Checking the performance of the original model")
        mean_acc, classwise_acc, acc = self.cal_acc(self.dataloaders["target"], self.netF, self.netB, self.netC, flag=True)
        Log.info("Source model accuracy on target domain: {:.2f}%".format(mean_acc*100) + '\nClasswise accuracy: {}\n'.format(classwise_acc))

        # adapt the batch normalization layer
        MAX_TEXT_ACC = mean_acc
        if self.args.bn_adapt:
            Log.info("Adapt Batch Norm parameters")
            self.netF, self.netB = bn_adapt(self.netF, self.netB, self.dataloaders["target"], runs=1000)

        Log.info("Start Training")
        # epochs to get decend pseudo label
        for epoch in range(self.args.max_epochs):
            Log.info('==> Init SeudoLabel | Start epoch {}'.format(epoch))
            if self.args.use_ncc:
                pred_labels, feats, labels, pred_probs = obtain_ncc_label(self.dataloaders["test"], self.model.netF, self.model.netB, self.model.netC, self.args)
            else:
                pred_labels, feats, labels, pred_probs = extract_features(self.dataloaders["test"], self.model.netF, self.model.netB, self.model.netC, self.args, epoch)

            # pred_probs: [55388, 12]   feats: [55388, 257]    labels: [55388]
            pred_labels, pred_probs = label_propagation(pred_probs, feats, labels, args, alpha=0.99, max_iter=20)

            # # select the confident logits
            # threshold = self.get_threshold(epoch, self.args.init_seudolabel_ep+self.args.add_aggreg_ep)
            # logits = torch.gather(torch.from_numpy(pred_probs), 1, torch.from_numpy(pred_labels).unsqueeze(1)).squeeze()
            # idx = torch.where(logits > threshold)

            # modify data loader: (1) add pseudo label to moco data loader
            self.reset_data_load(pred_probs, select_idx=None, moco_load=True)

            acc_tar, all_feats = self.finetune_one_epoch(epoch, get_acc=True)

            # step scheduler only works for the init adapt period
            self.scheduler.step()
            
            Log.info('Current lr is netF: {:.6f}, netB: {:.6f}, netC: {:.6f}'.format(
                self.optimizer.param_groups[0]['lr'], self.optimizer.param_groups[1]['lr'], self.optimizer.param_groups[2]['lr']))


    def reset_data_load(self, pred_prob, select_idx=None, moco_load=False):
        """
        modify the target data loader to return both image and pseudo label
        """
        txt_tar = open(self.args.target_data_path).readlines()
        if moco_load:
            data_trans = moco_transform
        else:
            data_trans = TransformSW(mean, std, aug_k=1) if self.args.data_trans == 'SW' else image_train()
        
        # slicing the target text list
        if select_idx is not None:
            new_target = []
            for i in select_idx[0]:
                new_target.append(txt_tar[i])

            dataset = ImageList(new_target, transform=data_trans, root=os.path.dirname(self.args.target_data_path), ret_idx=True, pprob=pred_prob[select_idx], ret_plabel=True, args=self.args)
        else:
            dataset = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(self.args.target_data_path), ret_idx=True, pprob=pred_prob, ret_plabel=True, args=self.args)

        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers, drop_last=False)
        if moco_load:
            self.dataloaders['target_moco'] = dataloader
        else:
            self.dataloaders['target'] = dataloader


    def cal_acc(self, loader, netF, netB, netC, flag=True):
        start_test = True
        netF.eval()
        netB.eval()
        netC.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                # if i % (len(loader) // 4) == 0:
                #     Log.info(f"     Calculating Acc | Current iter {i} | Total {len(loader)} iters")
                inputs = data[0].cuda()
                labels = data[1]
                outputs = netC(netB(netF(inputs)))
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

        if flag:
            matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
            acc = matrix.diagonal()/matrix.sum(axis=1)
            mean_acc = acc.mean()
            classwise_acc = [str(np.round(i*100, 2)) for i in acc]
            classwise_acc = ' '.join(classwise_acc)
            return mean_acc, classwise_acc, accuracy
        else:
            return accuracy, mean_ent


    def get_threshold(self, epoch, total_epoch):
        threshold = self.args.min_thres + (epoch+1) * (self.args.max_thres - self.args.min_thres) / total_epoch
        return math.log(threshold) / 5


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='config file path')
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--exp_id', required=True, type=str, help='config modifications')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)  # handle config file
    # update exp_name and exp_id
    cfg['exp_name'] = args.exp_name
    cfg['exp_id'] = args.exp_id
    return cfg


if __name__ == "__main__":
    args = parse_config()
    pdb.set_trace()
    seed = args.seed
    if seed is not None:
        cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # deal with thresholds
    args.max_thres = math.exp(args.max_logit_thres*5)
    args.min_thres = math.exp(args.min_logit_thres*5)

    if args.dataset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dataset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    # set data path
    args.source_data_path = args.root + args.dataset + '/' + args.source_data_name + '/image_list.txt'   # source path
    args.target_data_path = args.root + args.dataset + '/' + args.target_data_name + '/image_list.txt'   # target path
    args.test_data_path = args.root + args.dataset + '/' + args.target_data_name + '/image_list.txt'   # test path

    save_path = f"./results/{datetime.date.today()}:{args.exp_name}/{args.exp_id}"
    # save_path = f"./results/{args.exp_name}/{args.exp_id}"
    save_flag = ensure_path(save_path)
    args.save_path = save_path

    Log.init(
        log_file=os.path.join(save_path, 'output.log'),
        logfile_level='info',
        stdout_level='info',
        rewrite=True
    )
    Log.info('Save log to path {}'.format(args.save_path))

    # beautify the log output of the configuration
    msg = '\nConfig: \n'
    arg_lst = str(args).split('\n')
    for arg in arg_lst:
        msg += f'   {arg}\n'
    msg += f'\n[exp_name]: {args.exp_name}\n[exp_id]: {args.exp_id}\n[save_path]: {args.save_path}\n'
    Log.info(msg)
    
    # use the exp_id to update config
    args.update()

    trainer = Trainer(args)
    trainer.train()
