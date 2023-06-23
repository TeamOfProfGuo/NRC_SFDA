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
import torch.nn.functional as F
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
        self.netF = network.ResBase(res_name=self.args.net).to(self.device)
        self.netB = network.feat_bootleneck(type=self.args.classifier, feature_dim=self.netF.in_features, bottleneck_dim=self.args.bottleneck).to(self.device)
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
            {'params': self.model.netF.parameters(), 'lr': self.args.lr, 'lr_scale': 1},
            {'params': self.model.projection_layer.parameters(), 'lr': self.args.lr, 'lr_scale': 1},
            {'params': self.model.netB.parameters(), 'lr': self.args.lr, 'lr_scale': 1},
            {'params': self.model.netC.parameters(), 'lr': self.args.lr, 'lr_scale': 1}
        ]

        self.optimizer = optim.SGD(param_group, momentum=0.9, weight_decay=1e-3, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.25)


    def finetune_one_epoch(self, epoch, get_acc=True):
        self.model.train()
        start_time = time.time()
        batch_time = time.time()

        for iter_num, batch_data in enumerate(self.dataloaders["target_moco"]):
            if epoch < self.args.warmup_epochs:
                adjust_learning_rate(self.optimizer, (iter_num+1)/len(self.dataloaders["target_moco"]) + epoch, self.args)

            img_tar, _, tar_idx, plabel, _ = batch_data  # weight: [b]

            if img_tar[0].size(0) == 1:
                continue

            img_tar[0] = img_tar[0].cuda()  # img_tar: list | img_tar[0]: query image for moco | img_tar[1]: key image for moco 
            img_tar[1] = img_tar[1].cuda()
            plabel = plabel.cuda()
            # weight = weight.cuda()

            logit_tar0 = self.model(img_tar[0])
            prob_tar0 = nn.Softmax(dim=1)(logit_tar0)
            logit_tar1 = self.model(img_tar[1])
            prob_tar1 = nn.Softmax(dim=1)(logit_tar1)  # [B, K]

            logit_tar0, logit_tar1 = None, None

            # prob_dist = torch.abs(prob_tar1.detach() - prob_tar0.detach()).sum(dim=1) # [B]
            prob_dist = F.cosine_similarity(prob_tar1.detach(), prob_tar0.detach()) # [B]   change to cosine similarity
            # pdb.set_trace()
            confidence_weight = 1 - torch.nn.functional.sigmoid(prob_dist)

            if args.loss_wt:
                ce_loss = compute_loss(plabel, prob_tar0, type=self.args.loss_type, reduction='none', weight=confidence_weight, cls_weight=False)
            else:
                ce_loss = compute_loss(plabel, prob_tar0, type=self.args.loss_type)

            if img_tar[0].size(0) == self.args.batch_size:
                output, target, feat = self.model.moco_forward(im_q=img_tar[0], im_k=img_tar[1])

                nce_loss = nn.CrossEntropyLoss()(output, target)
                loss = ce_loss + self.args.nce_wt * nce_loss

                output, target, feat = None, None, None
            else:
                loss = ce_loss

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
            return mean_acc


    def train(self):
        # do not do these when testing
        if not self.args.test:
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

            if epoch > 3:
                # select the confident logits
                if self.args.thres_change == 'inc':
                    threshold = self.get_threshold_inc(epoch, self.args.max_epochs)
                else:
                    threshold = self.get_threshold_dec(epoch, self.args.max_epochs)
                logits = torch.gather(torch.from_numpy(pred_probs), 1, torch.from_numpy(pred_labels).unsqueeze(1)).squeeze()
                idx = torch.where(logits > threshold)
            else:
                idx = None
            
            # modify data loader: (1) add pseudo label to moco data loader
            self.reset_data_load(pred_probs, select_idx=idx, moco_load=True)

            # save memory usage
            idx, feats, logits, labels = None, None, None, None
            pred_probs, pred_labels = None, None

            # acc_tar = self.finetune_one_epoch(epoch, get_acc=True)
            self.finetune_one_epoch(epoch, get_acc=True)

            # step scheduler only works for the init adapt period
            if epoch >= self.args.warmup_epochs:
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


    # threshold increase
    def get_threshold_inc(self, epoch, total_epoch):
        threshold = self.args.log_min_thres + epoch * (self.args.log_max_thres - self.args.log_min_thres) / total_epoch
        return math.log(threshold)

    # threshold decrease
    def get_threshold_dec(self, epoch, total_epoch):
        threshold = self.args.log_min_thres + epoch * (self.args.log_max_thres - self.args.log_min_thres) / total_epoch
        return -math.log(threshold)



def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='config file path')
    parser.add_argument('--exp_name', required=True, type=str, help='experiment name')
    parser.add_argument('--exp_id', required=True, type=str, help='config modifications')
    args = parser.parse_args()
    cfg = load_cfg_from_cfg_file(args.config)
    # update exp_name and exp_id
    cfg['exp_name'] = args.exp_name
    cfg['exp_id'] = args.exp_id
    return cfg


if __name__ == "__main__":
    args = parse_config()

    # base on the current config, do some modifications
    # test mode only compute first several batches so that debugging will be much faster
    if args.exp_name == 'test':
        args.test = True
    else:
        args.test = False    

    seed = args.seed
    if seed is not None:
        cudnn.benchmark = True
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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

    args.update()

    # after the update, we do some modifications base on the configs
    if args.thres_change == 'inc':  # threshold increase
        # base on the threshold, do some modification
        args.log_min_thres = math.exp(args.min_thres)
        args.log_max_thres = math.exp(args.max_thres)
    else:                            # threshold decrease
        # base on the threshold, do some modification
        args.log_min_thres = math.exp(-args.max_thres)
        args.log_max_thres = math.exp(-args.min_thres)

    trainer = Trainer(args)
    trainer.train()
