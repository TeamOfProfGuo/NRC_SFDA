
import pdb
import numpy as np

import faiss
from faiss import normalize_L2

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy
import scipy.stats
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from utils.logger import Logger as Log


def compute_acc(labels, preds):
    """
    Args:
        labels:  n-d array
        preds:  n-d array

    Returns:

    """
    matrix = confusion_matrix(labels, preds)
    classwise_acc = matrix.diagonal() / matrix.sum(axis=1)
    acc = classwise_acc.mean()
    classwise_acc = [str(np.round(i, 2)) for i in classwise_acc]
    classwise_acc = ' '.join(classwise_acc)
    return acc, classwise_acc


class PPMModule(nn.Module):
    """
    Reference: 
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super(PPMModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = nn.Sequential(
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        out = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        out = self.bottleneck(torch.cat(out, 1))
        return out   # bottle


def extract_features(loader, netF, netB, netC, args, epoch=1, isMT = False):    
    netF.eval()
    netB.eval()
    netC.eval()
    temperature = args.labelprop_type if args.labelprop_type>0 else 1
    if args.labelprop_type > 0:
        temperature *= args.T_decay ** (epoch-1)
    Log.info('While extracting features T(temperature): {:.5f}'.format(temperature))

    all_feats, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            # if i % (len(loader) // 4) == 0:
            #     Log.info(f"     Extract feat | Current iter {i} | Total {len(loader)} iters")
            inputs, labels = batch_data[0], batch_data[1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            feats = netB(netF(inputs))
            logits = netC(feats)
            logits = logits/temperature
            probs = nn.Softmax(dim=1)(logits)

            all_feats.append(feats.float().cpu())  # [192, 256]
            all_probs.append(probs.float().cpu())  # [192, 12]
            all_labels.append(labels)

            # if i == 4:
            #     break

    all_feats = torch.cat(all_feats, dim=0)
    if args.distance == 'cosine':
        all_feats = torch.cat((all_feats, torch.ones(all_feats.shape[0], 1)), dim=1)
        all_feats = all_feats / torch.norm(all_feats, p=2, dim=1, keepdim=True)
    # can add more distance here

    all_feats = all_feats.numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    acc = float(np.sum(all_preds == all_labels)) / float(all_labels.shape[0])
    mean_acc, _ = compute_acc(all_labels, all_preds)
    Log.info("While extracting features Acc: {:.2f}% Mean Acc: {:.2f}%".format(acc*100, mean_acc*100))

    return all_preds, all_feats, all_labels, all_probs


def obtain_ncc_label(loader, netF, netB, netC, args, log):
    """ prediction from Nearest Centroid Classifier """

    all_feats, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            inputs, labels = batch_data[0], batch_data[1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            feats = netB(netF(inputs))
            logits = netC(feats)
            probs = nn.Softmax(dim=1)(logits)

            all_feats.append(feats.float().cpu())
            all_probs.append(probs.float().cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats, dim=0)
    if args.distance == 'cosine':
        all_feats = torch.cat((all_feats, torch.ones(all_feats.shape[0], 1)), dim=1)
        all_feats = all_feats / torch.norm(all_feats, p=2, dim=1, keepdim=True)

    all_feats = all_feats.numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    acc = float(np.sum(all_preds == all_labels)) / float(all_labels.shape[0])
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)

    aff = all_probs
    K = all_probs.shape[1]
    for run in range(1):
        initc = aff.transpose().dot(all_feats)  # [12, B] [B, 257]
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(K)[all_preds].sum(axis=0)
        labelset = np.where(cls_count >= args.threshold)[0]

        feat_centroid_dist = cdist(all_feats, initc[labelset], args.distance)
        new_preds = np.argmin(feat_centroid_dist, axis=1)
        new_preds = labelset[new_preds]
        new_probs = torch.zeros(all_probs.shape).float()
        new_probs[:, labelset] = nn.Softmax(dim=1)(torch.tensor(feat_centroid_dist)).float()
        new_probs = new_probs.cpu().numpy()

        new_acc = float(np.sum(new_preds == all_labels)) / len(all_labels)
        Log.info('Nearest Centroid Classifier Accuracy after {} runs = {:.2f}% -> {:.2f}%'.format(run+1, acc * 100, new_acc * 100))
        # aff = np.eye(K)[new_preds]  aff=new_probs
    
    mean_acc, classwise_acc = compute_acc(all_labels, new_preds)
    Log.info('After NCC, Acc: {:.2f}%, Mean Acc: {:.2f}%, Classwise accuracy {}'.format(new_acc*100, mean_acc*100, classwise_acc))

    return new_preds, all_feats, all_labels, new_probs


def bn_adapt(netF, netB, data_loader, runs=10):
    netF.eval()
    netB.eval()
    n_batch = 0
    mom_pre = 0.1
    
    iter_test = iter(data_loader)
    for _ in range(len(data_loader)):
        data = next(iter_test)
        inputs, label = data[0], data[1]
    
        mom_new = (mom_pre * 0.95)
        mom_pre = mom_new
        n_batch += 1
        if n_batch >= runs:
            break

        for m in netF.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + 0.05
        for m in netB.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.train()
                m.momentum = mom_new + 0.05

        _ = netB(netF(inputs.cuda()))
    return netF, netB


def label_propagation(pred_prob, feat, label, args, alpha=0.99, max_iter=20):
    """
    Args:
        pred_label: current predicted label
        feat: feature embedding for all samples (used for computing similarity)
        label: GT label

        alpha:
        max_iter:
    """
    pred_label = pred_prob if args.labelprop_type > 0 else np.argmax(pred_prob, axis=1)

    # kNN search for the graph
    N, d = feat.shape[0], feat.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index  index = faiss.IndexFlatL2(d)

    normalize_L2(feat)
    index.add(feat)
    # log('n total {}'.format(index.ntotal))
    D, I = index.search(feat, args.k_neighbors + 1)

    # Create the graph
    D = D[:, 1:] ** 3  # [N, k]
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (args.k_neighbors, 1)).T
    W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, args.class_num))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(args.class_num):
        if args.labelprop_type == 0:
            y = np.zeros((N,))
            cur_idx = np.where(pred_label==i)[0]   # pred_label [N]
            y[cur_idx] = 1.0 / cur_idx.shape[0]
        else:
            y = pred_label[:, i] / np.sum(pred_label[:, i])
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), p=1, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0

    new_pred = np.argmax(probs_l1, 1)
    new_acc = float(np.sum(new_pred == label)) / len(label)
    mean_acc, _ = compute_acc(label, new_pred)
    Log.info('After label propagation Acc :{:.2f}%, Mean Acc: {:.2f}%'.format(new_acc*100, mean_acc*100))
    return new_pred, probs_l1


class SubBatchNorm2d(nn.Module):
    """
    The standard BN layer computes stats across all examples in a GPU. In some
    cases it is desirable to compute stats across only a subset of examples
    (e.g., in multigrid training https://arxiv.org/abs/1912.00998).
    SubBatchNorm2d splits the batch dimension into N splits, and run BN on
    each of them separately (so that the stats are computed on each subset of
    examples (1/N of batch) independently. During evaluation, it aggregates
    the stats from all splits into one BN.
    """

    def __init__(self, num_splits, **args):
        """
        Args:
            num_splits (int): number of splits.
            args (list): other arguments.
        """
        super(SubBatchNorm2d, self).__init__()
        self.num_splits = num_splits
        num_features = args["num_features"]
        # Keep only one set of weight and bias.
        if args.get("affine", True):
            self.affine = True
            args["affine"] = False
            self.weight = torch.nn.Parameter(torch.ones(num_features))
            self.bias = torch.nn.Parameter(torch.zeros(num_features))
        else:
            self.affine = False
        self.bn = nn.BatchNorm2d(**args)
        args["num_features"] = num_features * num_splits
        self.split_bn = nn.BatchNorm2d(**args)

    def _get_aggregated_mean_std(self, means, stds, n):
        """
        Calculate the aggregated mean and stds.
        Args:
            means (tensor): mean values.
            stds (tensor): standard deviations.
            n (int): number of sets of means and stds.
        """
        mean = means.view(n, -1).sum(0) / n
        std = (
            stds.view(n, -1).sum(0) / n
            + ((means.view(n, -1) - mean) ** 2).view(n, -1).sum(0) / n
        )
        return mean.detach(), std.detach()

    def aggregate_stats(self):
        """
        Synchronize running_mean, and running_var. Call this before eval.
        """
        if self.split_bn.track_running_stats:
            (
                self.bn.running_mean.data,
                self.bn.running_var.data,
            ) = self._get_aggregated_mean_std(
                self.split_bn.running_mean,
                self.split_bn.running_var,
                self.num_splits,
            )

    def forward(self, x):
        if self.training:
            n, c, h, w = x.shape
            x = x.view(n // self.num_splits, c * self.num_splits, h, w)
            x = self.split_bn(x)
            x = x.view(n, c, h, w)
        else:
            x = self.bn(x)
        if self.affine:
            x = x * self.weight.view((-1, 1, 1, 1))
            x = x + self.bias.view((-1, 1, 1, 1))
        return x
