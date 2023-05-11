
import pdb
import numpy as np
import scipy
import scipy.stats
import faiss
from faiss import normalize_L2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist


def extract_features(loader, netF, netB, netC, args, log, isMT = False):
    netF.eval()
    netB.eval()
    netC.eval()

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
    log("Accuracy while extracting features {:.2f}".format(acc*100))

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
        log('Nearest Centroid Classifier Accuracy after {} runs = {:.2f}% -> {:.2f}%'.format(run+1, acc * 100, new_acc * 100))
        # aff = np.eye(K)[new_preds]  aff=new_probs
    
    matrix = confusion_matrix(all_labels, new_preds)
    cls_acc = matrix.diagonal()/(matrix.sum(axis=1) + 1e-10)* 100
    cls_acc = [str(np.round(i, 2)) for i in cls_acc]
    cls_acc = ' '.join(cls_acc)
    log('overall average acc {} classwise accuracy {}'.format(new_acc, cls_acc))

    return new_preds, all_feats, all_labels, new_probs


def bn_adapt(netF, netB, data_loader, runs=10):
    netF.eval()
    netB.eval()
    n_batch = 0
    mom_pre = 0.1
    
    iter_test = iter(data_loader)
    for _ in range(len(data_loader)):
        data = iter_test.next()
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


def label_propagation(pred_label, feat, label, args, log, alpha=0.99, max_iter=20):
    """
    Args:
        pred_label: current predicted label
        feat: feature embedding for all samples (used for computing similarity)
        label: GT label

        alpha:
        max_iter:

    Returns:

    """
    log('======= Updating pseudo-labels =======')

    # kNN search for the graph
    N, d = feat.shape[0], feat.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatIP(res, d, flat_config)  # build the index

    normalize_L2(feat)
    index.add(feat)
    log('n total {}'.format(index.ntotal))
    D, I = index.search(feat, args.k + 1)

    # Create the graph
    D = D[:, 1:] ** 3  # [N, k]
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (args.k, 1)).T
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
        cur_idx = np.where(pred_label==i)[0]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), p=1, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0

    new_pred = np.argmax(probs_l1, 1)
    new_acc = float(np.sum(new_pred == label)) / len(label)
    log('accuracy after label propagation with k={} is {:.4f}'.format(args.k, new_acc))
    return new_pred, probs_l1