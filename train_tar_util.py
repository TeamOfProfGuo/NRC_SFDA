
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


def obtain_ncc_label(loader, netF, netB, netC, args, log):
    """ prediction from Nearest Centroid Classifier """
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            input, label = data[0], data[1]
            feat = netB(netF(input.cuda()))
            output = netC(feat)
            if start_test:
                all_feat = feat.float().cpu()
                all_output = output.float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_feat = torch.cat((all_feat, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, output.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    acc = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_feat = torch.cat((all_feat, torch.ones(all_feat.size(0), 1)), 1)
        all_feat = (all_feat.t() / torch.norm(all_feat, p=2, dim=1)).t()

    all_feat = all_feat.float().cpu().numpy()  # [B, 257]
    aff = all_output.float().cpu().numpy()     # [B, 12]
    K = all_output.size(1)

    for run in range(2):
        initc = aff.transpose().dot(all_feat)  # [12, B] [B, 257]
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        cls_count = np.eye(K)[predict].sum(axis=0)
        labelset = np.where(cls_count>args.threshold)[0]

        dd = cdist(all_feat, initc[labelset], args.distance)
        pred_label = dd.argmin(axis=1)
        new_pred = labelset[pred_label]
        pred_score = torch.zeros_like(all_output).float()
        pred_score[:, labelset] = nn.Softmax(dim=1)(torch.tensor(dd)).float()

        aff = np.eye(K)[new_pred]            # need to experiment  aff = pred_score

        new_acc = np.sum(new_pred == all_label.float().numpy()) / len(all_feat)
        log('Nearest Centroid Classifier Accuracy after {} runs = {:.2f}% -> {:.2f}%'.format(run+1, acc * 100, new_acc * 100))
    
    matrix = confusion_matrix(all_label, new_pred.float())
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    new_acc = acc.mean()
    cls_acc = [str(np.round(i, 2)) for i in acc]
    cls_acc = ' '.join(cls_acc)
    log('overall acc {} classwise accuracy {}'.format(new_acc, cls_acc))

    return new_pred.astype('int'), all_feat, all_label, pred_score


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


def update_plabels(pred_label, feat, args, log, alpha=0.99, max_iter=20):

    log('======= Updating pseudo-labels =======')
    pred_label = np.asarray(pred_label)

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
    Z = np.zeros((N, len(args.class_num)))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(len(args.class_num)):
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
    return probs_l1
