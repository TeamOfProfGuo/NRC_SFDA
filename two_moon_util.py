import torch
import scipy
import faiss
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
# ========== use source model for fea_bank & score_bank ==========


def train_target(model, tar_loader, max_iter=200, K=5, beta=5.0, alpha=0.0):

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_sample = len(tar_loader.dataset)
    fea_bank = torch.randn(num_sample, model.hidden_dim)
    score_bank = torch.randn(num_sample, model.output_dim)

    model.eval()

    # ========== use source model for fea_bank & score_bank ==========

    with torch.no_grad():
        iter_test = iter(tar_loader)
        for i in range(len(tar_loader)):
            data = next(iter_test)
            inputs, indx = data[0], data[1]

            output = model.extract_feat(inputs)
            output_norm = F.normalize(output)
            outputs = model.cls(output)
            outputs = torch.nn.Softmax(-1)(outputs)

            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()

    # ========== start training ==========
    iter_num = 0
    model.train()

    real_max_iter = max_iter
    while iter_num < real_max_iter:
        try:
            inputs_test, tar_idx, _ = next(iter_test)
        except:
            iter_test = iter(tar_loader)
            inputs_test, tar_idx, _ = next(iter_test)

        if inputs_test.size(0) == 1:
            continue

            alpha = (1 + 10 * iter_num / max_iter) ** (-beta) * alpha

        iter_num += 1

        features_test = model.extract_feat(inputs_test)
        outputs_test = model.cls(features_test)
        softmax_out = torch.nn.Softmax(dim=1)(outputs_test)

        # ========== update fea & score bank ==========
        with torch.no_grad():
            output_f_norm = F.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = score_bank[idx_near]  # batch x K x C

        # nn
        softmax_out_un = softmax_out.unsqueeze(1).expand(-1, K, -1)  # batch x K x C

        # =================== loss ===================
        loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1))  # Equal to dot product

        mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag  # square matrix with only diagonal matrix = 0

        copy = softmax_out.T  # .detach().clone()# [c, batch]
        dot_neg = softmax_out @ copy  # batch x batch
        dot_neg = (dot_neg * mask).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)

        loss += neg_pred * alpha
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        acc = compute_acc(model, tar_loader)
        print('acc', acc)

        model.train()

    return model


def compute_acc(model, tar_loader):
    num_sample = len(tar_loader.dataset)
    model.eval()
    label_bank = torch.zeros(num_sample).long()
    pred_bank = torch.zeros(num_sample).long()
    iter_test = iter(tar_loader)
    with torch.no_grad():
        for i in range(len(tar_loader)):
            data = next(iter_test)
            inputs, idx, label = data[0], data[1], data[2]

            pred = model(inputs)
            label_bank[idx] = label
            pred_bank[idx] = pred.detach().argmax(1)
    acc = torch.sum(label_bank == pred_bank).float() / num_sample
    return acc



def extract_feat_label(model, test_loader):
    model.eval()
    all_feats, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for i, batch_data in enumerate(test_loader):
            inputs, labels = batch_data[0], batch_data[-1]
            if torch.cuda.is_available():
                inputs = inputs.cuda()

            feats = model.extract_feat(inputs)
            logits = model.cls(feats)
            probs = torch.nn.Softmax(dim=1)(logits)

            all_feats.append(feats.float().cpu())
            all_probs.append(probs.float().cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats, dim=0)
    all_feats = all_feats / torch.norm(all_feats, p=2, dim=1, keepdim=True)

    all_feats = all_feats.numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_preds = np.argmax(all_probs, axis=1)
    acc = float(np.sum(all_preds == all_labels)) / float(all_labels.shape[0])

    print("While extracting features Acc: {:.4f}".format(acc))
    return all_preds, all_feats, all_labels, all_probs


def label_propagation(pred_prob, feats, label, K=5, alpha=0.99, max_iter=20, class_num=2):
    pred_label = np.argmax(pred_prob, axis=1)

    # kNN search for the graph
    N, d = feats.shape[0], feats.shape[1]
    index = faiss.IndexFlatL2(d)  # build the index
    print(index.is_trained)
    index.add(feats)  # add vectors to the index
    print(index.ntotal)

    # log('n total {}'.format(index.ntotal))
    D, I = index.search(feats.copy(), K + 1)
    # Create the graph
    D = D[:, 1:] ** 3  # [N, k]

    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (K, 1)).T
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
    Z = np.zeros((N, class_num))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(class_num):
        y = np.zeros((N,))
        cur_idx = np.where(pred_label == i)[0]  # pred_label [N]
        y[cur_idx] = 1.0 / (cur_idx.shape[0] + 1e-10)
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)  # Use Conjugate Gradient iteration to solve Ax = b
        Z[:, i] = f

    # Handle numberical errors
    Z[Z < 0] = 0

    # Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), p=1, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0

    new_pred = np.argmax(probs_l1, 1)
    new_acc = float(np.sum(new_pred == label)) / len(label)
    print('After label propagation Acc: {:.4f}'.format(new_acc))

    return new_pred, probs_l1



def adapt_target(model, tar_x, tar_y,  K=5, batch_size=300, class_num=2, max_epoch=10):
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, max_epoch+1):
        tar_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tar_x).float(), torch.arange(tar_y.shape[0]),
                                                     torch.from_numpy(tar_y))
        test_loader = torch.utils.data.DataLoader(tar_dataset, batch_size=batch_size, shuffle=False)


        pred, feats, labels, pred_prob = extract_feat_label(model, test_loader)
        pred_labels, pred_probs = label_propagation(pred_prob, feats, labels, K=K, alpha=0.99, max_iter=20, class_num=2)


        entropy = scipy.stats.entropy(pred_probs, axis=1)
        weights = 1 - entropy / np.log(class_num)


        tar_dataset = torch.utils.data.TensorDataset(torch.from_numpy(tar_x).float(), torch.from_numpy(pred_probs), torch.from_numpy(weights),
                                                     torch.arange(tar_y.shape[0]), torch.from_numpy(tar_y))
        tar_loader = torch.utils.data.DataLoader(tar_dataset, batch_size=batch_size, shuffle=True)

        # finetune_one_epoch

        model.train()

        for iter_num, batch_data in enumerate(tar_loader):
            x_tar, plabel, weight, idx, y = batch_data

            logit_tar = model(x_tar)
            pred_tar = torch.nn.Softmax(dim=1)(logit_tar)

            ones = torch.eye(class_num)
            targets_2d = torch.index_select(ones, dim=0, index=plabel)

            loss = torch.sum(- targets_2d * pred_tar, 1)
            loss = loss.sum() / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        acc = compute_acc(model, test_loader)
        print('after fine-tuning, acc: {:.4f}'.format(acc))

