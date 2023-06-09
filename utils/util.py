import os
import pdb
import yaml
import math
import time
import errno
import shutil
import pickle
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix

from pathlib import Path
from typing import Tuple
from collections import defaultdict, deque
from typing import Callable, Iterable, List, TypeVar

import torch
from torch import inf
import torch.nn as nn
import torch.distributed as dist

from utils.logger import Logger as Log

# ======================================================================================================================
# ======== All following helper functions have been borrowed from from https://github.com/Jia-Research-Lab/PFENet ======
# ======================================================================================================================

class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict=None, key_list=None, new_allowed=False):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        key_list = [] if key_list is None else key_list
        for k, v in init_dict.items():
            if type(v) is dict:
                # Convert dict to CfgNode
                init_dict[k] = CfgNode(v, key_list=key_list + [k])
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            s = s_.split("\n")
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * " ") + line for line in s]
            s = "\n".join(s)
            s = first + "\n" + s
            return s

        r = ""
        s = []
        for k, v in sorted(self.items()):
            seperator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), seperator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())

    def update(self):
        """
        exp_id: ft_se32_te768_hd12_mp4_l9_w3_eff11
        [ft, se32, te768, hd12, mp4, l9, w3]
        freeze_enc: True, shrink_embed: 32, trans_embed: 768, num_head: 12
        num_layer: 9, warmup_epochs : 3, eff_batch_adjust: 11
        """
        msg = "\nUpdate: \n"

        update_lst = self.exp_id.split('_')  # ncew3_lp3_tdc10_sdep8_agit14_warm7

        self.nce_wt = int(update_lst[0][4:]) * 0.1  # weight for nce loss
        msg += f"   nce_wt: {self.nce_wt}\n"

        self.labelprop_type = int(update_lst[1][2:]) * 0.1   # label propagation type
        msg += f"   labelprop_type: {self.labelprop_type}\n"

        self.T_decay = int(update_lst[2][3:]) * 0.01  # Temperature decay for creating pseudo-label  90 * 0.01 => 0.9
        msg += f"   T_decay: {self.T_decay}\n"

        # self.init_seudolabel_ep = int(update_lst[3][4:])    # epochs to initize the model to get decend pseudo label 
        # msg += f"   init_seudolabel_ep: {self.init_seudolabel_ep}\n"

        # self.agg_init_ep = int(update_lst[4][4:])    # epochs to initize the aggregator weights
        # msg += f"   init_seudolabel_ep: {self.agg_init_ep}\n"

        self.warmup_epochs = int(update_lst[3][4:])    # warmup epochs after add the aggreagator
        msg += f"   warmup_epochs: {self.warmup_epochs}\n"

        Log.info(msg)

def _decode_cfg_value(v):
    if not isinstance(v, str):
        return v
    try:
        v = literal_eval(v)
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v

def _check_and_coerce_cfg_value_type(replacement, original, key, full_key):
    original_type = type(original)
    replacement_type = type(replacement)

    # The types must match (with some exceptions)
    if replacement_type == original_type:
        return replacement

    def conditional_cast(from_type, to_type):
        if replacement_type == from_type and original_type == to_type:
            return True, to_type(replacement)
        else:
            return False, None

    casts = [(tuple, list), (list, tuple)]
    try:
        casts.append((str, unicode))  # noqa: F821
    except Exception:
        pass

    for (from_type, to_type) in casts:
        converted, converted_value = conditional_cast(from_type, to_type)
        if converted:
            return converted_value

    raise ValueError(
        "Type mismatch ({} vs. {}) with values ({} vs. {}) for config "
        "key: {}".format(
            original_type, replacement_type, original, replacement, full_key
        )
    )

def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and file.endswith('.yaml'), \
        '{} is not a yaml file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v

    cfg = CfgNode(cfg)
    return cfg

def merge_cfg_from_list(cfg: CfgNode, cfg_list: List[str]):
    new_cfg = copy.deepcopy(cfg)
    assert len(cfg_list) % 2 == 0, cfg_list
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        subkey = full_key.split('.')[-1]
        assert subkey in cfg, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, cfg[subkey], subkey, full_key
        )
        setattr(new_cfg, subkey, value)

    return new_cfg

# ===========================================================================================================================
# === All following helper functions have been borrowed from from https://github.com/facebookresearch/mae/tree/main/utilt ===
# ===========================================================================================================================

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

def split_params(m, decayed_lr=1e-4, kw_list=['norm', 'bias']):
    without_kw, with_kw = split_params_by_keywords(m, kw_list)
    return [
        {'params': with_kw, 'lr': decayed_lr, 'weight_decay': 0.0},
        {'params': without_kw, 'lr': decayed_lr}
    ]

def split_params_by_keywords(m, kw_list):
    without_kw, with_kw = [], []
    for n, p in m.named_parameters():
        if all([n.find(kw) == -1 for kw in kw_list]):
            without_kw.append(p)
        else:
            with_kw.append(p)
    return without_kw, with_kw

def get_params(model, mode='none', lr=1e-4):
    assert mode in ('decay', 'freeze', 'none')
    if mode == 'none':
        return [p for p in model.parameters() if p.requires_grad]
    elif mode == 'freeze':
        for p in model.backbone.backbone.parameters():
            p.requires_grad = False
        return [p for p in model.parameters() if p.requires_grad]
    else:
        backbone_ids = list(map(id, model.backbone.backbone.parameters()))
        other_params = filter(lambda p: id(p) not in backbone_ids, model.parameters())
        param_groups = [{'params': other_params, 'lr': lr}]
        param_groups.append({'params': model.backbone.backbone.norm.parameters(), 'lr': lr, 'weight_decay': 0.0})
        for i in range(11, -1, -1):
            param_groups += split_params(model.backbone.backbone.blocks[i], decayed_lr=lr * (0.7 ** (12 - i)))
        param_groups += split_params(model.backbone.backbone.patch_embed, decayed_lr=lr * (0.7 ** 13))
        param_groups.append({'params': model.backbone.backbone.pos_embed, 'lr': lr * (0.7 ** 13), 'weight_decay': 0.0})
        return param_groups

def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.add_aggreg_ep+30 - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

def load_model_noddp(args, model, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        Log.info("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            Log.info("With optim & sched!")

def save_model_noddp(args, epoch, model, optimizer, loss_scaler, name=None):
    output_dir = Path(args.save_path)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / f'{name}.pth']
        # don't save encoder weight
        weight = {}
        model_dic = model.state_dict()
        for key in model_dic.keys():
            if key[:3] != 'enc':
                weight[key] = model_dic[key]

        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': weight,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

# ============
# === Mine ===
# ============

def ensure_path(path):
    remove_flag = False
    if os.path.exists(path):
        shutil.rmtree(path)
        remove_flag = True
    os.makedirs(path)
    return remove_flag

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# # utility function for saving outputs
# def ensure_path(path, remove=True):
#     if os.path.exists(path):
#         if remove or input('{} exists, remove? ([y]/n): '.format(path)) != 'n':
#             print('remove the existing folder {}'.format(path))
#             shutil.rmtree(path)

#     os.makedirs(path)
