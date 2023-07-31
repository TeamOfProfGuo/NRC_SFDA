
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_list import ImageList
from dataset.data_transform import TransformSW, GaussianBlur, TwoCropsTransform, DataAugmentationDINO

visda_classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse' 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def get_moco_base_augmentation0(min_scale=None, max_scale=None):
    if min_scale is None:
        min_scale = 0.2
    if max_scale is None: 
        max_scale = 1.0
    return [
        transforms.RandomResizedCrop(224, scale=(min_scale, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def get_moco_base_augmentation1(min_scale=None):
    if min_scale is None:
        min_scale = 0.5
    return [
        transforms.RandomResizedCrop(224, scale=(min_scale, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

def moco_transform(min_scales=None):
    if min_scales is None:
        return TwoCropsTransform(transforms.Compose(get_moco_base_augmentation0()),
                                 transforms.Compose(get_moco_base_augmentation1()))
    else:
        return TwoCropsTransform(transforms.Compose(get_moco_base_augmentation0(min_scales[0])),
                                 transforms.Compose(get_moco_base_augmentation1(min_scales[1])))


def mm_transform(min_scales=None):
    if min_scales is None:
        m_scale = 0.5
    else:
        m_scale = min_scales[0]

    return TwoCropsTransform(
        transforms.Compose(get_moco_base_augmentation0()),
        transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(m_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
  

def mn_transform(min_scales=None):
    if min_scales is None:
        m_scale = 0.5
    else:
        m_scale = min_scales[0]

    return TwoCropsTransform(
        transforms.Compose(get_moco_base_augmentation0()),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    

def data_load(args, ss_load=None):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)

    data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()
    dsets["target"] = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    dsets["test"] = ImageList(txt_test, transform=image_test(), root=os.path.dirname(args.test_dset_path), ret_idx=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker, drop_last=False)

    if ss_load == 'moco':
        dsets['target_ss'] = ImageList(txt_tar, transform=moco_transform, root=os.path.dirname(args.t_dset_path), ret_idx=True)
        dset_loaders['target_ss'] = DataLoader(dsets['target_ss'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)
    elif ss_load == 'dino':
        dino_transform = DataAugmentationDINO(global_crops_scale=[0.6, 1.0], local_crops_scale=[0.2, 0.6],
                                              local_crops_number=args.local_crops_num)
        dsets['target_ss'] = ImageList(txt_tar, transform=dino_transform, root=os.path.dirname(args.t_dset_path), ret_idx=True)
        dset_loaders['target_ss'] = DataLoader(dsets['target_ss'], batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders


def reset_data_load(dset_loaders, pred_prob, args, ss_load=None):
    """
    modify the target data loader to return both image and pseudo label
    """
    txt_tar = open(args.t_dset_path).readlines()

    if ss_load == 'moco':
        data_trans = moco_transform
    elif ss_load == 'dino':
        data_trans = DataAugmentationDINO(global_crops_scale=[0.6, 1.0], local_crops_scale=[0.2, 0.6],
                                          local_crops_number=args.local_crops_num)
    else:
        data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()

    dsets = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True, pprob=pred_prob, ret_plabel=True, args=args)
    dloader = DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=args.worker, drop_last=False)
    if ss_load == 'moco' or ss_load == 'dino':
        dset_loaders['target_ss'] = dloader
    else:
        dset_loaders['target'] = dloader