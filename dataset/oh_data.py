import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.data_list import ImageList
from dataset.data_transform import GaussianBlur, TwoCropsTransform

policies = [transforms.AutoAugmentPolicy.CIFAR10, transforms.AutoAugmentPolicy.IMAGENET, transforms.AutoAugmentPolicy.SVHN]

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


'''def image_test(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    start_first = 0
    start_center = (resize_size - crop_size - 1) / 2
    start_last = resize_size - crop_size - 1

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), normalize
    ])'''


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
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


def mn_transform(min_scales=None):
    if min_scales is None:
        m_scale = 0.2
    else:
        m_scale = min_scales[0]

    return TwoCropsTransform(
        transforms.Compose(get_moco_base_augmentation0(m_scale)),
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


def mw_transform(min_scales=None):
    resize_size = 256
    crop_size = 224

    return TwoCropsTransform(
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize]),
        transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(radius_min=0.1, radius_max=2.0)], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
    )


def get_AutoAug(args):
    if args.data_trans == 'ai':
        return TwoCropsTransform(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
                                 transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))

    elif args.data_trans == 'ac':
        return TwoCropsTransform(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                                 transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))

def get_RandAug(args):
    if args.data_aug is not None:
        num_ops, magnitude = args.data_aug  # list
    else:
        num_ops, magnitude = 2, 9
    return TwoCropsTransform(transforms.RandAugment(num_ops, magnitude),
                             transforms.RandAugment(num_ops, magnitude))


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0],
                       np.array([int(la) for la in val.split()[1:]]))
                      for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1]))
                      for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def office_load(args, ret_idx=False, ss_load=None):
    train_bs = args.batch_size
    if args.home == True:
        ss = args.dset.split('2')[0]
        tt = args.dset.split('2')[1]

        map_dict = {'a': 'Art', 'c': 'Clipart', 'p': 'Product', 'r': 'Real_World'}
        s = map_dict[ss]
        t = map_dict[tt]

        src_list = 'dataset/data_list/office-home/{}.txt'.format(s)
        src_list = open(src_list).readlines()
        s_tr = src_list
        s_ts = src_list
        tar_list = 'dataset/data_list/office-home/{}.txt'.format(t)
        tar_list = open(tar_list).readlines()
        t_tr = tar_list
        t_ts = tar_list

        train_source = ImageList(s_tr, transform=image_train(), root='../dataset/', ret_idx=ret_idx)
        test_source = ImageList(s_ts, transform=image_train(), root='../dataset/', ret_idx=ret_idx)
        train_target = ImageList(t_tr, transform=image_target(), root='../dataset/', ret_idx=ret_idx)
        test_target = ImageList(t_ts, transform=image_test(), root='../dataset/', ret_idx=ret_idx)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source,
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source,
                                           batch_size=train_bs * 2, #2
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dset_loaders["target"] = DataLoader(train_target,
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dset_loaders["test"] = DataLoader(test_target,
                                      batch_size=train_bs * 3, #3
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    if ss_load == 'moco':
        data_target = ImageList(tar_list, transform=moco_transform, root='../dataset/', ret_idx=True)
        dset_loaders['target_ss'] = DataLoader(data_target, batch_size=train_bs, shuffle=True, num_workers=args.worker, drop_last=False)

    return dset_loaders
