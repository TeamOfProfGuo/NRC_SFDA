import torch
import numpy as np
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from .data_list import ImageList


'''def image_train(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(crop_size),
        transforms.ToTensor(), normalize
    ])'''


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


def office_load(args):
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

        train_source = ImageList(s_tr, transform=image_train(), root='../dataset/')
        test_source = ImageList(s_ts, transform=image_train(), root='../dataset/')
        train_target = ImageList(t_tr, transform=image_target(), root='../dataset/')
        test_target = ImageList(t_ts, transform=image_test(), root='../dataset/')

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
                                      shuffle=True,
                                      num_workers=args.worker,
                                      drop_last=False)
    return dset_loaders
