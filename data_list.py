#from __future__ import print_function, division

import torch
import os
import scipy
import os.path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, root=None, transform=None, target_transform=None, mode='RGB', ret_idx=False, pprob=None, ret_plabel=False, args=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        self.root = root
        self.ret_idx = ret_idx
        self.ret_plabel = ret_plabel

        if pprob is not None and args is not None:
            entropy = scipy.stats.entropy(pprob, axis=1)
            weights = 1 - entropy / np.log(args.class_num)
            self.weights = weights / np.max(weights)
            self.plabel = np.argmax(pprob, 1)

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(os.path.join(self.root, path))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        ret = [img, target]
        if self.ret_idx:
            ret.append(index)
        if self.ret_plabel:
            plabel = self.plabel[index]
            weight = self.weights[index]
            if self.target_transform is not None:
                plabel = self.target_transform(plabel)
            ret.append(plabel)
            ret.append(weight)

        return ret  # img (tensor), target (array); idx (array), plabel (array), weight (array)


    def __len__(self):
        return len(self.imgs)
