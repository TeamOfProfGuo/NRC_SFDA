import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC


class TransformSW(object):
    def __init__(self, mean, std, resize=256, crop_size=224, aug_k=1):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(size=crop_size,
                                  padding=int(crop_size*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)