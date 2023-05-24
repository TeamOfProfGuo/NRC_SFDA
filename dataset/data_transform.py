import logging
import math
import random
from PIL import ImageFilter
from torchvision import transforms
from .randaugment import RandAugmentMC


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform0, base_transform1):
        self.base_transform0 = base_transform0
        self.base_transform1 = base_transform1

    def __call__(self, x):
        q = self.base_transform0(x)
        k = self.base_transform1(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


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