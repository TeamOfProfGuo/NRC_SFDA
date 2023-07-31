
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from dataset.data_list import ImageList


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(crop_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            torchvision.transforms.Normalize(
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ),
        ]
    )


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")


def l_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("L")


def office_load(args, ret_idx=False):
    train_bs = args.batch_size
    if args.office31 == True:  # and not args.home and not args.visda:
        ss = args.dset.split("2")[0]
        tt = args.dset.split("2")[1]

        map_dict = {'a': "amazon", "d": "dslr", "w": "webcam"}
        s = map_dict[ss]
        t = map_dict[tt]

        s_tr, s_ts = "./data/office/{}_list.txt".format(s), "./data/office/{}_list.txt".format(s)

        txt_src = open(s_tr).readlines()
        dsize = len(txt_src)
        """tv_size = int(1.0 * dsize)
        print(dsize, tv_size, dsize - tv_size)
        s_tr, s_ts = torch.utils.data.random_split(txt_src, [tv_size, dsize - tv_size])"""
        s_tr = txt_src
        s_ts = txt_src

        t_tr, t_ts = "./data/office/{}_list.txt".format(t), "./data/office/{}_list.txt".format(t)
        prep_dict = {}
        prep_dict["source"] = image_train()
        prep_dict["target"] = image_target()
        prep_dict["test"] = image_test()
        train_source = ImageList(s_tr, transform=prep_dict["source"], root='../dataset/', ret_idx=ret_idx)
        test_source = ImageList(s_tr, transform=prep_dict["source"], root='../dataset/', ret_idx=ret_idx)
        train_target = ImageList(open(t_tr).readlines(), transform=prep_dict["target"], root='../dataset/', ret_idx=ret_idx)
        test_target = ImageList(open(t_ts).readlines(), transform=prep_dict["test"], root='../dataset/', ret_idx=ret_idx)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(
        train_source,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["source_te"] = DataLoader(
        test_source,
        batch_size=train_bs * 2,  # 2
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["target"] = DataLoader(
        train_target,
        batch_size=train_bs,
        shuffle=True,
        num_workers=args.worker,
        drop_last=False,
    )
    dset_loaders["test"] = DataLoader(
        test_target,
        batch_size=train_bs * 3,  # 3
        shuffle=False,
        num_workers=args.worker,
        drop_last=False,
    )
    return dset_loaders
