"""
This module implements an abstract base class (ABC) 'BaseDataset' for datasets alnog with some transformation function for image preprocessing.
"""
import random
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import os.path as osp
from PIL import Image
from abc import ABC, abstractmethod
from util.dataset_preprocess import get_normalize_statistics


class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    -- <get_classes>:                   return mapping between class name and id of the dataset
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass
    
    @abstractmethod
    def get_classes(self):
        """Return the class information
        
        Returns:
            a dictionary containing the class name and its corresponding category id
        """
        pass


"""Following are some frequently used method for image transformation"""

def make_transform(opt, rescale_method=Image.BICUBIC, isLabel=False, toTensor=False):
    transform_list = []
    if 'crop' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: random_resize_crop(img, opt.load_size, rescale_method)))
    elif 'rescale' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: rescale(img, opt.load_size, rescale_method)))
    if 'jitter' in opt.preprocess and not isLabel:
        transform_list.append(transforms.ColorJitter(saturation=0.5, brightness=0.5))
    if 'blur' in opt.preprocess and not isLabel:
        transform_list.append(transforms.GaussianBlur((5, 9), sigma=(0.1, 5)))
    if 'grayscale' in opt.preprocess and not isLabel:
        transform_list.append(transforms.Grayscale(3))
    if 'flip' in opt.preprocess:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    if 'cifar10' in opt.preprocess and not isLabel:
        transform_list.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10, fill=opt.ignore_label))
    if 'rotate' in opt.preprocess:
        transform_list.append(transforms.RandomRotation(30, fill=opt.ignore_label))
    if toTensor:
        transform_list.append(transforms.ToTensor())
        transform_list += [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    print(transform_list)
    return transforms.Compose(transforms=transform_list)

def random_resize_crop(image, target_size, method=Image.BICUBIC, crop_ratio=0.65):
    ow, oh = image.size
    crop_ratio = float(torch.rand(1)) * (1-crop_ratio) + crop_ratio
    # get target size for resizing
    tw = target_size
    th = int(target_size * oh / ow)

    # get target size for cropping
    cw = ow*crop_ratio
    ch = oh*crop_ratio

    # get starting location for cropping
    sx = int(torch.randint(int(ow-cw), (1, ))) if int(ow-cw) > 0 else 0
    sy = int(torch.randint(int(oh-ch), (1, ))) if int(oh-ch) > 0 else 0

    img = image.crop((sx, sy, sx+cw, sy+ch))
    return img.resize((tw, th), method)

def rescale(image, target_size, method=Image.BICUBIC):
    if type(image) is np.ndarray:
        image = Image.fromarray(image)
    ow, oh = image.size
    tw = target_size
    th = int(target_size * oh / ow)
    return image.resize((tw, th), method)

def central_crop(img, new_size):
    w, h = img.size
    nh, nw = new_size
    dh, dw = (h-nh, w-nw)
    img = img.crop((dw/2, dh/2, dw/2+nw, dh/2+nh))
    img = np.asarray(img, dtype=np.uint32)
    return img
