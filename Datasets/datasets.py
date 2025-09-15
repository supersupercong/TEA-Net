import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms

import cv2


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in
               ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF', 'tiff'])


def transform():
    return Compose([
        ToTensor(),
    ])


def load_img(filepath, is_gray=False, is_ycbcr=False):
    img = Image.open(filepath).convert('RGB')
    if is_gray:
        img = img.convert('L')
    if is_ycbcr:
        img = img.convert('YCbCr').split()[0]
    return img


def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(A_image, B_image, patch_size, scale=1, ix=-1, iy=-1):
    (ih, iw) = B_image.size
    (th, tw) = (scale * ih, scale * iw)

    tp = scale * patch_size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    B_image = B_image.crop((iy, ix, iy + ip, ix + ip))
    A_image = A_image.crop((ty, tx, ty + tp, tx + tp))

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return A_image, B_image, info_patch


def augment(A_image, B_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}

    if random.random() < 0.5 and flip_h:
        A_image = ImageOps.flip(A_image)
        B_image = ImageOps.flip(B_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            A_image = ImageOps.mirror(A_image)
            B_image = ImageOps.mirror(B_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            A_image = A_image.rotate(180)
            B_image = B_image.rotate(180)

            info_aug['trans'] = True

    return A_image, B_image, info_aug


class Data(data.Dataset):  # self, config, is_train=True,
    def __init__(self, cfg, is_train=True, is_label=False, is_det=False, is_getpatch=False,
                            is_grayA=False, is_grayB=False, is_ycbcrA=False, is_ycbcrB=False, transform=transform()):
        super(Data, self).__init__()
        self.cfg = cfg
        print(cfg)
        self.is_train = is_train
        self.is_label = is_label
        self.is_grayA, self.is_grayB = is_grayA, is_grayB
        self.is_ycbcrA, self.is_ycbcrB = is_ycbcrA, is_ycbcrB
        self.is_getpatch = is_getpatch
        if is_train == True:
            data_dir_A = cfg[cfg['train_dataset']]['data_dir']['train_dir']['data_dir_A']
            data_dir_B = cfg[cfg['train_dataset']]['data_dir']['train_dir']['data_dir_B']
            self.data_augmentation = cfg[cfg['train_dataset']]['data_augmentation']
            self.A_image_filenames = sorted([join(data_dir_A, x) for x in listdir(data_dir_A) if is_image_file(x)])
            self.B_image_filenames = sorted([join(data_dir_B, x) for x in listdir(data_dir_B) if is_image_file(x)])
            #  self.cfg[self.cfg['train_dataset']]['max_value']
            self.patch_size = cfg[cfg['train_dataset']]['patch_size']
            self.transform = transform
        else:
            if is_det:
                data_dir_A = cfg[cfg['train_dataset']]['data_dir']['det_dir']['data_dir_A']
                data_dir_B = cfg[cfg['train_dataset']]['data_dir']['det_dir']['data_dir_B']
            else:
                data_dir_A = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_A']
                data_dir_B = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_B']
            
            self.A_image_filenames = sorted([join(data_dir_A, x) for x in listdir(data_dir_A) if is_image_file(x)])
            self.B_image_filenames = sorted([join(data_dir_B, x) for x in listdir(data_dir_B) if is_image_file(x)])
            if is_label == True:
                if is_det:
                    data_dir_gt = cfg[cfg['train_dataset']]['data_dir']['det_dir']['data_dir_gt']
                else:
                    data_dir_gt = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_gt']
                self.gt_image_filenames = sorted(
                    [join(data_dir_gt, x) for x in listdir(data_dir_gt) if is_image_file(x)])
            self.data_augmentation = False
            if self.is_getpatch:
                self.patch_size = cfg[cfg['train_dataset']]['patch_size']
            self.transform = transform

    def __getitem__(self, index):
        if self.is_train == True:
            A_image = load_img(self.A_image_filenames[index], is_gray=self.is_grayA, is_ycbcr=self.is_ycbcrA)
            B_image = load_img(self.B_image_filenames[index], is_gray=self.is_grayB, is_ycbcr=self.is_ycbcrB)
            _, file = os.path.split(self.A_image_filenames[index])
            A_image, B_image, _ = get_patch(A_image, B_image, self.patch_size)
            if self.data_augmentation:
                A_image, B_image, _ = augment(A_image, B_image)

            if self.transform:
                real_A = self.transform(A_image)
                real_B = self.transform(B_image)
            return real_A, real_B

        else:
            A_image = load_img(self.A_image_filenames[index], is_gray=self.is_grayA, is_ycbcr=self.is_ycbcrA)
            B_image = load_img(self.B_image_filenames[index], is_gray=self.is_grayB, is_ycbcr=self.is_ycbcrB)
            if self.is_label == True:
                gt_image = load_img(self.gt_image_filenames[index], is_gray=True)

            _, file = os.path.split(self.A_image_filenames[index])

            if self.is_getpatch:
                A_image, B_image, _ = get_patch(A_image, B_image, self.patch_size)

            if self.transform:
                real_A = self.transform(A_image)
                real_B = self.transform(B_image)
                if self.is_label == True:
                    gt_image = self.transform(gt_image)
                    return real_A, real_B, gt_image, file
                else:
                    return real_A, real_B, file

            else:
                if self.is_label == True:
                    return A_image, B_image, gt_image, file
                else:
                    return A_image, B_image, file

    def __len__(self):
        return len(self.A_image_filenames)


