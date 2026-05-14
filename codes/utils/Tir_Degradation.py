#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import os
import random
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm
from PIL import Image
from albumentations import GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast, Compose, PixelDropout
from albumentations.core.transforms_interface import ImageOnlyTransform

def stripe_noise(image, **params):
    if random.random() < 0.5:
        # g = np.random.randn(1, image.shape[1]) * (np.random.uniform(0.03, 0.1))
        # b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 5)
        g = np.random.randn(1, image.shape[1]) * (np.random.uniform(0.03, 0.07))
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 3)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.uniform(0.03, 0.07))
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 3)
        # g = np.random.randn(image.shape[0], 1) * (np.random.uniform(0.03, 0.1))
        # b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image

def nonuniformity_optical(image, **params):
    h, w = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 55 + 1)
    # delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image


def bad(
    dead_prob=0.002,
    hot_prob=0.001,
    p=1
):
    return Compose([
        PixelDropout(
            dropout_prob=dead_prob,
            per_channel=False,
            drop_value=0,
            p=1
        ),
        PixelDropout(
            dropout_prob=hot_prob,
            per_channel=False,
            drop_value=255,
            p=1
        ),
    ], p=p)

class _TIRVignetting(ImageOnlyTransform):
    def __init__(
        self,
        strength_range=(0.12, 0.28),
        center_x_range=(0.47, 0.53),
        center_y_range=(0.47, 0.53),
        gamma_range=(1.8, 2.6),
        p=1.0,
    ):
        super().__init__(p=p)
        self.strength_range = strength_range
        self.center_x_range = center_x_range
        self.center_y_range = center_y_range
        self.gamma_range = gamma_range

    def apply(self, img, strength=0.2, center_x=0.5, center_y=0.5, gamma=2.2, **params):
        """
        img: HxW uint8
        """
        orig_dtype = img.dtype
        out = img.astype(np.float32)

        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]

        xx = xx / max(w - 1, 1)
        yy = yy / max(h - 1, 1)

        r = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        r = r / (r.max() + 1e-8)

        mask = 1.0 - strength * (r ** gamma)
        mask = np.clip(mask, 0.0, 1.0)

        out = out * mask
        out = np.clip(out, 0, 255).astype(orig_dtype)
        return out

    def get_params(self):
        return {
            "strength": float(np.random.uniform(*self.strength_range)),
            "center_x": float(np.random.uniform(*self.center_x_range)),
            "center_y": float(np.random.uniform(*self.center_y_range)),
            "gamma": float(np.random.uniform(*self.gamma_range)),
        }


def TIRVignetting(p=1):
    return Compose([
        _TIRVignetting(
            strength_range=(0.30, 0.48),
            center_x_range=(0.47, 0.53),
            center_y_range=(0.47, 0.53),
            gamma_range=(1.2, 1.8),
            p=1
        )
    ], p=p)

def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(std_range=(5 / 255.0, 15 / 255.0), p=1),
        # GaussNoise(std_range=(5 / 255.0, 20 / 255.0), p=1),
    ], p=p)

def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.1, 0.2), contrast_limit=(-0.8, -0.4), p=p)
    # return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)

def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 17), sigma_limit=(1, 2), p=1)
        # GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1)
    ], p=p)




def HMTIR(root_path, degrad):
    random.seed(42)
    all_imgs = glob.glob(os.path.join(root_path, "imgs", "*.png"))
    random.shuffle(all_imgs)
    train_imgP = all_imgs[:int(len(all_imgs) * 0.8)]
    test_imgP = all_imgs[int(len(all_imgs) * 0.8):]

    for split, imgP in zip(['train', 'test'], [train_imgP, test_imgP]):
        op = os.path.join(root_path, split)
        tgt_path = os.path.join(op, "tgt")
        os.makedirs(tgt_path, exist_ok=True)
        opd = os.path.join(op, degrad)
        os.makedirs(opd, exist_ok=True)

        random.shuffle(imgP)
        tbar = tqdm(imgP, desc=f"HM-TIR ({split})-{degrad}")
        for i, imgp in enumerate(tbar):
            img_name = os.path.basename(imgp)
            img = Image.open(imgp).convert('L')
            img = np.array(img, dtype=np.uint8)
            img_list = {'image': img}

            if degrad == "contrast":
                img_list = LC()(**img_list)
            if degrad == "blur":
                img_list = Blur()(**img_list)
            if degrad == "noise":
                img_list = Noise()(**img_list)

            if degrad == "CB":
                img_list = LC()(**img_list)
                img_list = Blur()(**img_list)
            if degrad == "CN":
                img_list = LC()(**img_list)
                img_list = Noise()(**img_list)
            if degrad == "BN":
                img_list = Blur()(**img_list)
                img_list = Noise()(**img_list)

            if degrad == "CBN":
                img_list = LC()(**img_list)
                img_list = Blur()(**img_list)
                img_list = Noise()(**img_list)

            cor_img = Image.fromarray(img_list['image']).convert('L')


            cor_img.save(os.path.join(opd, img_name))
            if not os.path.exists(os.path.join(tgt_path, img_name)):
                shutil.copy(os.path.join(root_path,"imgs", img_name), os.path.join(tgt_path, img_name))

def M3FD(root_path, degrad):
    random.seed(42)
    all_imgs = glob.glob(os.path.join(root_path, "imgs", "*.png"))
    random.shuffle(all_imgs)
    test_imgP = all_imgs

    for split, imgP in zip(['test'], [test_imgP]):
        op = os.path.join(root_path, split)
        tgt_path = os.path.join(op, "tgt")
        os.makedirs(tgt_path, exist_ok=True)
        opd = os.path.join(op, degrad)
        os.makedirs(opd, exist_ok=True)

        random.shuffle(imgP)
        tbar = tqdm(imgP, desc=f"M3FD ({split})-{degrad}")
        for i, imgp in enumerate(tbar):
            img_name = os.path.basename(imgp)
            img = Image.open(imgp).convert('L')
            img = np.array(img, dtype=np.uint8)
            img_list = {'image': img}

            if degrad == "contrast":
                img_list = LC()(**img_list)
            if degrad == "blur":
                img_list = Blur()(**img_list)
            if degrad == "noise":
                img_list = Noise()(**img_list)

            if degrad == "CB":
                img_list = LC()(**img_list)
                img_list = Blur()(**img_list)
            if degrad == "CN":
                img_list = LC()(**img_list)
                img_list = Noise()(**img_list)
            if degrad == "BN":
                img_list = Blur()(**img_list)
                img_list = Noise()(**img_list)

            if degrad == "CBN":
                img_list = LC()(**img_list)
                img_list = Blur()(**img_list)
                img_list = Noise()(**img_list)

            cor_img = Image.fromarray(img_list['image']).convert('L')


            cor_img.save(os.path.join(opd, img_name))
            if not os.path.exists(os.path.join(tgt_path, img_name)):
                shutil.copy(os.path.join(root_path,"imgs", img_name), os.path.join(tgt_path, img_name))


if __name__ == "__main__":

    for data in ["HM-TIR", "M3FD"]:
        root_path = f"../../datasets/{data}"

        for deg in ["contrast", "blur", "noise", "CB", "CN", "BN", "CBN"]:
            if data == "HM-TIR": HMTIR(root_path, deg)
            if data == "M3FD": M3FD(root_path, deg)
