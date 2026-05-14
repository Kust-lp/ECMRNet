import random
from collections import defaultdict
from typing import List

import cv2
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

from model.GroupAE import IncModel


class AWDataset(Dataset):
    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir / "src"
        self.src_files = sorted(list(self.src_dir.glob("*.png")))


    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
        return lIr_tensor, name

class CleanAEDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = Path(root_dir)
        self.split = split
        if split == "train":
            self.src_dir = self.root_dir / split / "patches" / "tgt"

        else:
            self.src_dir = self.root_dir / split / "tgt"


        self.src_files = sorted(list(self.src_dir.glob("*.png")))


    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        Ir_p = self.src_files[idx]
        name = Ir_p.stem
        Ir = np.array(cv2.imread(Ir_p, cv2.IMREAD_GRAYSCALE))
        Ir_tensor = torch.from_numpy(Ir).float().div(255.0).unsqueeze(0)

        return Ir_tensor, Ir_tensor, name

class DCMDataset(Dataset):
    def __init__(self, root_dir, split='train', degarde="contrast"):
        self.root_dir = Path(root_dir)
        self.split = split
        if split == "train":
            self.src_dir = self.root_dir / split / "patches" / degarde
            self.tgt_dir = self.root_dir / split /  "patches" / "tgt"

        else:
            self.src_dir = self.root_dir / split /degarde
            self.tgt_dir = self.root_dir / split / "tgt"


        self.src_files = sorted(list(self.src_dir.glob("*.png")))
        self.tgt_files = sorted(list(self.tgt_dir.glob("*.png")))

        assert len(self.src_files) == len(self.tgt_files)

    def __len__(self):
        return len(self.tgt_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        hIr_p = self.tgt_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
        hIr = np.array(cv2.imread(hIr_p, cv2.IMREAD_GRAYSCALE))

        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
        hIr_tensor = torch.from_numpy(hIr).float().div(255.0).unsqueeze(0)

        return lIr_tensor, hIr_tensor, name

class PuningDataset(Dataset):
    def __init__(self, root_dir,  degarde="contrast", simple_num=300):
        self.root_dir = Path(root_dir)
        self.src_dir = self.root_dir/ "test" / degarde
        all_files = sorted(list(self.src_dir.glob("*.png")))
        self.src_files = random.sample(all_files, simple_num)

    def __len__(self):
        return len(self.src_files)

    def __getitem__(self, idx):
        lIr_p = self.src_files[idx]
        name = lIr_p.stem

        lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))

        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)

        return lIr_tensor, name

class CaptureDataset(Dataset):
    def __init__(self, root_dir,  his_deg, cur_deg):
        self.root_dir = Path(root_dir)
        self.his_dir = self.root_dir/ "test" / his_deg
        self.cur_dir = self.root_dir/ "test" / cur_deg
        self.gt_dir = self.root_dir/ "test" / "tgt"
        self.his_files = sorted(list(self.his_dir.glob("*.png")))
        self.cur_files = sorted(list(self.cur_dir.glob("*.png")))
        self.gt_files = sorted(list(self.gt_dir.glob("*.png")))
        assert len(self.his_files) == len(self.cur_files)

    def __len__(self):
        return len(self.his_files)

    def __getitem__(self, idx):
        his_p = self.his_files[idx]
        cur_p = self.cur_files[idx]
        gt_p = self.gt_files[idx]

        his = np.array(cv2.imread(his_p, cv2.IMREAD_GRAYSCALE))
        cur = np.array(cv2.imread(cur_p, cv2.IMREAD_GRAYSCALE))
        gt = np.array(cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE))

        his_tensor = torch.from_numpy(his).float().div(255.0).unsqueeze(0)
        cur_tensor = torch.from_numpy(cur).float().div(255.0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt).float().div(255.0).unsqueeze(0)

        return his_tensor, cur_tensor, gt_tensor

class MainLosses(nn.Module):
    def __init__(self, alpha=1, beta=0.5, gamma=0.1):
        super(MainLosses, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def grad(self, x, y):
        def sobel(z):

            grad_x = F.conv2d(z, self.sobel_x, padding=1)
            grad_y = F.conv2d(z, self.sobel_y, padding=1)
            return grad_x, grad_y

        x_gx, x_gy = sobel(x)
        y_gx, y_gy = sobel(y)


        grad_diff = 0.5 * (F.l1_loss(x_gx, y_gx) + F.l1_loss(x_gy, y_gy))

        return grad_diff

    def forward(self, pred, target):
        info = defaultdict(float)


        l1 = F.l1_loss(pred, target)
        info['L1'] = l1.item()


        ssimn = 1 - ms_ssim(pred, target, data_range=1.0, size_average=True)
        info['SSIM'] = ssimn.item()

        grad = self.grad(pred, target)
        info['Grad'] = grad.item()



        loss = self.alpha * l1 + self.beta * ssimn + self.gamma * grad
        info['Total'] = loss.item()

        return loss, info



from albumentations import Compose, RandomCrop, HorizontalFlip, VerticalFlip, GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast

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

def Gen_degrade(deg, patch):
    if deg == "contrast":
        return LC()(**patch)
    elif deg == "blur":
        return Blur()(**patch)
    elif deg == "noise":
        return Noise()(**patch)
    elif deg == "CB":
        patch = LC()(**patch)
        patch = Blur()(**patch)
        return patch
    elif deg == "CN":
        patch = LC()(**patch)
        patch = Noise()(**patch)
        return patch
    elif deg == "BN":
        patch = Blur()(**patch)
        patch = Noise()(**patch)
        return patch
    elif deg == "CBN":
        patch = LC()(**patch)
        patch = Blur()(**patch)
        patch = Noise()(**patch)
        return patch



class DCMDataset1(Dataset):
    def __init__(self, root_dir, split='train', degarde="contrast"):
        self.root_dir = Path(root_dir)
        self.split = split
        self.deg = degarde
        self.transform = Compose([
            RandomCrop(height=256, width=256, p=1.0),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
        ])
        if split == "train":
            self.img_dir = self.root_dir / split /  "tgt"
            self.img_files = sorted(list(self.img_dir.glob("*.png")))

        else:
            self.src_dir = self.root_dir / split / degarde
            self.tgt_dir = self.root_dir / split / "tgt"
            self.src_files = sorted(list(self.src_dir.glob("*.png")))
            self.tgt_files = sorted(list(self.tgt_dir.glob("*.png")))

            assert len(self.src_files) == len(self.tgt_files)

    def __len__(self):
        return len(self.img_files) if self.split == "train" else len(self.src_files)

    def __getitem__(self, idx):
        if  self.split == "train":
            img_p = self.img_files[idx]
            name = img_p.stem
            img = np.array(cv2.imread(img_p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8)
            patch = self.transform(image=img)
            hIr = patch['image']
            lIr = Gen_degrade(self.deg, patch)['image']
        else:

            lIr_p = self.src_files[idx]
            hIr_p = self.tgt_files[idx]
            name = lIr_p.stem

            lIr = np.array(cv2.imread(lIr_p, cv2.IMREAD_GRAYSCALE))
            hIr = np.array(cv2.imread(hIr_p, cv2.IMREAD_GRAYSCALE))

        lIr_tensor = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0)
        hIr_tensor = torch.from_numpy(hIr).float().div(255.0).unsqueeze(0)

        return lIr_tensor, hIr_tensor, name


def model_merge(args):
    C = torch.load(f"{args.checkpoint}/Group_contrast_tuning.pth", map_location=args.device)
    B = torch.load(f"{args.checkpoint}/Group_blur_tuning.pth", map_location=args.device)
    N = torch.load(f"{args.checkpoint}/Group_noise_tuning.pth", map_location=args.device)
    CB = torch.load(f"{args.checkpoint}/Group_CB_tuning.pth", map_location=args.device)
    CN = torch.load(f"{args.checkpoint}/Group_CN_tuning.pth", map_location=args.device)
    BN = torch.load(f"{args.checkpoint}/Group_BN_tuning.pth", map_location=args.device)
    CBN = torch.load(f"{args.checkpoint}/Group_CBN_tuning.pth", map_location=args.device)

    Keep_CHs = {"contrast": C["KeepCH_nums"], "blur": B["KeepCH_nums"], "noise": N["KeepCH_nums"],
                "CB": CB["KeepCH_nums"], "CN": CN["KeepCH_nums"], "BN": BN["KeepCH_nums"],
                "CBN": CBN["KeepCH_nums"]}
    BlockInCh = [32, 128, 512, 512, 128, 32]
    groups = [8, 16, 32, 32, 16, 8]
    group_size = [int(a / b) for a, b in zip(BlockInCh, groups)]
    new_groups = {k: [int(m / gs) for m, gs in zip(v, group_size)] for k, v in Keep_CHs.items()}
    sub_deg = {"contrast": 0, "blur": 0, "noise": 0, "CB": 2, "CN": 2, "BN": 2, "CBN": 3}


    model = IncModel(DCMGroups=new_groups, BlockInChs=Keep_CHs, sub_deg=sub_deg)
    model.contrast.load_state_dict(C["state_dict"], strict=False)
    model.blur.load_state_dict(B["state_dict"], strict=False)
    model.noise.load_state_dict(N["state_dict"], strict=False)
    model.CB.load_state_dict(CB["state_dict"], strict=False)
    model.CN.load_state_dict(CN["state_dict"], strict=False)
    model.BN.load_state_dict(BN["state_dict"], strict=False)
    model.CBN.load_state_dict(CBN["state_dict"], strict=False)

    model.stem.weight.data.copy_(C["state_dict"]["stem.weight"])
    model.stem.bias.data.copy_(C["state_dict"]["stem.bias"])

    model.bottom.weight.data.copy_(C["state_dict"]["bottom.weight"])
    model.bottom.bias.data.copy_(C["state_dict"]["bottom.bias"])

    model.out.weight.data.copy_(C["state_dict"]["out.weight"])
    model.out.bias.data.copy_(C["state_dict"]["out.bias"])

    ckpt = {
        "state_dict": model.state_dict(),
        "BlockInChs": Keep_CHs,
        "DCMGroups": new_groups,
        "Subdeg": sub_deg
    }
    torch.save(ckpt, f"{args.checkpoint}/ECMRNet.pth")


