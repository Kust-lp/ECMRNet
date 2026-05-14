import os

import cv2
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pyiqa
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
metrics = {
        'psnr': pyiqa.create_metric('psnr', device=device),
        'ssim': pyiqa.create_metric('ssim', device=device),
    }

def evaluate(HRs, SRs=None):

    results = {}
    with torch.no_grad():
        for name, metric in metrics.items():
                score = metric(HRs, SRs)
                results[name] = round(float(score.item()),4)
    return results

