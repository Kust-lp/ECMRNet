#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm



def compute_starts(L: int, patch: int, stride: int, cover_edges: bool):

    assert L >= patch, "Image size is too small!"

    starts = list(range(0, L - patch + 1, stride))
    if cover_edges:
        last = L - patch
        if starts[-1] != last:
            starts.append(last)
    return starts

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def save_patch(img, y, x, patch_size, out_dir, base):
    patch = img[y:y+patch_size, x:x+patch_size]
    out_name = f"{base}_y{y:05d}_x{x:05d}.png"
    cv2.imwrite(os.path.join(out_dir, out_name), patch)
    return out_name

def patchify_pair_dir(src, tgt, out_src, out_tgt, P, S, cover):

    ensure_dir(out_src)
    ensure_dir(out_tgt)

    p_num = 0
    img_list = sorted([f for f in os.listdir(tgt) if f.lower().endswith(".png")])
    for name in tqdm(img_list, desc=f"Patchifying {os.path.basename(os.path.dirname(out_src))}"):
        base = os.path.splitext(name)[0]
        src_path = os.path.join(src, name)
        tgt_path = os.path.join(tgt, name)
        if not (os.path.isfile(src_path) and os.path.isfile(tgt_path)):
            continue

        src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        tgt_img = cv2.imread(tgt_path, cv2.IMREAD_GRAYSCALE)
        H, W = src_img.shape
        assert tgt_img.shape == (H, W), f"Size mismatch: {src_path} vs {tgt_path}"

        ys = compute_starts(H, P, S, cover)
        xs = compute_starts(W, P, S, cover)

        for y in ys:
            y = min(y, H - P) if H >= P else 0
            for x in xs:
                x = min(x, W - P) if W >= P else 0
                if y + P <= H and x + P <= W:
                    save_patch(src_img, y, x, P, out_src, base)
                    if not os.path.exists(os.path.join(tgt_path, f"{base[:-4]}_y{y:05d}_x{x:05d}.png")):
                        save_patch(tgt_img, y, x, P, out_tgt, base)
                    p_num += 1
    print(f"Patches for train: {p_num}")

def main(args, deg):
    P = args.patch_size
    S = args.stride
    cover = bool(args.cover_edges)
    src = os.path.join(args.root, deg)
    tgt = os.path.join(args.root, "tgt")
    out_src = os.path.join(args.root, "patches", deg)
    out_tgt = os.path.join(args.root, "patches", "tgt")
    patchify_pair_dir(src, tgt, out_src, out_tgt, P, S, cover)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="../../datasets/HM-TIR/train")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--cover_edges", type=int, default=0,
                    help="1 covers to the right/bottom edge; 0 discards incomplete blocks")
    args = ap.parse_args()

    for deg in ["contrast", "blur", "noise", "CB", "CN", "BN", "CBN"]:
        print(deg)
        main(args, deg)
