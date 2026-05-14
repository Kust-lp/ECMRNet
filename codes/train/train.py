# -*- coding: utf-8 -*-
import argparse
import os
import sys

import torch
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from timm.utils import ModelEmaV2

from _2DSE_Pruning import Pruning

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from model.GroupAE import *
from utils.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

@torch.no_grad()
def Validate(model, val_loader, args, loss_fn, model_his, deg, his=True):
    model.eval()
    avg_loss = defaultdict(float)
    score = defaultdict(float)
    for lIr, hIr, name in tqdm(val_loader, desc="Val"):
        lIr = lIr.to(args.device)
        hIr = hIr.to(args.device)

        hs = None
        if his:
            if deg == "CB":
                with torch.no_grad():
                    hsc = model_his["contrast"].forward_stage(lIr)
                    hsb = model_his["blur"].forward_stage(lIr)
                    hs = [hsc, hsb]
            elif deg == "CN":
                with torch.no_grad():
                    hsc = model_his["contrast"].forward_stage(lIr)
                    hsn = model_his["noise"].forward_stage(lIr)
                    hs = [hsc, hsn]
            elif deg == "BN":
                with torch.no_grad():
                    hsb = model_his["blur"].forward_stage(lIr)
                    hsn = model_his["noise"].forward_stage(lIr)
                    hs = [hsb, hsn]
            elif deg == "CBN":
                with torch.no_grad():
                    hsc = model_his["contrast"].forward_stage(lIr)
                    hsb = model_his["blur"].forward_stage(lIr)
                    hsn = model_his["noise"].forward_stage(lIr)
                    hs = [hsc, hsb, hsn]

        z = model(lIr, hs)
        L, info = loss_fn(z, hIr)
        for k, v in info.items():
            avg_loss[k] += v

        feat = z.clamp(0.0, 1.0)
        res = evaluate(hIr, feat)
        for k, v in res.items():
            score[k] += v
        pre = z.squeeze().cpu().numpy()
        pre = (pre * 255 + 0.5).astype(np.uint8)
        name = name[0]
        cv2.imwrite(str(os.path.join(args.spath, f"{name}.png")), pre)

    score = {k: v / len(val_loader) for k, v in score.items()}
    avg_loss = {k: v / len(val_loader) for k, v in avg_loss.items()}
    return avg_loss, score


def model_train(model, opt, train_loader, val_loader, args, model_his, deg, epochs, meta= None, tuning= False):
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)
    warmup = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(opt, T_max=(total_steps - warmup_steps), eta_min=1e-6)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])
    ema = ModelEmaV2(model, decay=0.999)

    loss_fn = MainLosses().to(args.device)
    os.makedirs(args.checkpoint, exist_ok=True)
    best_score = 1e10
    patience = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        avg_loss = defaultdict(float)

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{epochs}") as pbar:
            for lIr, hIr, name in train_loader:
                lIr = lIr.to(args.device)
                hIr = hIr.to(args.device)

                opt.zero_grad(set_to_none=True)
                hs = None
                if deg == "CB":
                    with torch.no_grad():
                        hsc = model_his["contrast"].forward_stage(lIr)
                        hsb = model_his["blur"].forward_stage(lIr)
                        hs = [hsc, hsb]
                elif deg == "CN":
                    with torch.no_grad():
                        hsc = model_his["contrast"].forward_stage(lIr)
                        hsn = model_his["noise"].forward_stage(lIr)
                        hs = [hsc, hsn]
                elif deg == "BN":
                    with torch.no_grad():
                        hsb = model_his["blur"].forward_stage(lIr)
                        hsn = model_his["noise"].forward_stage(lIr)
                        hs = [hsb, hsn]
                elif deg == "CBN":
                    with torch.no_grad():
                        hsc = model_his["contrast"].forward_stage(lIr)
                        hsb = model_his["blur"].forward_stage(lIr)
                        hsn = model_his["noise"].forward_stage(lIr)
                        hs = [hsc, hsb, hsn]


                z = model(lIr, hs)
                L, info = loss_fn(z, hIr)

                L.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ema.update(model)
                scheduler.step()

                for k, v in info.items():
                    avg_loss[k] += v
                pbar.set_postfix({k: v / (pbar.n + 1) for k, v in avg_loss.items()})
                pbar.update(1)
        if epoch % 1 == 0:
            eval_model = ema.module
            avg_loss, score = Validate(eval_model, val_loader, args, loss_fn, model_his, deg)
            print(f"Val: \nLoss: {avg_loss} \nMetrics: {score}")
            current_score = (
                    (1 - score["ssim"]) +
                    (50 - score["psnr"]) / 50
            )
            if current_score < best_score:
                best_score = current_score
                ckpt = {
                    "state_dict": eval_model.state_dict(),
                    "KeepCH_nums": meta,
                }
                mname =  f"{args.checkpoint}/Group_{deg}_tuning.pth" if tuning else f"{args.checkpoint}/Group_{deg}.pth"
                torch.save(ckpt,mname)
                patience = 0
                best_epoch = epoch

            else:
                patience += 1
                if patience >= args.max_patience:
                    print(f"Early stopping, best epoch: {best_epoch}")
                    break


def model_load(args, hps):
    model = GroupAE(*hps)
    model.load_state_dict(torch.load(f"{args.checkpoint}/GroupAE.pth", map_location=args.device), strict=False)
    model.to(args.device)
    return model


def Tuning_Model_load(args, his_deg=["contrast","blur","noise"]):
    group_size = [int(a / b) for a, b in zip(args.modelHps[-2], args.modelHps[-3])]
    hps = args.modelHps.copy()

    hModels = {}
    for deg in his_deg:
        ckpt = torch.load(f"{args.checkpoint}/Group_{deg}_tuning.pth", map_location=args.device)
        hps[-3] = [a // b for a, b in zip(ckpt["KeepCH_nums"], group_size)]
        hps[-2] = ckpt["KeepCH_nums"]
        model = GroupAE(*hps)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.to(args.device).eval()
        hModels[deg] = model

    return hModels

def PFreeze(args, model):
    for p in model.stem.parameters():
        p.requires_grad = False
    for p in model.bottom.parameters():
        p.requires_grad = False
    for p in model.out.parameters():
        p.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_MB = total_params * 4 / (1024 * 1024)
    print(f"Trainable parameters: {total_params / 1e6:.2f}M ({size_in_MB:.2f} MB)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=args.weight_decay)
    return model, opt

def data_load(args, deg):
    train_dataset = DCMDataset(args.dataset, split='train', degarde=deg)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    val_dataset = DCMDataset(args.dataset, split='test', degarde=deg)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )
    return train_loader, val_loader



def Emain():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="../../datasets/HM-TIR")
    parser.add_argument('--max_patience', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--tuning_epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--checkpoint', type=str, default="../../ckpts")
    parser.add_argument('--pred_image', type=str, default="../../datasets/HM-TIR/test")
    parser.add_argument('--modelHps', type=list, default=[1, 32, [2,4,8],0.0, [8,16,32,32,16,8], [32, 128, 512, 512, 128, 32], 0 ], help="in_ch, base_ch, num_blocks, drop, groups, BlockInCh")

    args = parser.parse_args()
    for deg in ["contrast", "blur", "noise", "CB", "CN", "BN", "CBN"]:
        model_his = None

        print(f"\n\nCurrent degradation: {deg}")
        args.spath = os.path.join(args.pred_image, f"Pred_{deg}")
        os.makedirs(args.spath, exist_ok=True)

        train_loader, val_loader = data_load(args, deg)

        hps = args.modelHps.copy()
        if deg in ["CB", "CN", "BN"]:
            hps[-1] = 2
            if deg == "CB":
                model_his = Tuning_Model_load(args, ["contrast", "blur"])
            if deg == "CN":
                model_his = Tuning_Model_load(args, ["contrast", "noise"])
            if deg == "BN":
                model_his = Tuning_Model_load(args, ["blur", "noise"])
        if deg == "CBN":
            hps[-1] = 3
            model_his = Tuning_Model_load(args, ["contrast", "blur", "noise"])


        model, opt = PFreeze(args, model_load(args, hps))


        if os.path.exists(f"{args.checkpoint}/Group_{deg}.pth"):
            print(f"\n{deg} degradation model loaded")
            model.load_state_dict(torch.load(f"{args.checkpoint}/Group_{deg}.pth", map_location=args.device)["state_dict"], strict=True)

        else:
            print(f"\n{deg} degradation model training")
            model_train(model, opt, train_loader, val_loader, args, model_his, deg, args.epochs)


        if os.path.exists(f"{args.checkpoint}/Group_{deg}_tuning.pth"):
            print(f"\n{deg} degradation tuning model loaded")
        else:

            print(f"\n{deg} degradation model pruning")
            model, meta = Pruning(args, model, deg)
            model.to(args.device)

            print(f"\n{deg} degradation model tuning")
            model, opt = PFreeze(args, model)
            model_train(model, opt, train_loader, val_loader, args, model_his, deg, args.tuning_epochs, meta, tuning=True)

        del model_his, model
        torch.cuda.empty_cache()

    model_merge(args)





if __name__ == "__main__":
    Emain()
