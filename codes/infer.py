# -*- coding: utf-8 -*-

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from model.GroupAE import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
ckpt = torch.load(f"../ckpts/ECMRNet.pth", map_location=device)
model = IncModel(DCMGroups=ckpt["DCMGroups"], BlockInChs=ckpt["BlockInChs"], sub_deg=ckpt["Subdeg"])
model.load_state_dict(ckpt["state_dict"])
model.to(device)
model.eval()



def pad_to_multiple(x, multiple=4):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple

    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x_pad, (h, w)

def remove_pad(x, orig_size):
    h, w = orig_size
    return x[:, :, :h, :w]

@torch.no_grad()
def fit_Inctasks(tpath, outpath, deg):
    score = defaultdict(float)
    imgp = os.listdir(os.path.join(tpath,deg))
    for img_name in tqdm(imgp):

        lIr = np.array(cv2.imread(os.path.join(tpath,deg,img_name), cv2.IMREAD_GRAYSCALE))
        lIr = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0).unsqueeze(0)
        lIr, size = pad_to_multiple(lIr, 4)
        lIr = lIr.to(device)

        z = model(lIr, deg)
        z = remove_pad(z, size)
        z = z.clamp(0.0, 1.0)

        pre = z.squeeze().cpu().numpy()
        pre = (pre * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(os.path.join(outpath, img_name)), pre)

    score = {k: v / len(imgp) for k, v in score.items()}
    print(score)


@torch.no_grad()
def fit_Realdata(tpath, outpath, deg):
    score = defaultdict(float)
    imgp = os.listdir(os.path.join(tpath,"Ir"))
    for img_name in tqdm(imgp):

        lIr = np.array(cv2.imread(os.path.join(tpath,"Ir",img_name), cv2.IMREAD_GRAYSCALE))
        lIr = torch.from_numpy(lIr).float().div(255.0).unsqueeze(0).unsqueeze(0)
        lIr, size = pad_to_multiple(lIr, 4)
        lIr = lIr.to(device)

        z = model(lIr, deg)
        z = remove_pad(z, size)
        z = z.clamp(0.0, 1.0)
        pre = z.squeeze().cpu().numpy()
        pre = (pre * 255 + 0.5).astype(np.uint8)
        cv2.imwrite(str(os.path.join(outpath, img_name)), pre)

    score = {k: v / len(imgp) for k, v in score.items()}
    print(score)




if __name__ == "__main__":
    for dataset in ["HM-TIR", "M3FD", "AWMM", "TIR100", "EN"]:
        if dataset == "HM-TIR" or dataset == "M3FD":
            tpath = f"../datasets/{dataset}/test"
            for deg in ["contrast", "blur", "noise", "CB", "CN", "BN", "CBN"]:
                print(f"{dataset}-{deg}")
                outpath = f"../datasets/{dataset}/results/{deg}"
                os.makedirs(outpath, exist_ok=True)
                fit_Inctasks(tpath, outpath, deg)
        elif dataset == "AWMM" or dataset == "TIR100" or dataset == "EN":
            tpath = f"../datasets/{dataset}"
            outpath = f"../datasets/{dataset}/results"
            os.makedirs(outpath, exist_ok=True)
            if dataset == "AWMM": deg ="CBN"
            elif dataset == "TIR100": deg ="BN"
            else: deg ="CB"
            print(f"{dataset}-{deg}")
            fit_Realdata(tpath, outpath, deg)

