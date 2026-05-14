# -*- coding: utf-8 -*-
import argparse
import os
import sys
import torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
torch.backends.cudnn.benchmark = True
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from tqdm import tqdm
from utils.utils import *
from model.GroupAE import *
from model._2DSE import *
from utils.evaluate import evaluate
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from collections import OrderedDict,Counter

def build_keep_idx_from_groups(Keep_groups,group_size, G, device):

    Keep_groups = sorted(list(Keep_groups))
    keep_ch = []
    for g in Keep_groups:
        assert 0 <= g < G
        keep_ch.extend(range(g * group_size, (g + 1) * group_size))
    keep_idx = torch.tensor(keep_ch, dtype=torch.long, device=device)
    return keep_idx, Keep_groups

def groups_to_keep_idx(Keep_groups, group_size, device=None):
    Keep_groups = sorted(list(Keep_groups))
    idx = []
    for g in Keep_groups:
        idx.extend(range(g * group_size, (g + 1) * group_size))
    keep_idx = torch.tensor(idx, dtype=torch.long, device=device)
    return keep_idx, Keep_groups

@torch.no_grad()
def prune_groupresblock(old_blk: GroupResBlock,
                        keep_idx: torch.Tensor,
                        new_C: int,
                        new_groups: int):
    # 还原 drop 概率
    drop_p = old_blk.drop.p if isinstance(old_blk.drop, nn.Dropout2d) else 0.0
    new_blk = GroupResBlock(new_C, new_groups, drop=drop_p).to(old_blk.beta.device)

    # GroupNorm: (C,) -> 选通道
    new_blk.norm.weight.copy_(old_blk.norm.weight[keep_idx])
    new_blk.norm.bias.copy_(old_blk.norm.bias[keep_idx])

    # 两个 group conv：weight 形状 (C, group_size, k, k)，保留 out-channel 即可
    for i in [0, 1]:
        new_blk.conv[i].weight.copy_(old_blk.conv[i].weight[keep_idx])
        if old_blk.conv[i].bias is not None:
            new_blk.conv[i].bias.copy_(old_blk.conv[i].bias[keep_idx])

    # SCA 里的 1x1 group conv 在 sca[1]
    new_blk.sca[1].weight.copy_(old_blk.sca[1].weight[keep_idx])
    if old_blk.sca[1].bias is not None:
        new_blk.sca[1].bias.copy_(old_blk.sca[1].bias[keep_idx])

    # beta: (1,C,1,1) -> 选通道
    new_blk.beta.copy_(old_blk.beta[:, keep_idx])

    return new_blk

@torch.no_grad()
def prune_stage_bottleneck(seq: nn.Sequential, last_conv: nn.Conv2d, Keep_groups, orig_groups: int, his_ch: int):
    """
    seq: 形如 Conv1x1 -> (GroupResBlock...)-> Conv1x1
    Keep_groups: 你算出来要保留的 group id 集合
    orig_groups: 你原来设的 groups=32（用来算 group_size）
    返回：new_seq, keep_idx, Keep_groups_sorted
    """
    assert isinstance(seq, nn.Sequential)
    assert isinstance(last_conv, nn.Conv2d)

    pre = seq[0]
    post = last_conv
    assert isinstance(pre, nn.Conv2d) and pre.kernel_size == (1,1)
    assert isinstance(post, nn.Conv2d) and post.kernel_size == (1,1)

    # 原通道
    ch_in = pre.in_channels
    ch_mid = pre.out_channels   # 你原来是 ch_mid == ch_in == ch_out
    ch_out = post.out_channels

    # group_size 按原 groups=32 来划分
    assert ch_mid % orig_groups == 0
    group_size = ch_mid // orig_groups

    # 需要保留的通道 idx
    keep_idx, Keep_groups_sorted = groups_to_keep_idx(
        Keep_groups, group_size, device=pre.weight.device
    )
    new_groups = len(Keep_groups_sorted)
    new_C = keep_idx.numel()
    assert new_groups > 0, "Keep_groups 不能为空"
    assert new_C % new_groups == 0

    # 1) new pre: ch_in -> new_C（剪输出通道）
    new_pre = nn.Conv2d(ch_in, new_C, 1, 1, 0, bias=(pre.bias is not None)).to(pre.weight.device)
    new_pre.weight.copy_(pre.weight[keep_idx])                 # 选 out-channel
    if pre.bias is not None:
        new_pre.bias.copy_(pre.bias[keep_idx])

    # 2) new blocks：全部用 new_C/new_groups
    new_blocks = []
    for m in seq:
        if isinstance(m, GroupResBlock):
            new_blocks.append(prune_groupresblock(m, keep_idx, new_C, new_groups))

    # 3) new post: new_C -> ch_out（剪输入通道）
    keep_idx_cur = keep_idx
    if his_ch > 0:
        his_idx = torch.arange(0, his_ch, device=keep_idx.device)
        keep_idx_post = torch.cat([his_idx, keep_idx_cur + his_ch], dim=0)
    else:
        keep_idx_post = keep_idx_cur

    new_post = nn.Conv2d(keep_idx_post.numel(), ch_out, 1, 1, 0, bias=...)
    new_post.weight.copy_(post.weight[:, keep_idx_post])
    if post.bias is not None:
        new_post.bias.copy_(post.bias)  # 输出通道不变，bias 全保留

    new_seq = nn.Sequential(new_pre, *new_blocks)
    return new_seq, new_post, keep_idx, Keep_groups_sorted

@torch.no_grad()
def apply_pruning(model, s_cmtys, ogs, his_Keepch):
    meta=[]

    model.e1, model.e1_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.e1, model.e1_last, s_cmtys[0], ogs[0], his_Keepch[0])
    print(f"e1 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    model.e2, model.e2_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.e2, model.e2_last, s_cmtys[1], ogs[1], his_Keepch[1])
    print(f"e2 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    model.e3, model.e3_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.e3, model.e3_last, s_cmtys[2], ogs[2], his_Keepch[2])
    print(f"e3 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    model.d1, model.d1_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.d1, model.d1_last, s_cmtys[3], ogs[3], his_Keepch[3])
    print(f"d1 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    model.d2, model.d2_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.d2, model.d2_last, s_cmtys[4], ogs[4], his_Keepch[4])
    print(f"d2 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    model.d3, model.d3_last, keep_idx, Keep_groups = prune_stage_bottleneck(model.d3, model.d3_last, s_cmtys[5], ogs[5], his_Keepch[5])
    print(f"d3 Pruned : keep_ch ({keep_idx.numel()}), Keep_groups ({len(Keep_groups)})")
    meta.append(keep_idx.numel())

    return model, meta



def Pruning(args, model, deg, p=0.1, N = 300):
    args.simples = N
    val_dataset = PuningDataset(args.dataset, degarde=deg, simple_num=args.simples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    SEPruner = SE()
    s_cmtys = [[] for _ in range(6)]
    with torch.no_grad():
        for lIr,  name in tqdm(val_loader, desc="pruning"):
            lIr = lIr.to(args.device)

            outs = model.forward_purning(lIr)

            for ind,(tag,h) in enumerate(outs.items()):
                div, en  = SEPruner(h.squeeze(), args.modelHps[4][ind])

                divs_dict = defaultdict(list)
                for i, v in enumerate(div):
                    divs_dict[v] += [i]

                cmtys = list(divs_dict.values())
                for cmty in cmtys:
                    best_g = max(cmty, key=lambda g: en[g])
                    s_cmtys[ind].append(best_g)

    KeepGPs = [set() for _ in range(6)]
    for i, cmtys in enumerate(s_cmtys):
        count_dict = Counter(cmtys)
        for g, count in count_dict.items():
            if count >= args.simples * p:
                KeepGPs[i].add(g)



    model, meta = apply_pruning(model, KeepGPs, ogs=args.modelHps[4], his_Keepch=[0,0,0,0,0,0])

    return model, meta







