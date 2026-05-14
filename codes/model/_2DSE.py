from collections import defaultdict
from typing import List, Any, Tuple, Dict
import math
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F

class SE(nn.Module):
    def __init__(self):
        super(SE, self).__init__()
        self.g_vol = None
        self.adj: List[Dict[int, float]] = []
        self.node_idx = None
        self.node_deg = None
        self.div_cut = None
        self.div_vol = None
        self.node_in = None
        self.nodes = None
    @staticmethod
    def group_adj(h: torch.Tensor, groups: int):

        assert h.dim() == 2
        C, W = h.shape
        assert C % groups == 0
        gs = C // groups
        x = h.view(groups, gs, W)
        y = F.normalize(x, dim=2, eps=1e-8)  # (G, gs, W)

        return x, torch.matmul(y, y.transpose(1, 2))

    def __init__v(self, features, groups):
        fs = torch.flatten(features, start_dim=1)
        if groups == fs.shape[0]:
            gfs = fs
        else:
            C, W = fs.shape
            gs = C // groups
            fs = fs.view(groups, gs, W)

            gfs = []
            for i in range(fs.shape[0]):
                f = fs[i]
                Ai = torch.corrcoef(f)
                Ai.fill_diagonal_(0)
                Ai[Ai < 0.] = 1e-8
                vol = torch.sum(Ai)
                d = torch.sum(Ai, dim=1)
                _1dse = -(d / vol) * torch.log(d / vol)
                w = F.softmax(_1dse,dim=0)
                gfs.append((w.unsqueeze(1) * f).sum(dim=0))

            gfs = torch.stack(gfs,dim=0)

        self.node_num = groups
        self.nodes = [i for i in range(groups)]

        A = torch.corrcoef(gfs)
        A.fill_diagonal_(0)
        A[A < 0.] = 0.
        A_np = A.detach().cpu().numpy()
        self.adj = [
            {int(j): float(A_np[i, j]) for j in A_np[i].nonzero()[0]}
            for i in range(A_np.shape[0])
        ]

        self.g_vol =  A_np.sum()
        row_sum_np = A_np.sum(axis=1).tolist()
        self.node_deg = row_sum_np.copy()
        self.div_vol = row_sum_np.copy()
        self.div_cut = row_sum_np.copy()
        self.node_in = [0.]*self.node_num
        self.node_div = [i for i in range(self.node_num)]
        self.entropy = [0.]*self.node_num



    def delta_leave(self, i: int, d: int, i_in: float = 0) -> float:
        if math.fabs(self.div_vol[d] - self.node_deg[i]) < 1e-8 and i_in == 0.:
            return 0.

        g_vol = self.g_vol
        i_deg = self.node_deg[i]

        if i_in > 0:
            div_vol = self.div_vol[d] + i_deg
            div_vol_prm = self.div_vol[d]

            div_cut = self.div_cut[d] - 2 * i_in + i_deg
            div_cut_prm = self.div_cut[d]
        else:
            div_vol = self.div_vol[d]
            div_vol_prm = div_vol - i_deg

            div_cut = self.div_cut[d]
            div_cut_prm = div_cut + 2 * self.node_in[i] - i_deg

        try:
            div_prm_se = - (div_cut_prm / g_vol) * math.log(div_vol_prm / g_vol)
        except ValueError as e:
            print(e)
            div_prm_se = 0.

        try:
            div_se = (div_cut / g_vol) * math.log(div_vol / g_vol)
        except ValueError as e:
            print(e)
            div_se = 0.
        try:
            str_se = (div_vol_prm / g_vol) * math.log(div_vol_prm / div_vol)
        except ValueError as e:
            print(e)
            str_se = 0.

        i_se = (i_deg / g_vol) * math.log(g_vol / div_vol)

        return div_prm_se + div_se + str_se + i_se

    def strategy(self, i: int):
        src_d = self.node_div[i]
        tgt_d = src_d

        delta_l = self.delta_leave(i, src_d)
        self.entropy[i] = delta_l

        if delta_l < 0:
            tgt_d = -1

        adj_div = {}
        for j, w in self.adj[i].items():
            div_j = self.node_div[j]
            if div_j != src_d:
                adj_div[div_j] = adj_div.get(div_j, 0) + w

        i_in = 0
        delta_min = 0
        for k, ii in adj_div.items():
            delta_lk = self.delta_leave(i, k, ii)
            delta_tk = delta_l - delta_lk

            if delta_tk < delta_min:
                delta_min = delta_tk
                tgt_d = k
                i_in = ii

        return tgt_d, i_in, delta_min

    def _leave(self, i):
        # update clust C_k
        d = self.node_div[i]
        self.div_vol[d] -= self.node_deg[i]
        self.div_cut[d] = self.div_cut[d] + 2 * self.node_in[i] - self.node_deg[i]

        # new cluster {x}
        self.node_div[i] = i
        self.div_vol[i] = self.node_deg[i]
        self.div_cut[i] = self.div_vol[i]
        self.node_in[i] = 0

    def _transfer(self, i, tgt_d, i_in):
        # update clust C_src
        src_d = self.node_div[i]
        self.div_vol[src_d] -= self.node_deg[i]
        if self.div_vol[src_d] > 0:
            self.div_cut[src_d] = self.div_cut[src_d] + 2 * self.node_in[i] - self.node_deg[i]
        else:
            self.div_cut[src_d] = 0

        # update clust C_src
        self.node_div[i] = tgt_d
        self.div_vol[tgt_d] += self.node_deg[i]
        self.div_cut[tgt_d] = self.div_cut[tgt_d] - 2 * i_in + self.node_deg[i]
        self.node_in[i] = i_in

        for j, w in self.adj[i].items():
            if self.node_div[j] == src_d:
                self.node_in[j] -= w
            if self.node_div[j] == tgt_d:
                self.node_in[j] += w
    def forward(self, x, groups, max_iter=10):
        self.__init__()
        self.__init__v(x, groups)
        for it in range(max_iter):

            delta_sum = 0
            leave, transfer = 0, 0

            for i in self.nodes:
                if self.node_deg[i] == 0.:
                    continue
                tgt_div, cut_i, se_delta = self.strategy(i)

                if se_delta < 0:
                    if tgt_div < 0:
                        # strategy 1
                        self._leave(i)
                        leave += 1
                    else:
                        # strategy 2
                        self._transfer(i, tgt_div, cut_i)
                        transfer += 1

                    delta_sum += se_delta

            if leave + transfer == 0:
                break



        return self.node_div, self.entropy



# def make_tensors(n=6, shape=(16, 256, 256), target_corr=0.05, device=None, dtype=torch.float32):
#     """
#     生成 n 个形状为 `shape` 的张量，放在 device 上，
#     两两 Pearson 相关系数 ≈ target_corr（∈[0, 0.9)）。
#     思路：每个张量 = 独立噪声 + alpha * 公共噪声
#     其中 corr ≈ alpha^2 / (1 + alpha^2)，故 alpha = sqrt(r/(1-r))
#     """
#     assert 0 <= target_corr < 0.9, "target_corr 必须在 [0, 0.9) 内"
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # 根据期望相关系数反推 alpha
#     alpha = math.sqrt(target_corr / (1.0 - target_corr)) if target_corr > 0 else 0.0
#
#     # 公共噪声
#     common = torch.randn(shape, device=device, dtype=dtype)
#
#     # 独立噪声 + 公共噪声
#     xs = []
#     for _ in range(n):
#         indep = torch.randn(shape, device=device, dtype=dtype)
#         x = indep + alpha * common
#         xs.append(x)
#
#     return xs  # 保持在 GPU 上（若可用）
#
# def corr_matrix(tensors):
#     """计算一组（已在同一 device 上）的两两 Pearson 相关系数矩阵。"""
#     vecs = []
#     for t in tensors:
#         v = t.flatten()
#         v = v - v.mean()
#         v = v / (v.std(unbiased=False) + 1e-12)  # 标准化，避免数值问题
#         vecs.append(v)
#     M = torch.stack(vecs)  # [n, N]
#     # 相关矩阵 = 标准化后向量点积 / N
#     N = M.shape[1]
#     C = (M @ M.T) / N
#     return C
# if __name__ == '__main__':
#
#     xs = make_tensors(n=6, shape=(16, 256, 256), target_corr=0.5)
#     xs = torch.stack(xs, dim=0)
#     se = SE()
#     choosen_id, choosen_weight  = se(xs)
#     choose_feats= xs[choosen_id]
#     fa = (choose_feats * choosen_weight).sum(dim=0)
#     print(choosen_id, choosen_weight)


