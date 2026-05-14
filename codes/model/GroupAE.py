import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupResBlock(nn.Module):

    def __init__(self, c: int, groups:int, drop: float = 0.0):
        super().__init__()
        assert c % groups == 0

        self.norm = nn.GroupNorm(num_groups=groups, num_channels=c)
        self.conv = nn.Sequential(
            nn.Conv2d(c , c , 3, 1, 1, groups=groups),
            nn.Conv2d(c, c, 3, 1, 1, groups=groups))
        self.act  = nn.GELU()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, 1, 1, 0, groups=groups),
            nn.Sigmoid(),
        )

        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = self.conv(y)
        y = self.act(y)
        y = y * self.sca(y)
        y = self.drop(y)
        return x + y * self.beta


class GroupAE(nn.Module):
    def __init__(
            self,
            in_ch: int = 1,
            base_ch: int = 32,
            num_blocks: list = [2, 4, 8],
            drop: float = 0.0,
            groups: int = [8,16,32,32,16,8],
            BlockInCh: list = [32, 128, 512, 512, 128, 32],
            K:int = 0,
    ):
        super().__init__()
        self.K = K
        self.stem = nn.Conv2d(in_ch, base_ch, 3, 1, 1)

        self.e1 = nn.Sequential(
            nn.Conv2d(base_ch, BlockInCh[0], 1, 1, 0),
            *[GroupResBlock(BlockInCh[0], groups[0], drop) for _ in range(num_blocks[0])],  # DCM1
        )
        self.e1_last = nn.Conv2d(BlockInCh[0] , base_ch, 1, 1, 0)

        self.down1 = nn.PixelUnshuffle(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(base_ch*4, BlockInCh[1], 1, 1, 0),
            *[GroupResBlock(BlockInCh[1], groups[1], drop) for _ in range(num_blocks[1])],  # DCM2
        )
        self.e2_last = nn.Conv2d(BlockInCh[1], base_ch*4, 1, 1, 0)

        self.down2 = nn.PixelUnshuffle(2)
        self.e3 = nn.Sequential(
            nn.Conv2d(base_ch * 16, BlockInCh[2], 1, 1, 0),
            *[GroupResBlock(BlockInCh[2], groups[2], drop) for _ in range(num_blocks[2])],  # DCM3
        )
        self.e3_last = nn.Conv2d(BlockInCh[2], base_ch * 16, 1, 1, 0)

        self.bottom = nn.Conv2d(base_ch * 16, base_ch * 16, 1, 1, 0)

        self.d1 = nn.Sequential(
            nn.Conv2d(base_ch * 16, BlockInCh[3], 1, 1, 0),
            *[GroupResBlock(BlockInCh[3], groups[3], drop) for _ in range(num_blocks[2])],  # DCM4
        )
        self.d1_last = nn.Conv2d(BlockInCh[3], base_ch * 16, 1, 1, 0)

        self.up1 = nn.PixelShuffle(2)
        self.d2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, BlockInCh[4], 1, 1, 0),
            *[GroupResBlock(BlockInCh[4], groups[4], drop) for _ in range(num_blocks[1])],  # DCM5
        )
        self.d2_last = nn.Conv2d(BlockInCh[4], base_ch * 4, 1, 1, 0)

        self.up2 = nn.PixelShuffle(2)
        self.d3 = nn.Sequential(
            nn.Conv2d(base_ch, BlockInCh[5], 1, 1, 0),
            *[GroupResBlock(BlockInCh[5], groups[5], drop) for _ in range(num_blocks[0])],  # DCM6
        )
        self.d3_last = nn.Conv2d(BlockInCh[5], base_ch, 1, 1, 0)

        self.out = nn.Conv2d(base_ch, in_ch, 3, 1, 1)

        self.a = nn.Parameter(torch.ones(1, base_ch , 1, 1))
        self.b = nn.Parameter(torch.ones(1, base_ch * 4, 1, 1))
        self.c = nn.Parameter(torch.ones(1, base_ch  * 16, 1, 1))

        self.sccm = SCCMStage(base_ch, self.K) if self.K > 0 else nn.Identity()
    def forward_purning(self,x: torch.Tensor):
        Block_out = {}
        x = self.stem(x)

        x1 = self.e1(x)
        Block_out["e1_last"] = x1
        x1 = self.e1_last(x1)
        x = self.down1(x1)

        x2 = self.e2(x)
        Block_out["e2_last"] = x2
        x2 = self.e2_last(x2)
        x = self.down2(x2)

        x3 = self.e3(x)
        Block_out["e3_last"] = x3
        x3 = self.e3_last(x3)
        x = self.bottom(x3)

        y1 = self.d1(x)
        Block_out["d1_last"] = y1
        y1 = self.d1_last(y1)
        y = y1 + self.c * x3
        y = self.up1(y)

        y2 = self.d2(y)
        Block_out["d2_last"] = y2
        y2 = self.d2_last(y2)
        y = y2 + self.b * x2
        y = self.up2(y)

        y3 = self.d3(y)
        Block_out["d3_last"] = y3

        return  Block_out

    def forward_stage(self,x: torch.Tensor):
        x = self.stem(x)

        x1 = self.e1(x)
        x1 = self.e1_last(x1)
        x = self.down1(x1)

        x2 = self.e2(x)
        x2 = self.e2_last(x2)
        x = self.down2(x2)

        x3 = self.e3(x)
        x3 = self.e3_last(x3)
        x = self.bottom(x3)

        y1 = self.d1(x)
        y1 = self.d1_last(y1)
        y = y1 + self.c * x3
        y = self.up1(y)

        y2 = self.d2(y)
        y2 = self.d2_last(y2)
        y = y2 + self.b * x2
        y = self.up2(y)

        y3 = self.d3(y)
        y3 = self.d3_last(y3)
        y = y3 + self.a * x1


        return y

    def forward(self, x: torch.Tensor, hs = None) :
        x = self.stem(x)

        x1 = self.e1(x)
        x1 = self.e1_last(x1)
        x = self.down1(x1)

        x2 = self.e2(x)
        x2 = self.e2_last(x2)
        x = self.down2(x2)

        x3 = self.e3(x)
        x3 = self.e3_last(x3)
        x = self.bottom(x3)

        y1 = self.d1(x)
        y1 = self.d1_last(y1)
        y = y1 + self.c * x3
        y = self.up1(y)

        y2 = self.d2(y)
        y2 = self.d2_last(y2)
        y = y2 + self.b * x2
        y = self.up2(y)

        y3 = self.d3(y)
        y3 = self.d3_last(y3)
        y = y3 + self.a * x1

        if hs: y = self.sccm(y, hs)

        z = self.out(y)


        return z




class SCCMStage(nn.Module):

    def __init__(
        self,
        C: int,
        K:int,
        M: int = 32,
        rank: int = 8,
    ):
        super().__init__()
        self.K = K
        self.M = M
        self.rank = rank

        self.norm = nn.ModuleList([nn.GroupNorm(num_groups=8, num_channels=C)
                                  for _ in range(self.K)])
        self.phi = nn.Conv2d(C, M, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi_norm = nn.GroupNorm(num_groups=8, num_channels=C)

        self.psi = nn.Conv2d(K*C, M, kernel_size=1, stride=1, padding=0, bias=True)

        self.hyper =  nn.Sequential(
                nn.Linear(2 * M, 128),
                nn.GELU(),
                nn.Linear(128, 2 * M * rank)  # 输出 P 和 Q
            )

        self.gate = nn.Sequential(
                nn.Linear(2 * M, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -3.0)

        self.P = nn.Parameter(torch.zeros(M, rank))
        self.Q = nn.Parameter(torch.zeros(M, rank))

        self.adp = nn.Conv2d(M, C, kernel_size=1, stride=1, padding=0)
        self.out = nn.Sequential(nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1),
                                 nn.GELU(),
                                 nn.Conv2d(C, C, kernel_size=3, stride=1, padding=1))
        self.bias = nn.Parameter(torch.tensor(-4.0))



    def forward(self, ref: torch.Tensor, feats: list[torch.Tensor]):
        B, C, H, W = ref.shape

        U = self.phi(self.phi_norm(ref))  # (B,M,H,W)
        t_r = U.mean(dim=(2,3))

        feats = [f.detach() for f in feats]
        V = self.psi(torch.cat([self.norm[i](h) for i, h in enumerate(feats)],dim=1))
        t_i = V.mean(dim=(2, 3))

        pq = self.hyper(torch.cat([t_r, t_i], dim=1))
        g = torch.sigmoid(self.gate(torch.cat([t_r, t_i], dim=1))).view(B, 1, 1, 1)

        P = pq[:, :self.M * self.rank].view(B, self.M, self.rank)
        Q = pq[:, self.M * self.rank:].view(B, self.M, self.rank)

        A_dyn = torch.bmm(P, Q.transpose(1, 2))  # low-rank
        A_global = self.P @ self.Q.t()
        A = torch.sigmoid(A_global.unsqueeze(0) + A_dyn + self.bias)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
        deltaU = g * (torch.einsum('bmk,bkxy->bmxy', A, V) - U)

        inj = self.adp(deltaU)
        return self.out(ref + inj)

class IncModel(nn.Module):
    def __init__(
            self,
            in_ch: int = 1,
            base_ch: int = 32,
            num_blocks: list = [2, 4, 8],
            drop: float = 0.0,
            DCMGroups: dict = {},
            BlockInChs: dict ={},
            sub_deg: dict = {}
    ):
        super().__init__()

        self.stem = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.bottom = nn.Conv2d(base_ch * 16, base_ch * 16, 1, 1, 0)
        self.out = nn.Conv2d(base_ch, in_ch, 3, 1, 1)

        self.contrast = DCM(base_ch, num_blocks, drop, groups=DCMGroups["contrast"], BlockInCh=BlockInChs["contrast"], sub_deg=sub_deg["contrast"])
        self.blur = DCM(base_ch, num_blocks, drop, groups=DCMGroups["blur"], BlockInCh=BlockInChs["blur"], sub_deg=sub_deg["blur"])
        self.noise = DCM(base_ch, num_blocks, drop, groups=DCMGroups["noise"], BlockInCh=BlockInChs["noise"], sub_deg=sub_deg["noise"])
        self.CB = DCM(base_ch, num_blocks, drop, groups=DCMGroups["CB"], BlockInCh=BlockInChs["CB"], sub_deg=sub_deg["CB"])
        self.CN = DCM(base_ch, num_blocks, drop, groups=DCMGroups["CN"], BlockInCh=BlockInChs["CN"], sub_deg=sub_deg["CN"])
        self.BN = DCM(base_ch, num_blocks, drop, groups=DCMGroups["BN"], BlockInCh=BlockInChs["BN"], sub_deg=sub_deg["BN"])
        self.CBN = DCM(base_ch, num_blocks, drop, groups=DCMGroups["CBN"], BlockInCh=BlockInChs["CBN"], sub_deg=sub_deg["CBN"])

    def _Deg_(self, x, dcm, hs = None):

        x1 = dcm.e1(x)
        h1 = dcm.e1_last(x1)
        x = dcm.down1(h1)

        x2 = dcm.e2(x)
        h2 = dcm.e2_last(x2)
        x = dcm.down2(h2)

        x3 = dcm.e3(x)
        h3 = dcm.e3_last(x3)

        x = self.bottom(h3)

        y1 = dcm.d1(x)
        h4 = dcm.d1_last(y1)
        y = h4 + dcm.c * h3
        y = dcm.up1(y)

        y2 = dcm.d2(y)
        h5 = dcm.d2_last(y2)
        y = h5 + dcm.b * h2
        y = dcm.up2(y)

        y3 = dcm.d3(y)
        h6 = dcm.d3_last(y3)
        y = h6 + dcm.a * h1

        if hs: y = dcm.sccm(y, hs)

        return y


    def forward(self, x: torch.Tensor, deg = None) :
        x = self.stem(x)
        if deg == "contrast":
            y = self._Deg_(x, self.contrast)
        elif deg == "blur":
            y = self._Deg_(x, self.blur)
        elif deg == "noise":
            y = self._Deg_(x, self.noise)
        elif deg == "CB":
            y1 = self._Deg_(x, self.contrast)
            y2 = self._Deg_(x, self.blur)
            y = self._Deg_(x, self.CB, [y1, y2])
        elif deg == "CN":
            y1 = self._Deg_(x, self.contrast)
            y2 = self._Deg_(x, self.noise)
            y = self._Deg_(x, self.CN, [y1, y2])
        elif deg == "BN":
            y1 = self._Deg_(x, self.blur)
            y2 = self._Deg_(x, self.noise)
            y = self._Deg_(x, self.BN, [y1, y2])
        else:
            y1 = self._Deg_(x, self.contrast)
            y2 = self._Deg_(x, self.blur)
            y3 = self._Deg_(x, self.noise)
            y = self._Deg_(x, self.CBN, [y1, y2, y3])
        z = self.out(y)
        return z

class DCM(nn.Module):
    def __init__(
            self,
            base_ch: int = 32,
            num_blocks: list = [2, 4, 8],
            drop: float = 0.0,
            groups: list = [8,16,32,32,16,8],
            BlockInCh: list = [32, 128, 512, 512, 128, 32],
            sub_deg :int =0
    ):
        super().__init__()

        self.sub_deg = sub_deg

        self.e1 = nn.Sequential(
            nn.Conv2d(base_ch, BlockInCh[0], 1, 1, 0),
            *[GroupResBlock(BlockInCh[0], groups[0], drop) for _ in range(num_blocks[0])],  # DCM1
        )
        self.e1_last = nn.Conv2d(BlockInCh[0] , base_ch, 1, 1, 0)

        self.down1 = nn.PixelUnshuffle(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(base_ch*4, BlockInCh[1], 1, 1, 0),
            *[GroupResBlock(BlockInCh[1], groups[1], drop) for _ in range(num_blocks[1])],  # DCM2
        )
        self.e2_last = nn.Conv2d(BlockInCh[1], base_ch*4, 1, 1, 0)

        self.down2 = nn.PixelUnshuffle(2)
        self.e3 = nn.Sequential(
            nn.Conv2d(base_ch * 16, BlockInCh[2], 1, 1, 0),
            *[GroupResBlock(BlockInCh[2], groups[2], drop) for _ in range(num_blocks[2])],  # DCM3
        )
        self.e3_last = nn.Conv2d(BlockInCh[2], base_ch * 16, 1, 1, 0)

        self.d1 = nn.Sequential(
            nn.Conv2d(base_ch * 16, BlockInCh[3], 1, 1, 0),
            *[GroupResBlock(BlockInCh[3], groups[3], drop) for _ in range(num_blocks[2])],  # DCM4
        )
        self.d1_last = nn.Conv2d(BlockInCh[3], base_ch * 16, 1, 1, 0)

        self.up1 = nn.PixelShuffle(2)
        self.d2 = nn.Sequential(
            nn.Conv2d(base_ch * 4, BlockInCh[4], 1, 1, 0),
            *[GroupResBlock(BlockInCh[4], groups[4], drop) for _ in range(num_blocks[1])],  # DCM5
        )
        self.d2_last = nn.Conv2d(BlockInCh[4], base_ch * 4, 1, 1, 0)

        self.up2 = nn.PixelShuffle(2)
        self.d3 = nn.Sequential(
            nn.Conv2d(base_ch, BlockInCh[5], 1, 1, 0),
            *[GroupResBlock(BlockInCh[5], groups[5], drop) for _ in range(num_blocks[0])],  # DCM6
        )
        self.d3_last = nn.Conv2d(BlockInCh[5], base_ch, 1, 1, 0)

        self.sccm = SCCMStage(base_ch, self.sub_deg) if self.sub_deg > 0 else nn.Identity()

        self.a = nn.Parameter(torch.ones(1, base_ch , 1, 1))
        self.b = nn.Parameter(torch.ones(1, base_ch * 4, 1, 1))
        self.c = nn.Parameter(torch.ones(1, base_ch  * 16, 1, 1))






