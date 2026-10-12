"""零样本工业异常检测基类：冻结多尺度视觉特征 + 正常参考统计 + 频域/自适应增强。

方法总览
--------
1. 图像 -> 冻结 MobileCLIP-S1 视觉塔 (FastViT) -> {P3, P4, P5} 多尺度 patch 特征（真实多分辨率）
2. 打分：每个尺度用「正常训练图」patch 特征估计高斯参考（均值 + 收缩协方差），
   异常分数 = 马氏距离（相对正常分布的偏离）。严格零样本：训练只见正常图，无任何异常/掩码。
3. 融合：固定均值 / 单尺度 / ADA 自适应加权（按各尺度高频能量占比）。
4. FGSC：Haar 纹理能量引导的分数空间校准，抑制纹理误报。
5. DPA：按纹理复杂度路由的校准强度（L0/L1/L2），实现领域自适应的分数后处理。
6. 图像级分数 = top-k 像素均值。

消融开关（cfg['method']）：
    fusion : 'single' | 'mean' | 'ada'
    fgsc   : 'off' | 'conditional' | 'global'
    dpa    : True / False
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.ada import AdaptiveScaleWeight, fuse_scales, high_freq_ratio
from ..modules.dpa import route_level
from ..modules.fgsc import calibrate, image_score
from ..modules.wavelet import texture_confidence
from ..utils.common import count_params


class BaseZSAD(nn.Module):
    def __init__(
        self,
        backbone,
        text_encoder=None,
        adapters=None,
        cfg: dict | None = None,
        device: torch.device | str = "cuda",
    ):
        super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.backbone = backbone
        self.text_encoder = text_encoder  # 保留（供后续文本引导扩展），当前像素打分不依赖
        self.adapters = adapters

        self.device = torch.device(device) if isinstance(device, str) else device
        self.img_size = int(cfg.get("img_size", 256))
        self.map_size = int(cfg.get("map_size", 256))

        self.scale_names = tuple(cfg.get("scale_names", ("P3", "P4", "P5")))
        self.fusion = cfg.get("fusion", "single")
        self.single_scale = cfg.get("single_scale", "P3")
        self.fgsc_mode = cfg.get("fgsc", "off")
        self.lam = float(cfg.get("fgsc_lambda", 0.8))
        self.tau = float(cfg.get("fgsc_tau", 0.4))
        self.fgsc_kernel = int(cfg.get("fgsc_kernel", 33))
        self.tex_win = int(cfg.get("texture_win", 9))
        self.use_dpa = bool(cfg.get("dpa", False))
        self.dpa_thresholds = tuple(cfg.get("dpa_thresholds", (0.25, 0.45)))
        # DPA：不同纹理复杂度下的校准强度倍率（L0 通用 / L1 表面 / L2 类别）
        self.dpa_gain = tuple(cfg.get("dpa_gain", (1.0, 0.8, 0.6)))
        self.topk = float(cfg.get("image_score_topk", 0.4))
        self.shrinkage = float(cfg.get("shrinkage", 0.1))
        self.ref_smooth = int(cfg.get("ref_smooth", 0))
        # PaDiM 式邻域聚合半径 r：r>0 时对每个位置聚合 (2r+1)^2 邻域内的特征再估计
        # 参考统计（replication）。0 为关闭（保持原逐位置行为）。
        self.nbr_agg = int(cfg.get("nbr_agg", 0))
        # 邻域聚合后的打分口径：center=只用中心特征（PaDiM 原文做法，避免与 PSC 低通
        # 重复平滑）；mean=对 (2r+1)^2 个偏移的马氏距离取均值（等价于再做一次盒式平滑）。
        self.nbr_score = str(cfg.get("nbr_score", "center")).lower()
        # 邻域聚合之外再叠加一次参考均值的空域平滑（0=不再叠加；盒式聚合通常已足够）
        self.nbr_ref_smooth = int(cfg.get("nbr_ref_smooth", 0))
        # 测试期对称平均：1=关闭，4=四旋转，8=二面体群(4 旋转 x 2 翻转)
        self.sym_avg = int(cfg.get("sym_avg", 1))
        self.cov_mode = str(cfg.get("cov_mode", "full")).lower()
        self.use_knn = bool(cfg.get("knn", False))
        self.knn_k = int(cfg.get("knn_k", 3))
        self.bank_size = int(cfg.get("bank_size", 0))
        self.image_blur_k = int(cfg.get("image_blur_k", 11))
        self.image_area_t = float(cfg.get("image_area_t", 0.0))
        self.use_global_img = bool(cfg.get("global_img", True))

        self.ada_weight = AdaptiveScaleWeight(
            num_scales=len(self.scale_names),
            temperature=float(cfg.get("ada_temperature", 4.0)),
            learnable=bool(cfg.get("ada_learnable", True)),
        )

        self.category: Optional[str] = None
        # 正常参考统计（fit 后填充）：per-scale mean/precision + 距离归一化统计
        self._stats: dict[str, dict] = {}
        self._fitted = False
        self.to(self.device)

    def set_category(self, category: str) -> None:
        self.category = category

    # ---------------- 特征 ----------------
    def extract(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(image)

    # ---------------- 正常参考统计（fit） ----------------
    @torch.no_grad()
    def fit(self, loader=None, epochs: int = 0, lr: float = 1e-3, logger=None,
            max_samples: int = 0, feats: dict | None = None, keep_device: bool = False) -> None:
        """逐位置(PaDiM式)高斯正常参考：对每个尺度、每个空间位置，用正常训练图估计
        该位置的均值与收缩协方差；打分时按各自位置的分布做马氏距离并按位置内的正常
        距离分布标准化，从而抑制"本身方差大"的正常位置产生假阳性。无梯度、无掩码。

        feats 非空时跳过 backbone 前向，直接用给定的特征字典（各尺度 [N,C,H,W]，
        可选 "GLOBAL" [N,C']）估计参考统计。骨干是冻结的，故与走loader 完全等价，
        供消融实验复用同一份特征、只重算统计量（避免重复前向）。
        keep_device=True 时统计量在特征所在设备（GPU）上计算：全协方差的
        [HW,C,C] 求逆在 GPU 上比 CPU 快一个量级，是快速消融的关键。
        """
        feats_buf: dict[str, list] = {n: [] for n in self.scale_names}
        gbuf: list = []
        if feats is not None:
            for n in self.scale_names:
                if n in feats:
                    x = feats[n].detach().float()
                    feats_buf[n].append(x if keep_device else x.cpu())
            if self.use_global_img and "GLOBAL" in feats:
                g = feats["GLOBAL"].detach().float()
                gbuf.append(g if keep_device else g.cpu())
        else:
            for batch in loader:
                img = batch["image"].to(self.device)
                fb = self.extract(img)
                for n in self.scale_names:
                    if n in fb:
                        feats_buf[n].append(fb[n].detach().float().cpu())
                if self.use_global_img and "GLOBAL" in fb:
                    gbuf.append(fb["GLOBAL"].detach().float().cpu())
        for n in self.scale_names:
            if not feats_buf[n]:
                continue
            X = torch.cat(feats_buf[n], dim=0)  # [N,C,H,W]
            if max_samples and X.shape[0] > max_samples:
                X = X[torch.randperm(X.shape[0])[:max_samples]]
            N, C, H, W = X.shape
            r = self.nbr_agg
            if r > 0:
                # ---- PaDiM 式邻域聚合(replication)：统计量在 (2r+1)^2 邻域内聚合后再估计 ----
                # 偏移集合是连续方框，故"对偏移求和"严格等价于对逐位置统计图做盒式滤波，
                # 无需显式展开 O 份特征：对角协方差只需逐维一/二阶矩，全协方差需逐位置
                # 二阶矩 [HW,C,C]（与原全协方差分支同阶，不会额外放大内存）。
                # 注意：聚合本身即位置空间平滑，故不再叠加 ref_smooth（避免重复平滑）。
                assert not self.use_knn, "nbr_agg 与 kNN 记忆库分支互斥"
                k = 2 * r + 1
                O = k * k
                Xp = X.permute(2, 3, 0, 1).reshape(H * W, N, C)  # [HW,N,C]
                mu0 = Xp.mean(dim=1)                              # [HW,C] 逐位置样本均值
                mub = self._box_last2(mu0.reshape(H, W, C).permute(2, 0, 1).contiguous(), r)  # [C,H,W]
                mu = mub.reshape(C, H * W).t().contiguous()        # [HW,C]
                if self.nbr_ref_smooth > 1:
                    mu = self._spatial_blur(
                        mu.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0),
                        self.nbr_ref_smooth).squeeze(0).permute(1, 2, 0).reshape(H * W, C)
                score_mean = self.nbr_score == "mean"
                if self.cov_mode == "diag":
                    # 对角：邻域二阶矩 = 盒式聚合(逐位置逐维二阶矩)
                    q0 = (Xp * Xp).mean(dim=1)                     # [HW,C]
                    qb = self._box_last2(q0.reshape(H, W, C).permute(2, 0, 1).contiguous(), r)
                    var = (qb.reshape(C, H * W).t().contiguous() - mu * mu).clamp(min=0)
                    gvar = var.mean().clamp(min=1e-6)
                    var = (1 - self.shrinkage) * var + self.shrinkage * gvar + 1e-6
                    prec = 1.0 / var
                    Xc = Xp - mu[:, None, :]
                    if score_mean:
                        prec_map = prec.reshape(H, W, C).permute(2, 0, 1).contiguous()
                        d2 = torch.empty(H, W, N)
                        for i in range(N):
                            xb = X[i:i + 1].float()
                            wb = self._nbr_windows(xb, r).view(O, 1, C, H, W)
                            df = wb - mub[None, None]
                            d2[:, :, i:i + 1] = (df * df * prec_map[None, None]).sum(dim=2).mean(dim=0).permute(1, 2, 0)
                        d2 = d2.reshape(H * W, N)
                    else:
                        d2 = (Xc * Xc * prec[:, None, :]).sum(dim=-1)
                    del Xc
                else:
                    # 全协方差（PaDiM 原文口径）：距离只用中心特征，邻域只用于估计 mu/Sigma
                    T2 = torch.einsum("hnc,hnd->hcd", Xp, Xp)       # [HW,C,C] 逐位置二阶矩
                    T2b = self._box_last2(
                        T2.reshape(H * W, C * C).permute(1, 0).reshape(C * C, H, W).contiguous(), r)
                    # _box_last2 已对偏移取均值，故池化二阶矩 = T2b / N（不是 / (N*O)）
                    cov = T2b.reshape(C * C, H * W).t().reshape(H * W, C, C) / N
                    del T2, T2b
                    cov = cov - mu.unsqueeze(2) * mu.unsqueeze(1)
                    cov = 0.5 * (cov + cov.transpose(1, 2))          # 数值对称化
                    eye = torch.eye(C, device=cov.device).expand_as(cov)
                    tr = cov.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)
                    cov = (1 - self.shrinkage) * cov + self.shrinkage * (eye * (tr + 1e-6))
                    prec = torch.linalg.pinv(cov)
                    del cov
                    Xc = Xp - mu[:, None, :]
                    d2 = torch.einsum("hnc,hcd,hnd->hn", Xc, prec, Xc).clamp(min=0)
                    del Xc
                del Xp, mu0
            else:
                Xp = X.permute(2, 3, 0, 1).reshape(H * W, N, C)  # [HW,N,C]
                mu = Xp.mean(dim=1)  # [HW,C]
                # 位置空间平滑参考均值：对小幅错位（纹理相位、物体抖动）鲁棒，
                # 缓解 grid/screw 等错位类正常图假高分。仅平滑 μ（方向 Σ 不变）。
                if self.ref_smooth > 1:
                    mu_map = mu.reshape(H, W, C).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
                    mu_map = self._spatial_blur(mu_map, self.ref_smooth)
                    mu = mu_map.squeeze(0).permute(1, 2, 0).reshape(H * W, C)
                Xc = Xp - mu[:, None, :]  # [HW,N,C]
                denom = max(N - 1, 1)
                if self.cov_mode == "diag":
                    # 对角协方差：仅各维方差，对高维/小样本类稳健（N 仅需 >=2）
                    var = (Xc ** 2).mean(dim=1)  # [HW,C]
                    gvar = var.mean().clamp(min=1e-6)
                    var = (1 - self.shrinkage) * var + self.shrinkage * gvar + 1e-6
                    prec = 1.0 / var  # [HW,C] 对角逆方差（fit 内保持 CPU，与全协方差分支一致）
                    d2 = (Xc * Xc * prec[:, None, :]).sum(dim=-1)  # [HW,N]
                else:
                    cov = torch.einsum("hnc,hnd->hcd", Xc, Xc) / denom  # [HW,C,C]
                    # 收缩正则化：向 C*缩放的单位阵收缩，保证数值稳定
                    eye = torch.eye(C, device=cov.device).expand_as(cov)
                    tr = cov.diagonal(dim1=1, dim2=2).mean(dim=1).view(-1, 1, 1)
                    cov = (1 - self.shrinkage) * cov + self.shrinkage * (eye * (tr + 1e-6))
                    prec = torch.linalg.pinv(cov)  # [HW,C,C]
                    d2 = torch.einsum("hnc,hcd,hnd->hn", Xc, prec, Xc).clamp(min=0)  # [HW,N]
            d = d2.sqrt()
            self._stats[n] = {
                "mu": mu.reshape(H, W, C).permute(2, 0, 1).contiguous().to(self.device),  # [C,H,W]
                "prec": prec.to(self.device),  # [HW,C,C] 或 [HW,C]
                "d_mean": d.mean(dim=1).view(1, 1, H, W).to(self.device),
                "d_std": d.std(dim=1).clamp(min=1e-6).view(1, 1, H, W).to(self.device),
                "hw": (H, W),
                "nbr": self.nbr_agg,
                "nbr_score": self.nbr_score,
            }
            if self.use_knn:
                # PatchCore 式位置无关记忆库：存全部(或抽样)正常 patch 特征(L2 归一化)，
                # 标准化统计用"训练 patch 到记忆库的 k 近邻距离"估计(剔除自身)。
                patches = F.normalize(X.permute(0, 2, 3, 1).reshape(-1, C).float(), dim=1)
                if self.bank_size and patches.shape[0] > self.bank_size:
                    patches = patches[torch.randperm(patches.shape[0])[: self.bank_size]]
                chunk = 4096
                ds = []
                for s in range(0, patches.shape[0], chunk):
                    q = patches[s: s + chunk]
                    dd = 1.0 - (q @ patches.t())  # [q, M] 余弦距离
                    kk = min(self.knn_k + 1, dd.shape[1])
                    vals = torch.topk(dd, kk, dim=1, largest=False).values[:, 1:]  # 去掉自身(0)
                    ds.append(vals.mean(dim=1))
                nd_all = torch.cat(ds)
                self._stats[n]["bank"] = patches.to(self.device)
                self._stats[n]["d_mean"] = nd_all.mean().view(1, 1, 1, 1).to(self.device)
                self._stats[n]["d_std"] = nd_all.std().clamp(min=1e-6).view(1, 1, 1, 1).to(self.device)
        # 图像级全局描述子（CLS+register）的高斯参考：仅用正常训练图，无泄漏。
        # 用逐维标准化后的 L2 距离作为全局异常分数，再按训练分布标准化。
        if self.use_global_img and gbuf:
            G = torch.cat(gbuf, dim=0)  # [N, C']
            gmu = G.mean(dim=0)
            gstd = G.std(dim=0).clamp(min=1e-6)
            z = (G - gmu) / gstd
            gs = z.norm(dim=1)
            self._gstats = {
                "mu": gmu.to(self.device),
                "std": gstd.to(self.device),
                "s_mean": gs.mean().to(self.device),
                "s_std": gs.std().clamp(min=1e-6).to(self.device),
            }
        self._fitted = True
        if logger:
            logger.info(f"fit done, scales={list(self._stats.keys())}")

    # ---------------- 打分 ----------------
    def _scale_dist(self, feat: torch.Tensor, stat: dict) -> torch.Tensor:
        """逐位置马氏距离并按位置内正常距离标准化 -> [B,1,H,W]。
        支持全协方差 prec[HW,C,C] 与对角协方差 prec[HW,C]。"""
        B, C, H, W = feat.shape
        r = int(stat.get("nbr", 0) or 0)
        if r > 0 and stat["prec"].dim() == 2 and stat.get("nbr_score", "center") == "mean":
            # 对角 + 偏移均值口径：逐偏移算对角马氏距离再对偏移取均值（等价于再盒式平滑一次）
            O = (2 * r + 1) ** 2
            mu_map = stat["mu"]                                    # [C,H,W]
            prec_map = stat["prec"].reshape(H * W, C).permute(1, 0).reshape(C, H, W)
            wb = self._nbr_windows(feat.float(), r).view(O, B, C, H, W)  # [O,B,C,H,W]
            df = wb - mu_map[None, None]
            d2 = (df * df * prec_map[None, None]).sum(dim=2).mean(dim=0)  # [B,H,W]
            d = d2.clamp(min=0).sqrt().unsqueeze(1)                # [B,1,H,W]
            return (d - stat["d_mean"]) / stat["d_std"]
        x = feat.permute(2, 3, 0, 1).reshape(H * W, B, C).float()  # [HW,B,C]
        x = x - stat["mu"].permute(1, 2, 0).reshape(H * W, C)[:, None, :]  # [HW,B,C]
        prec = stat["prec"]
        if prec.dim() == 3:
            tmp = torch.einsum("hbc,hcd->hbd", x, prec)  # [HW,B,C]
            d2 = (tmp * x).sum(dim=-1).clamp(min=0)  # [HW,B]
        else:  # 对角：prec[HW,C]
            d2 = (x * x * prec[:, None, :]).sum(dim=-1).clamp(min=0)  # [HW,B]
        d = d2.sqrt().reshape(H, W, B).permute(2, 0, 1).unsqueeze(1)  # [B,1,H,W]
        return (d - stat["d_mean"]) / stat["d_std"]

    def _knn_dist(self, feat: torch.Tensor, stat: dict) -> torch.Tensor:
        """位置无关的 kNN 记忆库距离（PatchCore 式），对物体错位更鲁棒 -> [B,1,H,W]。

        分数 = 测试 patch 到正常 patch 记忆库的 k 近邻平均余弦距离，再按记忆库内
        正常样本距离的均值/标准差标准化。"""
        B, C, H, W = feat.shape
        x = feat.permute(0, 2, 3, 1).reshape(-1, C).float()  # [P,C]
        x = torch.nn.functional.normalize(x, dim=1)
        bank = stat["bank"]  # [M,C] 已归一化
        d = 1.0 - (x @ bank.transpose(0, 1))  # [P,M] 余弦距离
        k = min(self.knn_k, bank.shape[0])
        nd = torch.topk(d, k, dim=1, largest=False).values.mean(dim=1)  # [P]
        nd = nd.reshape(B, H, W).unsqueeze(1)
        return (nd - stat["d_mean"]) / stat["d_std"]

    @staticmethod
    def _sym_fwd(x: torch.Tensor, k: int) -> torch.Tensor:
        """二面体群元素 k 的正变换：k<4 为 90° 旋转 k 次，k>=4 再叠加水平翻转。"""
        y = torch.rot90(x, k % 4, dims=(2, 3))
        return torch.flip(y, dims=(3,)) if k >= 4 else y

    @staticmethod
    def _sym_inv(x: torch.Tensor, k: int) -> torch.Tensor:
        """_sym_fwd 的逆变换（先撤销翻转，再撤销旋转），保证能精确回到正立坐标系。"""
        if k >= 4:
            x = torch.flip(x, dims=(3,))
        return torch.rot90(x, -(k % 4), dims=(2, 3))

    def _sym_dist(self, feat: torch.Tensor, stat: dict) -> torch.Tensor:
        """对称平均打分：把特征图按二面体群元素变换后打分，再逆变换回正立并平均。

        方形特征图(28x28/14x14/7x7)上 90° 旋转与翻转都是特征的精确重排，不引入任何
        插值误差，因此这等价于在姿态上对参考距离做 k 折集成，能降低物体姿态/位置抖动
        带来的距离波动（screw/grid 等错位敏感类尤为明显）。代价是打分耗时 ×k。
        """
        H, W = stat["hw"]
        if self.sym_avg <= 1 or H != W:
            return self._scale_dist(feat, stat)
        acc = None
        for k in range(self.sym_avg):
            d = self._scale_dist(self._sym_fwd(feat, k).contiguous(), stat)  # 变换坐标系下的距离图
            d = self._sym_inv(d, k)                                          # 逆变换回正立
            acc = d if acc is None else acc + d
        return acc / self.sym_avg

    def _score_maps(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = {}
        for n in self.scale_names:
            if n not in feats or n not in self._stats:
                continue
            if self.use_knn:
                out[n] = self._knn_dist(feats[n], self._stats[n])
            elif self.sym_avg > 1:
                out[n] = self._sym_dist(feats[n], self._stats[n])
            else:
                out[n] = self._scale_dist(feats[n], self._stats[n])
        return out

    def _fuse_maps(self, prob: dict[str, torch.Tensor], feats: dict[str, torch.Tensor]) -> torch.Tensor:
        names = [n for n in self.scale_names if n in prob]
        if not names:
            raise RuntimeError("无可用尺度分数：请先调用 fit()")
        if self.fusion == "single" and self.single_scale in prob:
            out = prob[self.single_scale]
        else:
            ref = prob[names[0]]
            target = ref.shape[-2:]
            if self.fusion == "ada":
                ratios = torch.stack([high_freq_ratio(feats[n]) for n in names], dim=1)
                w = self.ada_weight(ratios)
            else:  # 'mean' / 'concat' -> 均匀
                w = self.ada_weight.uniform(ref.shape[0], ref.device)
            out = fuse_scales(prob, w, target)
        return F.interpolate(out, size=(self.map_size, self.map_size), mode="bilinear", align_corners=False)

    def _to01(self, x: torch.Tensor) -> torch.Tensor:
        """把标准化距离经 sigmoid 压到 [0,1]，作为校准前的概率图。"""
        return torch.sigmoid(x)

    def raw_score(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (异常概率图 [B,1,H,W]∈[0,1], 纹理图 [B,1,H,W])。"""
        b = image.shape[0]
        feats = self.extract(image)
        tex = texture_confidence(self._unnormalize(image), win=self.tex_win, out_size=self._map_feat_size(feats))
        smaps = self._score_maps(feats)
        raw01 = {n: self._to01(smaps[n]) for n in smaps}
        if self.use_dpa:
            comps = tex.reshape(b, -1).mean(dim=1)
            maps = []
            for i in range(b):
                lv = route_level(float(comps[i]), *self.dpa_thresholds)
                gain = self.dpa_gain[{"L0": 0, "L0+L1": 1, "L0+L1+L2": 2}[lv]]
                raw_i = {n: raw01[n][i: i + 1] for n in raw01}
                fused_i = self._fuse_maps(raw_i, {n: feats[n][i: i + 1] for n in feats})
                maps.append(torch.sigmoid(torch.logit(fused_i.clamp(1e-6, 1 - 1e-6)) * gain))
            raw = torch.cat(maps, dim=0)
        else:
            raw = self._fuse_maps(raw01, feats)
        tex = F.interpolate(tex, size=(self.map_size, self.map_size), mode="bilinear", align_corners=False)
        return raw, tex

    def _map_feat_size(self, feats: dict[str, torch.Tensor]) -> tuple[int, int]:
        names = [n for n in self.scale_names if n in feats]
        return feats[names[0]].shape[-2:]

    def _unnormalize(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.backbone.image_mean, device=image.device).view(1, 3, 1, 1)
        std = torch.tensor(self.backbone.image_std, device=image.device).view(1, 3, 1, 1)
        return (image * std + mean).clamp(0, 1)

    # ---------------- 推理 ----------------
    @staticmethod
    def _gaussian_blur(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
        g = torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2))
        g = (g / g.sum()).view(1, 1, 1, k)
        x = F.conv2d(x, g, padding=(0, k // 2))
        x = F.conv2d(x, g.transpose(2, 3), padding=(k // 2, 0))
        return x

    @staticmethod
    def _spatial_blur(x: torch.Tensor, k: int) -> torch.Tensor:
        """逐通道空域高斯模糊，用于平滑正常参考均值 μ（对位置错位鲁棒）。x: [1,C,H,W]。"""
        if k <= 1:
            return x
        C = x.shape[1]
        ax = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
        g = (torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2)) / torch.exp(-(ax ** 2) / (2 * (k / 4.0) ** 2)).sum())
        gw = g.view(1, 1, 1, k).repeat(C, 1, 1, 1)  # [C,1,1,k]
        gh = gw.transpose(2, 3)  # [C,1,k,1]
        x = F.conv2d(x, gw, padding=(0, k // 2), groups=C)
        x = F.conv2d(x, gh, padding=(k // 2, 0), groups=C)
        return x

    @staticmethod
    def _nbr_windows(x: torch.Tensor, r: int) -> torch.Tensor:
        """邻域聚合的平移窗口：x[B,C,H,W] -> [O*B,C,H,W]，O=(2r+1)^2。

        out[o*B+b] = x[b, :, i+dy, j+dx]（边界 replicate 填充），偏移按 dy、dx 双层
        升序展开。fit 与打分共用同一组偏移，保证参考统计与测试特征口径一致。
        """
        B, C, H, W = x.shape
        xp = F.pad(x, (r, r, r, r), mode="replicate")
        outs = [xp[:, :, r + dy: r + dy + H, r + dx: r + dx + W]
                for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        return torch.cat(outs, dim=0)

    @staticmethod
    def _box_last2(t: torch.Tensor, r: int) -> torch.Tensor:
        """对 [M,H,W] 的最后两维做 (2r+1)^2 邻域平均（replicate 边界），形状不变。

        偏移集合为连续方框，故"对 (2r+1)^2 个偏移求和/取均值"与本函数严格等价；
        显式 replicate 填充后 avg_pool2d 无 padding，每个窗口都恰好除以 (2r+1)^2，
        边界不会偏置。分块处理以限制 [C*C,H,W] 这类大张量的临时内存。
        """
        if r <= 0:
            return t
        k = 2 * r + 1
        M, H, W = t.shape
        out = torch.empty_like(t)
        step = max(1, 8192 * 1024 // (k * k * H * W))
        for i in range(0, M, step):
            blk = t[i:i + step].unsqueeze(1)                     # [b,1,H,W]
            blk = F.pad(blk, (r, r, r, r), mode="replicate")
            out[i:i + step] = F.avg_pool2d(blk, k, 1).squeeze(1)
        return out

    def predict(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        self.eval()
        raw, tex = self.raw_score(image)
        score = calibrate(raw, tex, lam=self.lam, tau=self.tau, mode=self.fgsc_mode, kernel=self.fgsc_kernel)
        sm = self._gaussian_blur(score, self.image_blur_k)
        iscore = image_score(sm, topk=self.topk)
        # 双线索图像级打分：峰值(top-k) 对居中物体类优；面积占比(超阈值像素比例) 对
        # 小/错位缺陷类(screw)优。取两者 max 兼得，修复错位类图像 AUROC。
        if self.image_area_t > 0:
            amap = F.interpolate(score, size=(self.map_size, self.map_size), mode="bilinear", align_corners=False)
            area = (amap > self.image_area_t).float().reshape(amap.shape[0], -1).mean(dim=1)
            iscore = torch.max(iscore, area)
        return {
            "anomaly_map": score.detach(),
            "raw_map": raw.detach(),
            "texture": tex.detach(),
            "image_score": iscore.detach(),
        }

    def trainable_params(self) -> int:
        return count_params(self)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}(fusion={self.fusion}, fgsc={self.fgsc_mode}(λ={self.lam},τ={self.tau}), "
            f"dpa={self.use_dpa}, trainable={self.trainable_params()})"
        )