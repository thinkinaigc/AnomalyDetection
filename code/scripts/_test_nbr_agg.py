import sys, torch, torch.nn.functional as F
sys.path.insert(0, r'd:\cvs\pythoncode\AnomalyDetection\code')
from src.methods.base import BaseZSAD
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
B, C, H, W, N, r, sh = 3, 5, 9, 11, 7, 2, 0.1
O = (2 * r + 1) ** 2
ok = True
def chk(nm, a, b, tol):
    global ok
    d = (a.double().cpu() - b.double().cpu()).abs().max().item()
    g = d <= tol
    ok = ok and g
    print(('PASS' if g else 'FAIL'), nm, 'max|diff|=%.3e' % d)
x = torch.randn(B, C, H, W, device=dev)
X = torch.randn(N, C, H, W, device=dev)
xp = F.pad(x, (r, r, r, r), mode='replicate')
ref = torch.cat([xp[:, :, r+dy:r+dy+H, r+dx:r+dx+W] for dy in range(-r, r+1) for dx in range(-r, r+1)], 0)
chk('windows', BaseZSAD._nbr_windows(x, r), ref, 0.0)
chk('r=0 identity', BaseZSAD._nbr_windows(x, 0), x, 0.0)
S = Q = None
for i in range(0, N, 3):
    xb = X[i:i+3].float()
    b = xb.shape[0]
    wb = BaseZSAD._nbr_windows(xb, r).view(O, b, C, H, W)
    s, q = wb.sum(1), (wb * wb).sum(1)
    S = s if S is None else S + s
    Q = q if Q is None else Q + q
M = S / N
V = (Q / N - M * M).clamp(min=0)
mu_map = M.mean(0)
dif = M.permute(2, 3, 1, 0) - mu_map.permute(1, 2, 0)[:, :, :, None]
var_v = (V.permute(2, 3, 1, 0) + dif * dif).mean(-1)
mu_r = torch.zeros(H, W, C, dtype=torch.float64)
var_r = torch.zeros(H, W, C, dtype=torch.float64)
Xd = X.double().cpu()
for i in range(H):
    for j in range(W):
        acc = [Xd[:, :, min(max(i+dy, 0), H-1), min(max(j+dx, 0), W-1)] for dy in range(-r, r+1) for dx in range(-r, r+1)]
        A = torch.stack(acc, 0).reshape(-1, C)
        mu_r[i, j] = A.mean(0)
        var_r[i, j] = A.var(0, unbiased=False)
# 参考实现侧同样施加收缩，口径才一致
var_r_raw = var_r.clone()
g_r = var_r.mean()
var_r = (1 - sh) * var_r + sh * g_r + 1e-6
chk('mu vs naive', mu_map.permute(1, 2, 0), mu_r, 1e-4)
g = var_v.mean().clamp(min=1e-6)
chk('var raw vs naive', var_v, var_r_raw, 1e-4)
chk('var shrunk vs naive', (1 - sh) * var_v + sh * g + 1e-6, var_r, 1e-4)
prec = 1.0 / ((1 - sh) * var_v + sh * g + 1e-6)
pm = prec.reshape(H, W, C).permute(2, 0, 1).contiguous()
wb = BaseZSAD._nbr_windows(x, r).view(O, B, C, H, W)
agg = ((wb - mu_map[None, None])**2 * pm[None, None]).sum(2).mean(0)
xpad = F.pad(x.double(), (r, r, r, r), mode='replicate')
d2n = torch.zeros(B, H, W, device=dev, dtype=torch.float64)
for dy in range(-r, r+1):
    for dx in range(-r, r+1):
        wdw = xpad[:, :, r+dy:r+dy+H, r+dx:r+dx+W]
        d2n = d2n + ((wdw - mu_map.double()[None])**2 * pm.double()[None]).sum(1)
chk('d2 mean-over-offsets', agg.double(), d2n / O, 1e-4)

# ---- _box_last2 与朴素邻域平均等价 ----
T = torch.randn(3, H, W, device=dev)
Tp = F.pad(T, (r, r, r, r), mode='replicate')
Tn = torch.zeros_like(T)
for dy in range(-r, r + 1):
    for dx in range(-r, r + 1):
        Tn += Tp[:, r + dy:r + dy + H, r + dx:r + dx + W]
chk('box_last2', BaseZSAD._box_last2(T, r), Tn / O, 1e-5)
chk('box r=0 identity', BaseZSAD._box_last2(T, 0), T, 0.0)

# ---- 新实现(盒式聚合) vs 朴素偏移循环：均值 / 全协方差 ----
Xp = Xd.permute(2, 3, 0, 1).reshape(H * W, N, C)          # [HW,N,C]
mu0 = Xp.mean(1)                                           # [HW,C] 逐位置样本均值
mu_box = BaseZSAD._box_last2(mu0.reshape(H, W, C).permute(2, 0, 1).contiguous(), r)
mu_pool = mu_box.reshape(C, H * W).t().contiguous()        # 邻域池化均值
mu_ref = torch.zeros(H * W, C, dtype=torch.float64)
for i in range(H):
    for j in range(W):
        acc = [Xd[:, :, min(max(i + dy, 0), H - 1), min(max(j + dx, 0), W - 1)]
               for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        mu_ref[i * W + j] = torch.stack(acc, 0).reshape(-1, C).mean(0)
chk('mu pooled vs naive', mu_pool, mu_ref, 1e-4)
T2 = torch.einsum('hnc,hnd->hcd', Xp, Xp)                   # [HW,C,C]
T2b = BaseZSAD._box_last2(
    T2.reshape(H * W, C * C).permute(1, 0).reshape(C * C, H, W).contiguous(), r)
cov_box = T2b.reshape(C * C, H * W).t().reshape(H * W, C, C) / N \
    - mu_pool.unsqueeze(2) * mu_pool.unsqueeze(1)
cov_box = 0.5 * (cov_box + cov_box.transpose(1, 2))
# 朴素：把 (2r+1)^2*N 个特征堆起来直接求协方差
cov_ref = torch.zeros(H * W, C, C, dtype=torch.float64)
for i in range(H):
    for j in range(W):
        acc = [Xd[:, :, min(max(i + dy, 0), H - 1), min(max(j + dx, 0), W - 1)]
               for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
        A = torch.stack(acc, 0).reshape(-1, C) - mu_ref[i * W + j]
        cov_ref[i * W + j] = A.t() @ A / (N * O)
chk('cov pooled vs naive', cov_box, cov_ref, 1e-4)
print('ALL PASS' if ok else 'SOME FAILED')
