import sys
import torch

sys.path.insert(0, r'd:\cvs\pythoncode\AnomalyDetection\code')
from src.methods.base import BaseZSAD

ok = True


def chk(nm, cond):
    global ok
    ok = ok and bool(cond)
    print(('PASS' if cond else 'FAIL'), nm)


x = torch.randn(2, 3, 8, 8)
# 1) 群元素的正逆变换必须精确互逆（否则距离图会错位）
for k in range(8):
    chk('roundtrip k=%d' % k, torch.equal(BaseZSAD._sym_inv(BaseZSAD._sym_fwd(x, k), k), x))
# 2) 8 个群元素必须互不相同（否则对称平均退化为重复同一视角）
els = [BaseZSAD._sym_fwd(torch.arange(64.0).reshape(1, 1, 8, 8), k).flatten().tolist()
        for k in range(8)]
chk('8 distinct elements', len({tuple(e) for e in els}) == 8)
# 3) 旋转四次回到原图、旋转+翻转两次回到原图
chk('k=4 is horizontal flip', torch.equal(BaseZSAD._sym_fwd(x, 4), torch.flip(x, dims=(3,))))
chk('k=4 involution', torch.equal(BaseZSAD._sym_fwd(BaseZSAD._sym_fwd(x, 4), 4), x))

# 4) 位置无关的参考统计 + 常数特征图下，对称平均必须与单次打分完全一致
#    （此时 8 个群元素给出同一张距离图，可验证链路与逆变换无残差）
H = W = 6
C = 4
torch.manual_seed(0)
feat = torch.ones(2, C, H, W)
mu0 = torch.randn(C)
prec0 = torch.rand(C) + 0.5
mu = mu0.view(C, 1, 1).expand(C, H, W).contiguous()
prec = prec0.view(1, C).expand(H * W, C).contiguous()
stat = {'mu': mu, 'prec': prec, 'hw': (H, W), 'nbr': 0,
        'nbr_score': 'center',
        'd_mean': torch.full((1, 1, H, W), 0.3),
        'd_std': torch.full((1, 1, H, W), 0.2)}
m = BaseZSAD.__new__(BaseZSAD)
m.sym_avg = 8
for k in (4, 8):
    m.sym_avg = k
    a = m._sym_dist(feat, stat)
    b = m._scale_dist(feat, stat)
    chk('sym_avg=%d constant-feature identity' % k,
        torch.allclose(a, b, atol=1e-5) and a.shape == b.shape)
# 5) 非方形特征图应自动退回单次打分
m.sym_avg = 8
st2 = dict(stat, hw=(6, 5))
st2['mu'] = torch.randn(6, 5, C).permute(2, 0, 1).contiguous()
st2['prec'] = torch.rand(30, C) + 0.5
st2['d_mean'] = torch.rand(6, 5).view(1, 1, 6, 5)
st2['d_std'] = (torch.rand(6, 5) + 0.1).view(1, 1, 6, 5)
f2 = torch.ones(2, C, 6, 5)
chk('non-square fallback', torch.allclose(m._sym_dist(f2, st2), m._scale_dist(f2, st2), atol=1e-6))

print('ALL PASS' if ok else 'SOME FAILED')