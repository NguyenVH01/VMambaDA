import torch
import torch.nn.functional as F

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

def discrepancy_slice_wasserstein(p1, p2, dims = 128):
    p1 = F.softmax(p1)
    p2 = F.softmax(p2)
    s = p1.shape
    if s[1] > 1:
        proj = torch.randn(s[1], dims).cuda()
        proj *= torch.rsqrt(torch.sum(torch.mul(proj, proj), 0, keepdim=True))
        p1 = torch.matmul(p1, proj)
        p2 = torch.matmul(p2, proj)
    p1 = torch.topk(p1, s[0], dim=0)[0]
    p2 = torch.topk(p2, s[0], dim=0)[0]
    dist = p1 - p2
    wdist = torch.mean(torch.mul(dist, dist))

    return wdist