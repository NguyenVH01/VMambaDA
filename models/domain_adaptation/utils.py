import torch
import torch.nn.functional as F

def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1)- F.softmax(out2)))

# sliced wasserstein computation use
def discrepancy_slice_wasserstein(p1, p2, dims = 128):
    s = p1.shape
    if p1.shape[1] > 1:
        # For data more than one-dimensional, perform multiple random projection to 1-D
        proj = torch.randn(p1.shape[1], dims).to(p1.device)  # Random projection
        proj *= torch.rsqrt(torch.sum(proj ** 2, dim=0, keepdim=True))  # Normalize projections
        p1 = torch.matmul(p1, proj)  # Project p1
        p2 = torch.matmul(p2, proj)  # Project p2

    # Sort along the rows
    p1, _ = torch.sort(p1, dim=0)
    p2, _ = torch.sort(p2, dim=0)

    # Compute Wasserstein distance
    wdist = torch.mean((p1 - p2) ** 2, dim=0)
    swd = torch.mean(wdist)
    # print(f'SWD: {swd:2f}')
    return swd