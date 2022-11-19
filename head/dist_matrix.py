import torch
import numpy as np
from torch.nn import functional as F
from tqdm import tqdm

def compute_euc_matrix(x):
    """
    Params:
        x: [n(samples), d(512)]
    Return:
        euc_matrix: [n, n]
    """
    n = x.shape[0]
    x = F.normalize(x, dim=-1)
    euc_matrix = torch.zeros((n, n), dtype=torch.float32).to(x.device)

    for i in tqdm(range(0, n)):
         y = x[i:i+1].repeat((n, 1))
         euc_matrix[i] = (x - y).norm(dim=-1)
    return euc_matrix

def compute_ot_matrix(x):
    """
    Params:
        x: [n(samples), c(channels), d(h*w)]
    Return:
        ot_matrix: [n, n]
    """
    n = x.shape[0]
    ot_matrix = torch.zeros((n, n), dtype=torch.float32).to(x.device)

    for i in tqdm(range(0, n)):
        y = x[i:i+1].repeat((n, 1, 1)) # x 的第 i 个元素与所有元素计算距离
        ot_matrix[i] = sinkhorn_divergence(x, y)
    return ot_matrix

def sinkhorn_divergence(x, y, distance='cosine'):
    """
    Params:
        x: [n(samples), c(channels), d(h*w)]
        y: [n, c, d]
    Return:
        ot_dists: [n]
    """
    if distance == 'cosine':
        cost = pairwise_cosine_distance(x, y)
    else:
        cost = None

    kernel = torch.exp(- cost / 1) # [n, d, d]

    n = x.shape[0]
    d = x.shape[-1]
    a = None
    b = torch.ones(d, dtype=torch.float32).cuda(0).unsqueeze(0).repeat((n, 1)) # [n, d]
    ones = torch.ones(d, dtype=torch.float32).cuda(0) / d 

    for iteration in range(10):
        a = ones / torch.einsum('ndi,ni->nd', kernel, b) # [n, d]
        b = ones / torch.einsum('ndi,ni->nd', kernel, a)

    w = torch.einsum('ni,ni->n', torch.einsum('ndi,ni->nd', kernel * cost, b), a)

    return w

def pairwise_cosine_distance(x, y):
    """
    Params:
        x: [n, c, d]
        y: [n, c, d]
    Return:
        cost: [n, d, d]
    """
    x_norm = F.normalize(x, dim=1)
    y_norm = F.normalize(y, dim=1)

    cost = 1.0 - torch.einsum('nik,nil->nkl', x_norm, y_norm)
    return cost


# if __name__ == '__main__':
    # pass
    # x = torch.randn((7000, 512, 49)).cuda(0).to(torch.float32)
    # compute_ot_matrix(x)