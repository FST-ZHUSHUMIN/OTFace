import torch
import torch.nn as nn
import numpy as np


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, C, mu, nu):
        # The Sinkhorn algorithm takes as input three variables :
        u = torch.zeros_like(mu).cuda()
        # print('nu',u.shape)
        v = torch.zeros_like(nu).cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break


        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))
        # print(cost.item())

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return cost, pi, C


    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps
    
    

def sinkhorn_divergence(x, y, distance = 'cosine', device='cuda', critic=None, flat_size=512):
    """
    Compute the Sinkhorn divergence between batches x and y, where x and y are assumed to have the same size,
    using Sinkhorn algorithm.
    :param x, y: two batches of samples (n_samples x d)
    :param distance: transport distance to use ('euclidean' or 'cosine') (str)
    :param parameters: a parser containing the number of iterations for the Sinkhorn algorithm and
    the entropy regularization value
    :param critic: a learnable cost with NN representation. None if fixed L2 cost.
    :param flat_size: flat size of the input if critic is None.
    :return: the Sinkhorn divergence between x and y
    """
    # if critic is not None:
        # x, y = critic(x), critic(y)
    # elif parameters.dataset == 'mnist':
    # x, y = x.view(-1, flat_size), y.view(-1, flat_size)

    # Compute pairwise transport cost matrix
    # print(x.size(), y.size())
    if distance == 'cosine':
        cost = pairwise_cosine_distance(x, y)
    elif distance == 'euclidean':
        cost = torch.cdist(x, y)
    else:
        cost = None

    kernel = torch.exp(- cost / 1)

    n = x.shape[1]
    a = None
    b = torch.ones(n).to(device)
    ones = torch.ones(n).to(device) / n

    for iteration in range(10):
        a = ones / torch.matmul(kernel, b)
        b = ones / torch.matmul(kernel, a)

    p = torch.dot(torch.matmul(kernel, b), a)
    w = torch.dot(torch.matmul(kernel * cost, b), a)

    return w,p


def pairwise_cosine_distance(x, y):
    """
    Compute the pairwise batch cosine distance between two batches x and y.
    :param x, y: batches of samples (n_samples x d)
    :return: the pairwise cosine distance matrix (n_samples, n_samples)
    """
    # L2 normalize batches x and y
    # x_center = x - x.mean(0).unsqueeze(0)
    # y_center = y - y.mean(0).unsqueeze(0)
    # print(x.size(), y.size())
    x_norm = l2_norm(x, axis = 0)
    y_norm = l2_norm(y, axis = 0)
    
    
    # x_norm, y_norm = torch.norm(x, dim=1, keepdim=True), torch.norm(y, dim=1, keepdim=True)
    # print(x_norm.size(), y_norm.size())
    # x, y = x / x_norm, y / y_norm

    # Compute pairwise cosine distance
    # cost = 1 - torch.clamp(torch.matmul(torch.transpose(x_norm, 1, 0), y_norm), -1, 1)
    cost = 1 - torch.matmul(x_norm.transpose(0,1), y_norm)
    # print('cost', cost.size(), cost)
    return cost


def sinkhorn_divergence_numpy(x, y, distance = 'cosine', device='cuda', critic=None, flat_size=512):
    """
    Compute the Sinkhorn divergence between batches x and y, where x and y are assumed to have the same size,
    using Sinkhorn algorithm.
    :param x, y: two batches of samples (n_samples x d)
    :param distance: transport distance to use ('euclidean' or 'cosine') (str)
    :param parameters: a parser containing the number of iterations for the Sinkhorn algorithm and
    the entropy regularization value
    :param critic: a learnable cost with NN representation. None if fixed L2 cost.
    :param flat_size: flat size of the input if critic is None.
    :return: the Sinkhorn divergence between x and y
    """
    # if critic is not None:
        # x, y = critic(x), critic(y)
    # elif parameters.dataset == 'mnist':
    # x, y = x.view(-1, flat_size), y.view(-1, flat_size)

    # Compute pairwise transport cost matrix
    # print(x.size(), y.size())
    if distance == 'cosine':
        cost = numpy_pairwise_cosine_distance(x, y)
    elif distance == 'euclidean':
        cost = torch.cdist(x, y)
    else:
        cost = None

    kernel = np.exp(- cost / 1)

    n = x.shape[1]
    a = None
    b = np.ones(n)
    ones = np.ones(n) / n

    for iteration in range(10):
        a = ones / np.matmul(kernel, b)
        b = ones / np.matmul(kernel, a)

    # p = np.dot(np.matmul(kernel, b), a)
    w = np.dot(np.matmul(kernel * cost, b), a)
    # print(w, cost)
    return w, cost


def numpy_pairwise_cosine_distance(x, y):
    """
    Compute the pairwise batch cosine distance between two batches x and y.
    :param x, y: batches of samples (n_samples x d)
    :return: the pairwise cosine distance matrix (n_samples, n_samples)
    """
    # L2 normalize batches x and y
    # x_center = x - x.mean(0).unsqueeze(0)
    # y_center = y - y.mean(0).unsqueeze(0)
    # print(x.size(), y.size())
    x_norm = numpy_l2_norm(x, axis = 0)
    y_norm = numpy_l2_norm(y, axis = 0)
    
    
    # x_norm, y_norm = torch.norm(x, dim=1, keepdim=True), torch.norm(y, dim=1, keepdim=True)
    # print(x_norm.size(), y_norm.size())
    # x, y = x / x_norm, y / y_norm

    # Compute pairwise cosine distance
    # cost = 1 - torch.clamp(torch.matmul(torch.transpose(x_norm, 1, 0), y_norm), -1, 1)
    # print(np.transpose(x_norm).shape, y_norm.shape)
    cost = 1 - np.matmul(np.transpose(x_norm), y_norm)
    # cost = np.mean(cost)
    # print('cost', cost.shape, cost)
    return cost


def numpy_l2_norm(input, axis = 0):
    norm = np.linalg.norm(input,2, axis = 0)
    # print(norm.shape, input.shape)
    output = np.divide(input,norm)
    test = np.sum(output*output, axis= 0)
    # print(test)
    # print(output)
    # norm = torch.norm(input, 2, axis, True)
    # output = torch.div(input, norm)
    return output