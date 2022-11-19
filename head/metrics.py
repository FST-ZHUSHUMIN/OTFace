from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from torch.autograd import Variable

from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
# from .layers import SinkhornDistance
from .Sinkhorn_layer import SinkhornDistance
from .Sinkhorn_layer import sinkhorn_divergence
# Support: []

def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AMsoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30, m=0.35, easy_margin=False):
        super(AMsoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.pair = []

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos_theta: [batchsz, embeddingsz] @ [embeddingsz, nrof_classes] = [batchsz, nrof_classes]
        cos_theta = torch.matmul(embeddings, kernel_norm)
        # for numerical steady
        cos_theta = torch.clamp(cos_theta, -1, 1)
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        phi = cos_theta - self.m
        label_onehot = F.one_hot(label, self.out_features)
        adjust_theta = self.s * torch.where(torch.eq(label_onehot, 1), phi, cos_theta)

        return adjust_theta, origin_cos*self.s

class MV_AMsoftmax_a(nn.Module):
    def __init__(self, in_features, out_features, s=30, m=0.35, easy_margin=False):
        super(MV_AMsoftmax_a, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.pair = []
        self.t = 0.2

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()

        # print(embeddings.size(0), label)
        target_theta = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)
        cos_theta_m = target_theta - self.m

        mask = cos_theta > cos_theta_m
        final_target_logit = cos_theta_m
        hard_example = cos_theta[mask]
        cos_theta[mask] = (self.t+1)*hard_example+self.t
        cos_theta.scatter_(1, label.view(-1, 1).long(), target_theta)
        output = cos_theta * self.s
        return output, origin_cos * self.s


class MV_ArcFace_a(nn.Module):
    def __init__(self, in_features, out_features, m = 0.5, s = 64.):
        super(MV_ArcFace_a, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.kernel = Parameter(torch.Tensor(in_features, out_features))
        self.t = 0.3
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        mask = cos_theta > cos_theta_m
        final_target_logit = torch.where(target_logit > self.threshold, cos_theta_m, target_logit - self.mm)

        hard_example = cos_theta[mask]
        with torch.no_grad():
            self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
        cos_theta[mask] =(self.t+1)*hard_example+self.t
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s


class ArcFace(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.45, easy_margin=False):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m = m

        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, label):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        output = cos_theta * self.s
        return output, origin_cos * self.s
    

class OTFace_Arcface(nn.Module):
    r"""Implement of ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            device_id: the ID of GPU where the model will be trained by model parallel.
                       if device_id=None, it will be trained on CPU without model parallel.
            s: norm of input feature
            m: margin
            cos(theta+m)
        """
    def __init__(self, in_features, out_features, s=64.0, m=0.45, easy_margin=False, alpha = 0, beta = 0):
        super(OTFace_Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        # nn.init.xavier_uniform_(self.kernel)
        nn.init.normal_(self.kernel, std=0.01)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.alpha = alpha
        self.beta = beta
        self.eps = 0.1
        self.niter = 6
        self.c = torch.arange(0, in_features, 1).clone().detach()

    def forward(self, embeddings, conv_features, label, epoch):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embeddings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        with torch.no_grad():
            origin_cos = cos_theta.clone()
        target_logit = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(target_logit > self.th, cos_theta_m, target_logit - self.mm)

        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_logit)
        arcfaceloss = cos_theta * self.s
        #otloss
        label_eq = label.view(1, -1) == label.view(-1, 1)
        label_neg = ~label_eq   
        dists = torch.matmul(embeddings, torch.transpose(embeddings, 1, 0)) # cosdist
        neg_dist = dists.repeat((1, embeddings.shape[0])).view(-1, embeddings.shape[0]) 
        pos_dist = dists.view(-1, 1).repeat((1, embeddings.shape[0]))
        label_neg_teil = label_neg.repeat(1, embeddings.shape[0]).view(-1, embeddings.shape[0])
        label_eq_teil = label_eq.view(-1, 1).repeat(1, embeddings.shape[0])
        diff_low = neg_dist - pos_dist 
        mask = label_eq_teil * label_neg_teil
        loss_matrix_low = torch.where(mask, diff_low, torch.zeros(diff_low.shape).cuda())
        cosine_pair = (loss_matrix_low > -self.alpha).nonzero()
        # wass_size = torch.unique(cosine_pair[:, 0])
        ot_pair = 0
        wass_dis = torch.zeros(size=[1], requires_grad=True).cuda()
        if epoch >= 0:
            # wass_dis = torch.zeros(size=[1], requires_grad=True).cuda()
            for i in range(len(cosine_pair)):
                triplet = cosine_pair[i]
                anchor = triplet[0] // embeddings.shape[0]
                positive = triplet[0] % embeddings.shape[0]
                negative = triplet[1]
                anchor_fea = conv_features[anchor.int(), :, :, :]
                pos_fea = conv_features[positive.int(), :, :, :]
                neg_fea = conv_features[negative.int(), :, :, :]
                anchor_fea = torch.reshape(anchor_fea, (anchor_fea.shape[0], anchor_fea.shape[1] * anchor_fea.shape[1]))
                pos_fea = torch.reshape(pos_fea, (pos_fea.shape[0], pos_fea.shape[1] * pos_fea.shape[1]))
                neg_fea = torch.reshape(neg_fea, (neg_fea.shape[0], neg_fea.shape[1] * neg_fea.shape[1]))
            
                dist_ap, trans_ap = sinkhorn_divergence(anchor_fea, pos_fea)
                dist_an, trans_an = sinkhorn_divergence(anchor_fea, neg_fea)
                dis = max(dist_ap - dist_an, -self.beta)
                if dis>0:
                    wass_dis = wass_dis + dis
                    ot_pair = ot_pair + 1             
        
        if  wass_dis.item() > 0:
            wass_loss = wass_dis/ot_pair

        else:
            wass_loss = torch.zeros([1])

        return arcfaceloss, wass_loss, origin_cos * self.s, len(cosine_pair), ot_pair


def square_distance(features1,features2,squared=False):
    square_features1 = torch.sum(features1 * features1, dim=1)
    square_features2 = torch.sum(features2 * features2, dim=1)
    dot_product = torch.mm(features1, features2.transpose(0, 1))
    square_dists = square_features1.view(-1, 1) - 2 * dot_product + square_features2.view(1, -1)
    return square_dists


def union(X, Y):
    d = {}
    for x in X:
        d[tuple(x.tolist())] = True;
    mask = torch.tensor([d.get(tuple(y.tolist()), False) for y in Y])
    return Y[mask]


class OTFace_AMsoftmax(nn.Module):
    def __init__(self, in_features, out_features, s=30, m=0.35, easy_margin=False):
        super(OTFace_AMsoftmax, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.eps = 0.1
        self.niter = 6
        self.c = torch.arange(0,in_features, 1).clone().detach()
        self.register_buffer('t', torch.zeros(1))

    def forward(self, embeddings, conv_features, label, epoch):
        embeddings = l2_norm(embeddings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        # cos_theta: [batchsz, embeddingsz] @ [embeddingsz, nrof_classes] = [batchsz, nrof_classes]
        cos_theta = torch.matmul(embeddings, kernel_norm)
        cos_theta = torch.clamp(cos_theta, -1, 1)
        with torch.no_grad():
            origin_cos = cos_theta.clone()

        label_onehot = F.one_hot(label, self.out_features)
        # get AM softmax
        target_theta = cos_theta[torch.arange(0, embeddings.size(0)), label].view(-1, 1)
        cos_theta_m = target_theta - self.m
        final_target_theta = cos_theta_m
        cos_theta.scatter_(1, label.view(-1, 1).long(), final_target_theta)
        amloss = cos_theta * self.s

        # get positive pair label
        label_eq = label.view(1, -1) == label.view(-1, 1)
        label_neg = ~label_eq   
        dists = torch.matmul(embeddings, torch.transpose(embeddings, 1, 0)) # cosdist
        neg_dist = dists.repeat((1, embeddings.shape[0])).view(-1, embeddings.shape[0]) 
        pos_dist = dists.view(-1, 1).repeat((1, embeddings.shape[0]))
        label_neg_teil = label_neg.repeat(1, embeddings.shape[0]).view(-1, embeddings.shape[0])
        label_eq_teil = label_eq.view(-1, 1).repeat(1, embeddings.shape[0])
        diff_low = neg_dist - pos_dist 
        mask = label_eq_teil * label_neg_teil
        loss_matrix_low = torch.where(mask, diff_low, torch.zeros(diff_low.shape).cuda())
        cosine_pair = (loss_matrix_low > -self.alpha).nonzero()
        # wass_size = torch.unique(cosine_pair[:, 0])
        ot_pair = 0
        wass_dis = torch.zeros(size=[1], requires_grad=True).cuda()
        if epoch >= 0:
            # wass_dis = torch.zeros(size=[1], requires_grad=True).cuda()
            for i in range(len(cosine_pair)):
                triplet = cosine_pair[i]
                anchor = triplet[0] // embeddings.shape[0]
                positive = triplet[0] % embeddings.shape[0]
                negative = triplet[1]
                anchor_fea = conv_features[anchor.int(), :, :, :]
                pos_fea = conv_features[positive.int(), :, :, :]
                neg_fea = conv_features[negative.int(), :, :, :]
                anchor_fea = torch.reshape(anchor_fea, (anchor_fea.shape[0], anchor_fea.shape[1] * anchor_fea.shape[1]))
                pos_fea = torch.reshape(pos_fea, (pos_fea.shape[0], pos_fea.shape[1] * pos_fea.shape[1]))
                neg_fea = torch.reshape(neg_fea, (neg_fea.shape[0], neg_fea.shape[1] * neg_fea.shape[1]))
            
                dist_ap, trans_ap = sinkhorn_divergence(anchor_fea, pos_fea)
                dist_an, trans_an = sinkhorn_divergence(anchor_fea, neg_fea)
                dis = max(dist_ap - dist_an, -self.beta)
                if dis>0:
                    wass_dis = wass_dis + dis
                    ot_pair = ot_pair + 1             
        
        if  wass_dis.item() > 0:
            wass_loss = wass_dis/ot_pair
        else:
            wass_loss = torch.zeros([1])
        return amloss, wass_loss, origin_cos*self.s


def l2_norm(input, axis = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def select_TeacStudPN(cos_theta, cos_theta_m, label):
    h,w = cos_theta.size()
    max_cos,max_idx = -1,-1
    min_cos,min_idx = -1,-1
    TeacP = torch.Tensor()
    TeacN = torch.Tensor()
    StudP = torch.Tensor()
    StudN = torch.Tensor()

    for i in range(h):
        for j in range(w):
            if max_cos < cos_theta[i,j]:
                max_cos = cos_theta[i,j].unsqueeze(dim=0)
                max_idx = j
            if min_cos > cos_theta[i,j]:
                min_cos = cos_theta[i,j].unsqueeze(dim=0)
                min_idx = j

        if max_idx == label[i]:
            TeacP = torch.cat((TeacP,max_cos))
            TeacN = torch.cat((TeacN, min_cos))
        else:

            StudN = torch.cat((StudN,torch.Tensor(max_cos)))
            StudP = torch.cat((StudP, cos_theta[i,label[i]].unsqueeze(dim=0)))

    return TeacP, TeacN,StudP, StudN


# p_logit: [batch, class_num]
# q_logit: [batch, class_num]
def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)- F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)

def Get_his(TeacP, TeacN,StudP, StudN,R=20):
    His_TeacP = histogram(TeacP, R)
    His_TeacN = histogram(TeacN, R)
    His_StudP = histogram(StudP, R)
    His_StudN = histogram(StudN, R)

    return His_TeacP, His_TeacN, His_StudP, His_StudN


def histogram(samilist,R):
    st = (1-(-1))/R
    n = samilist.shape
    r_list = np.arange(-1,1,st)
    bin = torch.Tensor(r_list)
    bin = bin.view(-1,1)
    samilist = samilist.view(-1,1)
    his_metrix = square_distance(samilist,bin)
    gama = 0.5
    deta_metrix = torch.exp(-gama*his_metrix)
    histogram = torch.sum(deta_metrix,dim=0)/n
    return histogram
