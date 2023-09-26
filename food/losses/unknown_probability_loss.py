import torch
import torch.distributions as dists
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np


class UPLoss(nn.Module):
    """Unknown Probability Loss
    """

    def __init__(self,
                 num_classes: int,
                 sampling_metric: str = "min_score",
                 sampling_ratio: int = 1,
                 topk: int = 3,
                 alpha: float = 1.0,
                 unk: int = 2):
        super().__init__()
        self.num_classes = num_classes
        assert sampling_metric in ["min_score", "max_entropy", "random", "max_unknown_prob", "max_energy", "max_condition_energy", "VIM", "edl_dirichlet"]
        self.sampling_metric = sampling_metric
        self.sampling_ratio = sampling_ratio
        # if topk==-1, sample len(fg)*2 examples
        self.topk = topk
        self.alpha = alpha
        self.unk = unk
        weight = torch.FloatTensor(1).fill_(0.1)
        self.weight = nn.Parameter(weight, requires_grad=True)
        bias = torch.FloatTensor(1).fill_(0)
        self.bias = nn.Parameter(bias, requires_grad=True)

    def _soft_cross_entropy(self, input_gt_scores: Tensor, un_id, input: Tensor, target: Tensor):
        # -log(P_u)
        logprobs = F.log_softmax(input, dim=1)

        # # -log(P_u)-log(1-P_u)
        # # a = target.clone()
        # # b = target
        # # b[:self.topk, self.num_classes - un_id] = 0
        # # b[self.topk:, self.num_classes-1] = 1
        # # c = a * F.softmax(input) + b
        # # c[c==0] = 1
        # # logprobs = torch.log(c)
        # # # target[:self.topk, self.num_classes - un_id] = input_gt_scores[:self.topk]*10
        # # # target[self.topk:, self.num_classes-1] = input_gt_scores[self.topk:]*10
        # # target[:self.topk, self.num_classes - un_id] = 1
        # # target[self.topk:, self.num_classes-1] = 1
        #
        # # -log(P_u)
        # logprobs = F.logsigmoid(0.09*input)
        return -(target * logprobs).sum() / input.shape[0]


    def _sampling(self, scores: Tensor, labels: Tensor):
        fg_inds = labels != self.num_classes
        fg_scores, fg_labels = scores[fg_inds], labels[fg_inds]
        bg_scores, bg_labels = scores[~fg_inds], labels[~fg_inds]

        # remove unknown classes
        _fg_scores = torch.cat(
            [fg_scores[:, :self.num_classes-1], fg_scores[:, -1:]], dim=1)
        _bg_scores = torch.cat(
            [bg_scores[:, :self.num_classes-1], bg_scores[:, -1:]], dim=1)

        num_fg = fg_scores.size(0)
        topk = num_fg if (self.topk == -1) or (num_fg <
                                               self.topk) else self.topk
        # use maximum entropy as a metric for uncertainty
        # we select topk proposals with maximum entropy
        if self.sampling_metric == "max_entropy":
            pos_metric = dists.Categorical(
                _fg_scores.softmax(dim=1)).entropy()
            neg_metric = dists.Categorical(
                _bg_scores.softmax(dim=1)).entropy()
        # use minimum score as a metric for uncertainty
        # we select topk proposals with minimum max-score
        elif self.sampling_metric == "min_score":
            pos_metric = -_fg_scores.max(dim=1)[0]
            neg_metric = -_bg_scores.max(dim=1)[0]

        # we randomly select topk proposals
        elif self.sampling_metric == "random":
            pos_metric = torch.rand(_fg_scores.size(0),).to(scores.device)
            neg_metric = torch.rand(_bg_scores.size(0),).to(scores.device)
        # max unknown prob
        elif self.sampling_metric == "max_unknown_prob":
            pos_metric = -fg_scores[:, -2]
            neg_metric = -bg_scores[:, -2]
        elif self.sampling_metric == "max_energy":
            pos_metric = -torch.logsumexp(fg_scores, dim=1)
            neg_metric = -torch.logsumexp(bg_scores, dim=1)
        elif self.sampling_metric == "edl_dirichlet":
            pos_metric = (self.num_classes+1)/torch.sum(torch.exp(fg_scores)+1, dim=1)
            neg_metric = (self.num_classes+1)/torch.sum(torch.exp(bg_scores)+1, dim=1)
        elif self.sampling_metric == "max_condition_energy":
            pos_metric = -torch.logsumexp(_fg_scores, dim=1)
            neg_metric = -torch.logsumexp(_bg_scores, dim=1)
        elif self.sampling_metric == "VIM":
            fg_scores_mean = fg_scores - torch.mean(fg_scores, dim=0)
            _fg_scores_mean_transpose = torch.transpose(fg_scores_mean, dim0=1, dim1=0)
            A = torch.mm(fg_scores_mean, _fg_scores_mean_transpose)
            # A_norm = torch.norm(A, p=2, dim=1).unsqueeze(1).expand_as(A)
            # A_normized = A.div(A_norm + 1e-5)
            A = A/(A.size()[0]-1)
            (evals, evecs) = torch.eig(A, eigenvectors=True)
            evecs = evecs.detach()
            pos_metric = - evals[:, 0]
            _, pos_inds = pos_metric.topk(topk)
            R = evecs[:, pos_inds]
            R_transpose = torch.transpose(R, dim0=1, dim1=0)
            fg_scores_transform = torch.mm(R_transpose, fg_scores)
            fg_scores = fg_scores_transform
            fg_labels = fg_labels[pos_inds]
            neg_metric = -_bg_scores.max(dim=1)[0]

        if self.sampling_metric == "VIM":
            _, neg_inds = neg_metric.topk(topk * self.sampling_ratio)
            bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]
        else:
            _, pos_inds = pos_metric.topk(topk)
            _, neg_inds = neg_metric.topk(topk*self.sampling_ratio)
            fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
            bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]
        # _, pos_inds = pos_metric.topk(topk)
        # _, neg_inds = neg_metric.topk(topk*self.sampling_ratio)
        # fg_scores, fg_labels = fg_scores[pos_inds], fg_labels[pos_inds]
        # bg_scores, bg_labels = bg_scores[neg_inds], bg_labels[neg_inds]

        return fg_scores, bg_scores, fg_labels, bg_labels

    def forward(self, scores: Tensor, labels: Tensor, un_id: Tensor):
        fg_scores, bg_scores, fg_labels, bg_labels = self._sampling(
            scores, labels)
        # sample both fg and bg
        scores = torch.cat([fg_scores, bg_scores])
        labels = torch.cat([fg_labels, bg_labels])

        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != labels[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)

        # remove max score except unknown class
        # remove_un_scores = torch.cat(
        #     [scores[:, :self.num_classes - 1], scores[:, -1:]], dim=1)
        # max_score, index = torch.max(remove_un_scores, 1)
        # mask_ = torch.arange(self.num_classes).repeat(num_sample, 1).to(scores.device)
        # inds_ = mask_ != index[:, None].repeat(1, self.num_classes)
        # mask_ = mask_[inds_].reshape(num_sample, self.num_classes-1)
        # remove_max_scores = torch.gather(remove_un_scores, 1, mask_)
        # s = scores[:, 80:81]
        # remove_max_scores = torch.cat(
        #     [remove_max_scores, scores[:, 80:81]], dim=1)
        # mask_scores = remove_max_scores

        gt_scores = torch.gather(
            F.softmax(scores, dim=1), 1, labels[:, None]).squeeze(1)
        mask_scores = torch.gather(scores, 1, mask)

        S = torch.sum(torch.exp(scores)+1, dim=1, keepdim=True)
        # ints_new = ~inds
        # y = ints_new.long()
        # A = torch.sum(y * (torch.digamma(S)-torch.digamma(torch.exp(scores)+1)), dim=1, keepdim=True)
        A = self.num_classes/S
        A = A.squeeze(1)
        # y = torch.zeros_like(scores)
        # y[:3, self.num_classes - 2] = 1.0
        # y[3:, self.num_classes - 1] = 1.0

        gt_scores[gt_scores < 0] = 0.0
        targets = torch.zeros_like(mask_scores)
        num_fg = fg_scores.size(0)
        targets[:num_fg, self.num_classes-2] = gt_scores[:num_fg] * \
            (1-gt_scores[:num_fg]).pow(self.alpha)
        targets[num_fg:, self.num_classes-1] = gt_scores[num_fg:] * \
            (1-gt_scores[num_fg:]).pow(self.alpha)
        # targets[:num_fg, self.num_classes - un_id] = 1.0
        # targets[num_fg:, self.num_classes - 1] = 1.0

        return self._soft_cross_entropy(A, un_id, mask_scores, targets.detach())
