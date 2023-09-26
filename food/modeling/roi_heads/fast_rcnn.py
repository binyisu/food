# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import math
import os
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributions as dists
from detectron2.config import configurable
from detectron2.layers import (ShapeSpec, batched_nms, cat,
                               nonzero_tuple)
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers

# from detectron2.modeling.roi_heads.fast_rcnn import (FastRCNNOutputLayers,
#                                                      _log_classification_stats)

from detectron2.structures import Boxes, Instances, pairwise_iou
# from detectron2.structures.boxes import matched_boxlist_iou
#  fast_rcnn_inference)
from detectron2.utils import comm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from ...layers import MLP
from ...losses import ICLoss, UPLoss, IOULoss, ELoss

ROI_BOX_OUTPUT_LAYERS_REGISTRY = Registry("ROI_BOX_OUTPUT_LAYERS")
ROI_BOX_OUTPUT_LAYERS_REGISTRY.__doc__ = """
ROI_BOX_OUTPUT_LAYERS
"""

def fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    logits: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float = 1.0,
):
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, logits, image_shape, score_thresh, nms_thresh, topk_per_image, vis_iou_thr
        )
        for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    logits,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    vis_iou_thr: float,
):
    valid_mask = torch.isfinite(boxes).all(
        dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    scores = scores[:, :-1]
    second_scores = scores
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
        second_scores = second_scores[filter_inds[:, 0], :]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    logits = logits[filter_inds[:, 0], :]
    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, logits, filter_inds = boxes[keep], scores[keep], logits[keep], filter_inds[keep]

    # u = 22/torch.sum(torch.exp(logits)+1, dim=1)
    # keep = u<0.015
    # keep1 = filter_inds[:, 1]==20
    # keep2 = keep
    # for i in range(len(keep)):
    #     if keep[i] & keep1[i]:
    #         keep2[i] = True
    #     else:
    #         keep2[i] = False
    # keep = ~keep2
    # boxes, scores, logits, filter_inds = boxes[keep], scores[keep], logits[keep], filter_inds[keep]

    # # uncertain label reassignment.
    # second_pred_classes = filter_inds[:, 1]
    # for un_id in range(len(second_pred_classes)):
    #     if second_pred_classes[un_id] == 80:
    #         un_score = second_scores[un_id]
    #         no_un_score = un_score[:-1]
    #         un_label = no_un_score.argmax(dim=-1)
    #         filter_inds[un_id,1] = un_label

    # apply nms between known classes and unknown class for visualization.
    # vis_iou_thr = 0.1
    uncertain_id = 80
    if vis_iou_thr < 1.0:
        boxes, scores, filter_inds = unknown_aware_nms(
            boxes, scores, filter_inds, uncertain_id, iou_thr=vis_iou_thr)
    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]

def unknown_aware_nms(boxes, scores, labels, ukn_class_id=20, iou_thr=0.9):
    u_inds = labels[:, 1] == ukn_class_id
    k_inds = ~u_inds
    if k_inds.sum() == 0 or u_inds.sum() == 0:
        return boxes, scores, labels

    k_boxes, k_scores, k_labels = boxes[k_inds], scores[k_inds], labels[k_inds]
    u_boxes, u_scores, u_labels = boxes[u_inds], scores[u_inds], labels[u_inds]

    ious = pairwise_iou(Boxes(k_boxes), Boxes(u_boxes))
    mask = torch.ones((ious.size(0), ious.size(1), 2), device=ious.device)
    inds = (ious > iou_thr).nonzero()
    if not inds.numel():
        return boxes, scores, labels

    for [ind_x, ind_y] in inds:
        if k_scores[ind_x] >= u_scores[ind_y]:
            mask[ind_x, ind_y, 1] = 0
        else:
            mask[ind_x, ind_y, 0] = 0

    k_inds = mask[..., 0].mean(dim=1) == 1
    u_inds = mask[..., 1].mean(dim=0) == 1

    k_boxes, k_scores, k_labels = k_boxes[k_inds], k_scores[k_inds], k_labels[k_inds]
    u_boxes, u_scores, u_labels = u_boxes[u_inds], u_scores[u_inds], u_labels[u_inds]

    boxes = torch.cat([k_boxes, u_boxes])
    scores = torch.cat([k_scores, u_scores])
    labels = torch.cat([k_labels, u_labels])

    return boxes, scores, labels


logger = logging.getLogger(__name__)


def build_roi_box_output_layers(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
    return ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(name)(cfg, input_shape)

class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.
        Returns:
            scalar Tensor
        """
        self._log_accuracy()
        return F.cross_entropy(
            self.pred_class_logits, self.gt_classes, reduction="mean"
        )

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.
        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.
        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
            .expand(num_pred, K, B)
            .reshape(-1, B),
        )
        return boxes.view(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        # a = self.pred_class_logits[:, :-2]
        # b = self.pred_class_logits[:, -1:]
        # self.pred_class_logits = torch.cat([a, b], dim=1)
        probs = F.softmax(self.pred_class_logits, dim=-1)

        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        logits = self.pred_class_logits
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            logits,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class CosineFastRCNNOutputLayers(FastRCNNOutputLayers):

    @configurable
    def __init__(
        self,
        *args,
        scale: int = 20,
        vis_iou_thr: float = 1.0,
        number_classes: int = 20,
        **kargs,
    ):
        super().__init__(*args, **kargs)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.num_classes = number_classes
        self.cls_score = nn.Linear(
            self.cls_score.in_features, self.num_classes + 1, bias=False)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        # scaling factor
        self.scale = scale
        self.vis_iou_thr = vis_iou_thr


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['scale'] = cfg.MODEL.ROI_HEADS.COSINE_SCALE
        ret['vis_iou_thr'] = cfg.MODEL.ROI_HEADS.VIS_IOU_THRESH
        ret['number_classes'] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def forward(self, feats):

        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]):

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.vis_iou_thr,
        )

    def predict_boxes(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        if not len(proposals):
            return []
        proposal_deltas = predictions[1]
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = cat(
            [p.proposal_boxes.tensor for p in proposals], dim=0)
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(
        self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class FOODFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    @configurable
    def __init__(
        self,
        *args,
        num_known_classes,
        max_iters,
        up_loss_enable,
        up_loss_start_iter,
        up_loss_sampling_metric,
        up_loss_sampling_ratio,
        up_loss_topk,
        up_loss_alpha,
        up_loss_weight,
        e_loss_enable,
        e_loss_weight,
        hsic_loss_enable,
        ic_loss_out_dim,
        ic_loss_queue_size,
        ic_loss_in_queue_size,
        ic_loss_batch_iou_thr,
        ic_loss_queue_iou_thr,
        ic_loss_queue_tau,
        ic_loss_weight,
        _do_cls_dropout,
        DROPOUT_RATIO,
        **kargs
    ):
        super().__init__(*args, **kargs)
        self.num_known_classes = num_known_classes
        self.max_iters = max_iters

        self.up_loss = UPLoss(
            self.num_classes,
            sampling_metric=up_loss_sampling_metric,
            sampling_ratio=up_loss_sampling_ratio,
            topk=up_loss_topk,
            alpha=up_loss_alpha,
        )

        self.iou_loss = IOULoss(
            self.num_classes,
            sampling_metric=up_loss_sampling_metric,
            sampling_ratio=up_loss_sampling_ratio,
            topk=up_loss_topk,
            alpha=up_loss_alpha,
        )

        self.e_loss = ELoss(
            self.num_classes,
            sampling_metric=up_loss_sampling_metric,
            sampling_ratio=up_loss_sampling_ratio,
            topk=up_loss_topk,
            alpha=up_loss_alpha,
        )

        self.up_loss_enable = up_loss_enable
        self.up_loss_start_iter = up_loss_start_iter
        self.up_loss_weight = up_loss_weight

        self.e_loss_enable = e_loss_enable
        self.e_loss_weight = e_loss_weight

        self.hsic_loss_enable = hsic_loss_enable

        self.encoder = MLP(self.cls_score.in_features, ic_loss_out_dim)
        self.ic_loss_loss = ICLoss(tau=ic_loss_queue_tau)
        self.ic_loss_out_dim = ic_loss_out_dim
        self.ic_loss_queue_size = ic_loss_queue_size
        self.ic_loss_in_queue_size = ic_loss_in_queue_size
        self.ic_loss_batch_iou_thr = ic_loss_batch_iou_thr
        self.ic_loss_queue_iou_thr = ic_loss_queue_iou_thr
        self.ic_loss_weight = ic_loss_weight

        self.register_buffer('queue_cls', torch.zeros(
            ic_loss_queue_size, self.num_classes+1, ic_loss_out_dim))

        self.register_buffer('queue_reg', torch.zeros(
            ic_loss_queue_size, 4, ic_loss_out_dim))

        self.register_buffer('queue_label', torch.empty(
            self.num_known_classes, ic_loss_queue_size).fill_(-1).long())
        self.register_buffer('queue_ptr', torch.zeros(
            1, dtype=torch.long))
        self._do_cls_dropout = _do_cls_dropout
        self.DROPOUT_RATIO = DROPOUT_RATIO
        # new add
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'num_known_classes': cfg.MODEL.ROI_HEADS.NUM_KNOWN_CLASSES,
            "max_iters": cfg.SOLVER.MAX_ITER,

            "up_loss_enable": cfg.UPLOSS.ENABLE_UPLOSS,
            "up_loss_start_iter": cfg.UPLOSS.START_ITER,
            "up_loss_sampling_metric": cfg.UPLOSS.SAMPLING_METRIC,
            "up_loss_sampling_ratio": cfg.UPLOSS.SAMPLING_RATIO,
            "up_loss_topk": cfg.UPLOSS.TOPK,
            "up_loss_alpha": cfg.UPLOSS.ALPHA,
            "up_loss_weight": cfg.UPLOSS.WEIGHT,

            "e_loss_enable": cfg.ELOSS.ENABLE_ELOSS,
            "e_loss_weight": cfg.ELOSS.WEIGHT,

            "hsic_loss_enable": cfg.HSICLOSS.ENABLE_HSICLOSS,

            "ic_loss_out_dim": cfg.ICLOSS.OUT_DIM,
            "ic_loss_queue_size": cfg.ICLOSS.QUEUE_SIZE,
            "ic_loss_in_queue_size": cfg.ICLOSS.IN_QUEUE_SIZE,
            "ic_loss_batch_iou_thr": cfg.ICLOSS.BATCH_IOU_THRESH,
            "ic_loss_queue_iou_thr": cfg.ICLOSS.QUEUE_IOU_THRESH,
            "ic_loss_queue_tau": cfg.ICLOSS.TEMPERATURE,
            "ic_loss_weight": cfg.ICLOSS.WEIGHT,
            '_do_cls_dropout': cfg.MODEL.ROI_HEADS.CLS_DROPOUT,
            'DROPOUT_RATIO': cfg.MODEL.ROI_HEADS.DROPOUT_RATIO
        })
        return ret

    def forward(self, feats):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        # class dropout
        if self._do_cls_dropout:
            x_normalized = F.dropout(x_normalized, self.DROPOUT_RATIO, training=self.training)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        # encode feature with MLP
        mlp_feat = self.encoder(cls_x)

        return scores, proposal_deltas, mlp_feat

    def get_up_loss(self, scores, gt_classes, un_id):
        # start up loss after several warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            loss_cls_up = self.up_loss(scores, gt_classes, un_id)
        else:
            loss_cls_up = scores.new_tensor(0.0)

        return {"loss_cls_up_{}".format(un_id): self.up_loss_weight * loss_cls_up}

    def get_iou_loss(self, scores, gt_classes, iou):
        # start up loss after several warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            loss_cls_iou = self.iou_loss(scores, gt_classes, iou)
        else:
            loss_cls_iou = scores.new_tensor(0.0)
        decay_weight = 1.0 - 0.01*torch.exp(-torch.log(torch.tensor(0.01))/self.max_iters * storage.iter)
        return {"loss_iou": self.up_loss_weight * loss_cls_iou}

    def get_e_loss(self, scores, gt_classes):
        # start up loss after several warmup iters
        storage = get_event_storage()
        if storage.iter > self.up_loss_start_iter:
            loss_cls_e = self.e_loss(scores, gt_classes)
        else:
            loss_cls_e = scores.new_tensor(0.0)
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        # decay_weight = 1.0 - 0.01*torch.exp(-torch.log(torch.tensor(0.01))/self.max_iters * storage.iter)
        return {"loss_cls_e": self.e_loss_weight * decay_weight * loss_cls_e}

    def get_ic_loss(self, feat, gt_classes, ious):
        # select foreground and iou > thr instance in a mini-batch
        pos_inds = (ious > self.ic_loss_batch_iou_thr) & (
            gt_classes != self.num_classes)
        feat, gt_classes = feat[pos_inds], gt_classes[pos_inds]

        queue = self.queue.reshape(-1, self.ic_loss_out_dim)
        queue_label = self.queue_label.reshape(-1)
        queue_inds = queue_label != -1  # filter empty queue
        queue, queue_label = queue[queue_inds], queue_label[queue_inds]

        loss_ic_loss = self.ic_loss_loss(feat, gt_classes, queue, queue_label)
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_cls_ic": self.ic_loss_weight * decay_weight * loss_ic_loss}

    def get_hsic_loss(self, loss_hsic_loss):
        # loss decay
        storage = get_event_storage()
        decay_weight = 1.0 - storage.iter / self.max_iters
        return {"loss_cls_hsic": self.ic_loss_weight * decay_weight * loss_hsic_loss}

    @torch.no_grad()
    def _dequeue_and_enqueue(self, cls_weight, reg_weight):
        ptr = self.queue_ptr
        # in queue
        self.queue_cls[ptr] = cls_weight
        self.queue_reg[ptr] = reg_weight
        # out queue
        self.queue_ptr[0] = ptr[0] + 1 if ptr[0] + 1 < self.ic_loss_queue_size else 0


    @torch.no_grad()
    def concat_all_gather(self, tensor):
        world_size = comm.get_world_size()
        # single GPU, directly return the tensor
        if world_size == 1:
            return tensor
        # multiple GPUs, gather tensors
        tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
        output = torch.cat(tensors_gather, dim=0)
        return output

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < self.num_classes))[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds]
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas, mlp_feat = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(
                proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes")
                  else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": F.cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        # up loss
        ious = cat([p.iou for p in proposals], dim=0)

        if self.up_loss_enable:
            un_id = 2
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))
            losses.update(self.get_iou_loss(scores, gt_classes, ious))

            # un_id = 3
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))
            # un_id = 4
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))
            # un_id = 5
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))
            # un_id = 6
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))
            # un_id = 7
            # losses.update(self.get_up_loss(scores, gt_classes, un_id))

        # EDL loss
        if self.e_loss_enable:
            losses.update(self.get_e_loss(scores, gt_classes))

        if self.hsic_loss_enable:
            unbiased = True
            storage = get_event_storage()
            if storage.iter % 10 == 0:
                cls_weight = self.cls_score.weight.data
                reg_weight = self.bbox_pred.weight.data
                self._dequeue_and_enqueue(cls_weight, reg_weight)
            if storage.iter > 10:
                # if storage.iter % 10 == 0:
                # cls
                input1 = self.cls_score.weight.data
                input2 = torch.mean(self.queue_cls, dim=0)
                # hsic_cls = 1 - torch.mean(F.cosine_similarity(input1, input2))
                N = len(input1)
                sigma_x = np.sqrt(input1.size()[1])
                sigma_y = np.sqrt(input2.size()[1])
                # compute the kernels
                kernel_XX = self._kernel(input1, sigma_x)
                kernel_YY = self._kernel(input2, sigma_y)
                if unbiased:
                    tK = kernel_XX - torch.diag(kernel_XX)
                    tL = kernel_YY - torch.diag(kernel_YY)
                    hsic_cls = (
                            torch.trace(tK @ tL)
                            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
                    )
                    hsic_loss_cls = (1-hsic_cls) / (N * (N - 3))
                else:
                    KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
                    LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
                    hsic_cls = torch.trace(KH @ LH / (N - 1) ** 2)
                    hsic_loss_cls = 1 - hsic_cls
                # hsic_loss_cls = 1 - hsic_cls
                # hsic_cls = 0.5
                self.cls_score.weight.data = (1-hsic_cls) * self.cls_score.weight.data + hsic_cls * torch.mean(self.queue_cls, dim=0)
            # reg
                input11 = self.bbox_pred.weight.data
                input22 = torch.mean(self.queue_reg, dim=0)
                # hsic_reg = 1 - torch.mean(F.cosine_similarity(input1, input2))
                N = len(input11)
                sigma_x = np.sqrt(input11.size()[1])
                sigma_y = np.sqrt(input22.size()[1])
                # compute the kernels
                kernel_XX = self._kernel(input11, sigma_x)
                kernel_YY = self._kernel(input22, sigma_y)
                if unbiased:
                    tK = kernel_XX - torch.diag(kernel_XX)
                    tL = kernel_YY - torch.diag(kernel_YY)
                    hsic_reg = (
                            torch.trace(tK @ tL)
                            + (torch.sum(tK) * torch.sum(tL) / (N - 1) / (N - 2))
                            - (2 * torch.sum(tK, 0).dot(torch.sum(tL, 0)) / (N - 2))
                    )
                    hsic_loss_reg = (1-hsic_reg) / (N * (N - 3))
                else:
                    KH = kernel_XX - kernel_XX.mean(0, keepdim=True)
                    LH = kernel_YY - kernel_YY.mean(0, keepdim=True)
                    hsic_reg = torch.trace(KH @ LH / (N - 1) ** 2)
                    hsic_loss_reg = 1 - hsic_reg
                # hsic_reg = 0.5
                # hsic_loss_reg = 1 - hsic_reg
                self.bbox_pred.weight.data = (1-hsic_reg) * self.bbox_pred.weight.data + hsic_reg * torch.mean(self.queue_reg, dim=0)
            # HSIC loss
                self.hsic_loss = hsic_loss_cls + hsic_loss_reg
                if self.hsic_loss_enable:
                    losses.update(self.get_hsic_loss(self.hsic_loss))
                # if storage.iter == self.max_iters:
                #     self.cls_score.weight.data = (1 - hsic_cls) * self.cls_score.weight.data + hsic_cls * torch.mean(self.queue_cls,
                #                                                                                              dim=0)
                #     self.bbox_pred.weight.data = (1 - hsic_reg) * self.bbox_pred.weight.data + hsic_reg * torch.mean(self.queue_reg,
                #                                                                                          dim=0)
        # ious = cat([p.iou for p in proposals], dim=0)
        # # we first store feats in the queue, then compute loss
        # self._dequeue_and_enqueue(
        #     mlp_feat, gt_classes, ious, iou_thr=self.ic_loss_queue_iou_thr)
        # losses.update(self.get_ic_loss(mlp_feat, gt_classes, ious))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def _kernel(self, X, sigma):
        X = X.view(len(X), -1)
        XX = X @ X.t()
        X_sqnorms = torch.diag(XX)
        X_L2 = -2 * XX + X_sqnorms.unsqueeze(1) + X_sqnorms.unsqueeze(0)
        gamma = 1 / (2 * sigma ** 2)

        kernel_XX = torch.exp(-gamma * X_L2)
        return kernel_XX

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class PROSERFastRCNNOutputLayers(CosineFastRCNNOutputLayers):
    """PROSER
    """
    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.proser_weight = 0.1

    def get_proser_loss(self, scores, gt_classes):
        num_sample, num_classes = scores.shape
        mask = torch.arange(num_classes).repeat(
            num_sample, 1).to(scores.device)
        inds = mask != gt_classes[:, None].repeat(1, num_classes)
        mask = mask[inds].reshape(num_sample, num_classes-1)
        mask_scores = torch.gather(scores, 1, mask)

        targets = torch.zeros_like(gt_classes)
        fg_inds = gt_classes != self.num_classes
        targets[fg_inds] = self.num_classes-2
        targets[~fg_inds] = self.num_classes-1

        loss_cls_proser = F.cross_entropy(mask_scores, targets)
        return {"loss_cls_proser": self.proser_weight * loss_cls_proser}

    def losses(self, predictions, proposals, input_features=None):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions
        # parse classification outputs
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(
                proposals) else torch.empty(0)
        )
        # _log_classification_stats(scores, gt_classes)

        # parse box regression outputs
        if len(proposals):
            proposal_boxes = cat(
                [p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes")
                  else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty(
                (0, 4), device=proposal_deltas.device)

        losses = {
            "loss_cls_ce": F.cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        losses.update(self.get_proser_loss(scores, gt_classes))

        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}


@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class DropoutFastRCNNOutputLayers(CosineFastRCNNOutputLayers):

    @configurable
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.dropout = nn.Dropout(p=0.5)
        self.entropy_thr = 0.25

    def forward(self, feats, testing=False):
        # support shared & sepearte head
        if isinstance(feats, tuple):
            reg_x, cls_x = feats
        else:
            reg_x = cls_x = feats

        if reg_x.dim() > 2:
            reg_x = torch.flatten(reg_x, start_dim=1)
            cls_x = torch.flatten(cls_x, start_dim=1)

        x_norm = torch.norm(cls_x, p=2, dim=1).unsqueeze(1).expand_as(cls_x)
        x_normalized = cls_x.div(x_norm + 1e-5)

        # normalize weight
        temp_norm = (
            torch.norm(self.cls_score.weight.data, p=2, dim=1)
            .unsqueeze(1)
            .expand_as(self.cls_score.weight.data)
        )
        self.cls_score.weight.data = self.cls_score.weight.data.div(
            temp_norm + 1e-5
        )
        if testing:
            self.dropout.train()
            x_normalized = self.dropout(x_normalized)
        cos_dist = self.cls_score(x_normalized)
        scores = self.scale * cos_dist
        proposal_deltas = self.bbox_pred(reg_x)

        return scores, proposal_deltas

    def inference(self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions[0], proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_probs(
        self, predictions: List[Tuple[torch.Tensor, torch.Tensor]], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        # mean of multiple observations
        scores = torch.stack([pred[0] for pred in predictions], dim=-1)
        scores = scores.mean(dim=-1)
        # threshlod by entropy
        norm_entropy = dists.Categorical(scores.softmax(
            dim=1)).entropy() / np.log(self.num_classes)
        inds = norm_entropy > self.entropy_thr
        max_scores = scores.max(dim=1)[0]
        # set those with high entropy unknown objects
        scores[inds, :] = 0.0
        scores[inds, self.num_classes-1] = max_scores[inds]

        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

@ROI_BOX_OUTPUT_LAYERS_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
        self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(x)

        if self._do_cls_dropout:
            x = F.dropout(x, self._dropout_ratio, training=self.training)
        scores = self.cls_score(x)

        return scores, proposal_deltas