# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from mmdet.models.backbones.csp_darknet import Focus
from mmdet.models.layers import ChannelAttention
from mmengine.config import ConfigDict
from torch import Tensor

from mmyolo.models import RepVGGBlock
from mmyolo.models.dense_heads import (PPYOLOEHead, RTMDetHead, YOLOv5Head, YOLOv6Head,
                                       YOLOv7Head, YOLOv8Head, YOLOXHead)
from mmyolo.models.layers import ImplicitA, ImplicitM
from ..backbone import DeployFocus, GConvFocus, NcnnFocus
from ..bbox_code import (rtmdet_bbox_decoder, yolov5_bbox_decoder,
                         yolox_bbox_decoder)
from ..nms import batched_nms, efficient_nms, onnx_nms
from .backend import MMYOLOBackend


class DeployModel(nn.Module):
    transpose = False

    def __init__(self,
                 baseModel: nn.Module,
                 backend: MMYOLOBackend,
                 postprocess_cfg: Optional[ConfigDict] = None):
        super().__init__()
        self.baseModel = baseModel
        self.baseHead = baseModel.bbox_head
        self.backend = backend
        if postprocess_cfg is None:
            self.with_postprocess = False
        else:
            self.with_postprocess = True
            self.__init_sub_attributes()
            self.detector_type = type(self.baseHead)
            self.pre_top_k = postprocess_cfg.get('pre_top_k', 1000)
            self.keep_top_k = postprocess_cfg.get('keep_top_k', 100)
            self.iou_threshold = postprocess_cfg.get('iou_threshold', 0.65)
            self.score_threshold = postprocess_cfg.get('score_threshold', 0.25)
            self.export_type = postprocess_cfg.get('export_type', None)
        self.__switch_deploy()

    def __init_sub_attributes(self):
        self.bbox_decoder = self.baseHead.bbox_coder.decode
        self.prior_generate = self.baseHead.prior_generator.grid_priors
        self.num_base_priors = self.baseHead.num_base_priors
        self.featmap_strides = self.baseHead.featmap_strides
        self.num_classes = self.baseHead.num_classes

    def __switch_deploy(self):
        headType = type(self.baseHead)
        if not self.with_postprocess:
            if headType in (YOLOv5Head, YOLOv7Head):
                self.baseHead.head_module.forward_single = self.forward_single
            elif headType in (PPYOLOEHead, YOLOv8Head):
                self.baseHead.head_module.reg_max = 0

        if self.backend in (MMYOLOBackend.HORIZONX3, MMYOLOBackend.NCNN,
                            MMYOLOBackend.TORCHSCRIPT):
            self.transpose = True
        for layer in self.baseModel.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, ChannelAttention):
                layer.global_avgpool.forward = self.forward_gvp
            elif isinstance(layer, Focus):
                # onnxruntime openvino tensorrt8 tensorrt7
                if self.backend in (MMYOLOBackend.ONNXRUNTIME,
                                    MMYOLOBackend.OPENVINO,
                                    MMYOLOBackend.TENSORRT8,
                                    MMYOLOBackend.TENSORRT7):
                    self.baseModel.backbone.stem = DeployFocus(layer)
                # ncnn
                elif self.backend == MMYOLOBackend.NCNN:
                    self.baseModel.backbone.stem = NcnnFocus(layer)
                # switch focus to group conv
                else:
                    self.baseModel.backbone.stem = GConvFocus(layer)

    def pred_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     objectnesses: Optional[List[Tensor]] = None,
                     **kwargs):
        assert len(cls_scores) == len(bbox_preds)
        dtype = cls_scores[0].dtype
        device = cls_scores[0].device

        nms_func = self.select_nms()
        if self.detector_type in (YOLOv5Head, YOLOv7Head):
            bbox_decoder = yolov5_bbox_decoder
        elif self.detector_type is RTMDetHead:
            bbox_decoder = rtmdet_bbox_decoder
        elif self.detector_type is YOLOXHead:
            bbox_decoder = yolox_bbox_decoder
        else:
            bbox_decoder = self.bbox_decoder

        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generate(
            featmap_sizes, dtype=dtype, device=device)

        flatten_priors = torch.cat(mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size[0] * featmap_size[1] * self.num_base_priors, ),
                stride) for featmap_size, stride in zip(
                    featmap_sizes, self.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)
        if torch.onnx.is_in_onnx_export() and isinstance(self.baseHead, (YOLOv8Head, YOLOXHead, YOLOv6Head)):
            num_scales = len(cls_scores)
            scale_outputs = []
            for _, scale_id in enumerate(range(num_scales)):
                if objectnesses is not None:
                    scale_output = torch.cat([bbox_preds[scale_id],
                                              objectnesses[scale_id],
                                              cls_scores[scale_id]], axis=1)
                else:
                    scale_output = torch.cat([bbox_preds[scale_id],
                                              cls_scores[scale_id]], axis=1)
                scale_outputs.append(scale_output)
            num_elements_per_box = bbox_preds[0].shape[1]
            bbox_preds = []
            objectnesses = [] if objectnesses is not None else None
            cls_scores = []
            for _, scale_id in enumerate(range(num_scales)):
                bbox_preds.append(scale_outputs[scale_id][:, :num_elements_per_box, ...])
                if objectnesses is not None:
                    objectnesses.append(scale_outputs[scale_id][:, num_elements_per_box:num_elements_per_box+1, ...])
                    cls_scores.append(scale_outputs[scale_id][:, num_elements_per_box+1:, ...])
                else:
                    cls_scores.append(scale_outputs[scale_id][:, num_elements_per_box:, ...])

        if isinstance(self.baseHead, YOLOv8Head):
            for scale_idx, _ in enumerate(bbox_preds):
                b, _, h, w = cls_scores[scale_idx].shape
                bbox_preds[scale_idx] = bbox_preds[scale_idx].reshape(
                        [-1, 4, self.baseHead.head_module.reg_max, h * w]).permute(0, 3, 1, 2)
                bbox_preds[scale_idx] = bbox_preds[scale_idx].softmax(3).matmul(
                    self.baseHead.head_module.proj.view([-1, 1])).squeeze(-1)
                bbox_preds[scale_idx] = bbox_preds[scale_idx].transpose(1, 2).reshape(b, -1, h, w)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.num_classes)
            for cls_score in cls_scores
        ]
        cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        if objectnesses is not None:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
            cls_scores = cls_scores * (flatten_objectness.unsqueeze(-1))

        scores = cls_scores

        bboxes = bbox_decoder(flatten_priors[None], flatten_bbox_preds,
                              flatten_stride)

        return nms_func(bboxes, scores, self.keep_top_k, self.iou_threshold,
                        self.score_threshold, self.pre_top_k, self.keep_top_k, export_type=self.export_type)

    def select_nms(self):
        if self.backend in (MMYOLOBackend.ONNXRUNTIME, MMYOLOBackend.OPENVINO):
            nms_func = onnx_nms
        elif self.backend == MMYOLOBackend.TENSORRT8:
            nms_func = efficient_nms
        elif self.backend == MMYOLOBackend.TENSORRT7:
            nms_func = batched_nms
        else:
            raise NotImplementedError
        if type(self.baseHead) in (YOLOv5Head, YOLOv7Head, YOLOXHead):
            nms_func = partial(nms_func, box_coding=1)

        return nms_func

    def forward(self, inputs: Tensor):
        neck_outputs = self.baseModel(inputs)
        if self.with_postprocess:
            return self.pred_by_feat(*neck_outputs)
        else:
            outputs = []
            if self.transpose:
                for feats in zip(*neck_outputs):
                    if self.backend in (MMYOLOBackend.NCNN,
                                        MMYOLOBackend.TORCHSCRIPT):
                        outputs.append(
                            torch.cat(
                                [feat.permute(0, 2, 3, 1) for feat in feats],
                                -1))
                    else:
                        outputs.append(torch.cat(feats, 1).permute(0, 2, 3, 1))
            else:
                for feats in zip(*neck_outputs):
                    outputs.append(torch.cat(feats, 1))
            return tuple(outputs)

    @staticmethod
    def forward_single(x: Tensor, convs: nn.Module) -> Tuple[Tensor]:
        if isinstance(convs, nn.Sequential) and any(
                type(m) in (ImplicitA, ImplicitM) for m in convs):
            a, c, m = convs
            aw = a.implicit.clone()
            mw = m.implicit.clone()
            c = deepcopy(c)
            nw, cw, _, _ = c.weight.shape
            na, ca, _, _ = aw.shape
            nm, cm, _, _ = mw.shape
            c.bias = nn.Parameter(c.bias + (
                c.weight.reshape(nw, cw) @ aw.reshape(ca, na)).squeeze(1))
            c.bias = nn.Parameter(c.bias * mw.reshape(cm))
            c.weight = nn.Parameter(c.weight * mw.transpose(0, 1))
            convs = c
        feat = convs(x)
        return (feat, )

    @staticmethod
    def forward_gvp(x: Tensor) -> Tensor:
        return torch.mean(x, [2, 3], keepdim=True)
