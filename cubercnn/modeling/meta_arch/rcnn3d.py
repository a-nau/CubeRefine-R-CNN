# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import BACKBONE_REGISTRY, Backbone
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.roi_heads import (
    build_roi_heads as generalized_rcnn_build_roi_heads,
)
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import _log_api_usage
from torch import nn

from cubercnn.modeling.roi_heads import build_roi_heads


@META_ARCH_REGISTRY.register()
class RCNN3D(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        cube_on: bool = True,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        self.cube_on = cube_on

    @classmethod
    def from_config(cls, cfg, priors=None):
        backbone = build_backbone(cfg, priors=priors)

        try:
            if cfg.MODEL.CUBE_ON:
                roi_heads = build_roi_heads(cfg, backbone.output_shape(), priors=priors)
            else:
                roi_heads = generalized_rcnn_build_roi_heads(
                    cfg, backbone.output_shape()
                )
        except TypeError as e:
            raise TypeError(
                f'ERROR: Make sure to set MODEL.ROI_HEADS.NAME="StandardROIHeads" or "ROIHeads3D", respectively: {repr(e)}'
            )

        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(
                cfg, backbone.output_shape()
            ),
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cube_on": cfg.MODEL.CUBE_ON,
        }

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):

        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [
            info["height"] / im.shape[1] for (info, im) in zip(batched_inputs, images)
        ]

        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info["K"]) for info in batched_inputs]

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )

        if self.cube_on:
            _, detector_losses = self.roi_heads(
                images, features, proposals, Ks, im_scales_ratio, gt_instances
            )
        else:  # default GeneralizedRCNN
            _, detector_losses = self.roi_heads(
                images, features, proposals, gt_instances
            )

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(
                    batched_inputs, proposals
                )  # default visualization of bbox proposals only

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)

        # scaling factor for the sample relative to its original scale
        # e.g., how much has the image been upsampled by? or downsampled?
        im_scales_ratio = [
            info["height"] / im.shape[1] for (info, im) in zip(batched_inputs, images)
        ]

        # The unmodified intrinsics for the image
        Ks = [torch.FloatTensor(info["K"]) for info in batched_inputs]

        features = self.backbone(images.tensor)

        # Pass oracle 2D boxes into the RoI heads
        if type(batched_inputs == list) and np.any(
            ["oracle2D" in b for b in batched_inputs]
        ):
            oracles = [b["oracle2D"] for b in batched_inputs]
            results, _ = self.roi_heads(
                images, features, oracles, Ks, im_scales_ratio, None
            )

        # normal inference
        else:
            proposals, _ = self.proposal_generator(images, features, None)
            if self.cube_on:
                results, _ = self.roi_heads(
                    images, features, proposals, Ks, im_scales_ratio, None
                )
            else:  # default GeneralizedRCNN
                results, _ = self.roi_heads(images, features, proposals, None)

        if do_postprocess:
            assert (
                not torch.jit.is_scripting()
            ), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results


def build_model(cfg, priors=None):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model


def build_backbone(cfg, input_shape=None, priors=None):
    """
    Build a backbone from `cfg.MODEL.BACKBONE.NAME`.

    Returns:
        an instance of :class:`Backbone`
    """
    if input_shape is None:
        input_shape = ShapeSpec(channels=len(cfg.MODEL.PIXEL_MEAN))

    backbone_name = cfg.MODEL.BACKBONE.NAME
    backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape, priors)
    assert isinstance(backbone, Backbone)
    return backbone
