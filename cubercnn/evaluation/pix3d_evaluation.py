# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime

import cv2
import detectron2.utils.comm as comm
import numpy as np
import pycocotools.mask as mask_util
import torch
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from pycocotools.coco import COCO
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

import cubercnn.util.VOCap as VOCap
from cubercnn.structures.mesh import MeshInstances, batch_crop_meshes_within_box
from cubercnn.util import shape as shape_utils
from cubercnn.util.metrics import compare_meshes

logger = logging.getLogger(__name__)


class Pix3DEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection, segmentation and meshes
    outputs.
    """

    def __init__(self, dataset_name, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have the following corresponding metadata:
                    "json_file": the path to the COCO format annotation
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._tasks = ("bbox", "mesh")
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._device = torch.device("cuda")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._filter_iou = 0.3
        self._iou_thresholds = [0.5, 0.75]

        # load unique obj files
        assert dataset_name is not None
        # load unique obj meshes
        # Pix3D models are few in number (= 735) thus it's more efficient
        # to load them in memory rather than read them at every iteration
        logger.info("Loading unique objects from {}...".format(dataset_name))
        json_file = MetadataCatalog.get(dataset_name).json_file
        model_root = MetadataCatalog.get(dataset_name).image_root
        self._mesh_models = load_unique_meshes(json_file, model_root)
        logger.info("Unique objects loaded: {}".format(len(self._mesh_models)))

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            self._predictions = comm.gather(self._predictions, dst=0)
            self._predictions = list(itertools.chain(*self._predictions))

            if not comm.is_main_process():
                return

        self._results = OrderedDict()

        if "instances" in self._predictions[0]:
            self._eval_predictions()
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_predictions(self):
        """
        Evaluate mesh rcnn predictions.
        """

        if "mesh" in self._tasks:
            self._results["shape"] = {}
            for iou_thresh in self._iou_thresholds:
                results = evaluate_for_pix3d(
                    self._predictions,
                    self._coco_api,
                    self._metadata,
                    self._filter_iou,
                    iou_thresh=iou_thresh,
                    mesh_models=self._mesh_models,
                    device=self._device,
                )
                # print results
                self._logger.info("Box AP %.5f" % (results["box_ap@%s" % iou_thresh]))
                self._logger.info("Mesh AP %.5f" % (results["mesh_ap@%s" % iou_thresh]))
                self._results["shape"].update(results)


def evaluate_for_pix3d(
    predictions,
    dataset,
    metadata,
    filter_iou,
    mesh_models=None,
    iou_thresh=0.5,
    device=None,
    vis_preds=False,
):
    from PIL import Image

    if device is None:
        device = torch.device("cpu")

    F1_TARGET = "F1@0.300000"

    # classes
    cat_ids = sorted(dataset.getCatIds())
    reverse_id_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }

    # initialize tensors to record box & mesh AP, number of gt positives
    box_apscores, box_aplabels = {}, {}
    mesh_apscores, mesh_aplabels = {}, {}
    mesh_f1, mesh_chamfer, mesh_normal = {}, {}, {}
    npos = {}
    for cat_id in cat_ids:
        box_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        box_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        mesh_apscores[cat_id] = [torch.tensor([], dtype=torch.float32, device=device)]
        mesh_aplabels[cat_id] = [torch.tensor([], dtype=torch.uint8, device=device)]
        npos[cat_id] = 0.0
        mesh_f1[cat_id] = []
        mesh_chamfer[cat_id] = []
        mesh_normal[cat_id] = []
    box_covered = []
    mesh_covered = []

    # number of gt positive instances per class
    for gt_ann in dataset.dataset["annotations"]:
        gt_label = gt_ann["category_id"]
        image_file_name = dataset.loadImgs([gt_ann["image_id"]])[0]["file_name"]
        npos[gt_label] += 1.0

    for prediction in predictions:

        original_id = prediction["image_id"]
        image_width = dataset.loadImgs([original_id])[0]["width"]
        image_height = dataset.loadImgs([original_id])[0]["height"]
        image_size = [image_height, image_width]
        image_file_name = dataset.loadImgs([original_id])[0]["file_name"]

        if "instances" not in prediction:
            continue

        num_img_preds = len(prediction["instances"])
        if num_img_preds == 0:
            continue

        # ground truth
        # anotations corresponding to original_id (aka coco image_id)
        gt_ann_ids = dataset.getAnnIds(imgIds=[original_id])
        for gt_ann_id in range(len(gt_ann_ids)):
            gt_anns = dataset.loadAnns(gt_ann_ids)[gt_ann_id]
            assert gt_anns["image_id"] == original_id

            # get original ground truth mask, box, label & mesh
            gt_box = np.array(gt_anns["bbox"]).reshape(-1, 4)  # xywh from coco
            gt_box = BoxMode.convert(gt_box, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            gt_label = gt_anns["category_id"]
            faux_gt_targets = Boxes(
                torch.tensor(gt_box, dtype=torch.float32, device=device)
            )

            # load gt mesh and extrinsics/intrinsics
            gt_R = torch.tensor(gt_anns["model_R"]).to(device)
            gt_t = torch.tensor(gt_anns["model_t"]).to(device)
            gt_K = torch.tensor(gt_anns["K"]).to(device)
            if mesh_models is not None:
                modeltype = gt_anns["model"]
                gt_verts, gt_faces = (
                    mesh_models[modeltype][0].clone(),
                    mesh_models[modeltype][1].clone(),
                )
                gt_verts = gt_verts.to(device)
                gt_faces = gt_faces.to(device)
            else:
                # load from disc
                raise NotImplementedError
            gt_verts = shape_utils.transform_verts(gt_verts, gt_R, gt_t)
            # # gt zrange (zrange stores min_z and max_z)
            # gt_zrange = torch.stack([gt_verts[:, 2].min(), gt_verts[:, 2].max()])
            # # zranges = torch.stack([gt_zrange] * len(meshes), dim=0)  # can use GT
            gt_mesh_world = Meshes(verts=[gt_verts], faces=[gt_faces])

            gt_mesh_unitcube3d = batch_crop_meshes_within_box(
                MeshInstances([(gt_verts, gt_faces)]),
                torch.tensor(gt_box, device=device),
                gt_K.unsqueeze(0),
            ).to(device)

            # predictions
            scores = prediction["instances"].scores
            boxes = prediction["instances"].pred_boxes.to(device)
            labels = prediction["instances"].pred_classes

            if hasattr(prediction["instances"], "pred_meshes_world"):
                meshes_world = prediction["instances"].pred_meshes_world.to(
                    device
                )  # preditected meshes in world CS
            if hasattr(prediction["instances"], "pred_meshes") and hasattr(
                prediction["instances"], "pred_bbox3D"
            ):
                meshes_unitcube3d = prediction["instances"].pred_meshes.to(
                    device
                )  # preditected meshes inside unit cube -> need to convert
                pred_bbox3D = prediction["instances"].pred_bbox3D
                if not hasattr(prediction["instances"], "pred_meshes_world"):
                    zranges = torch.stack(
                        [
                            torch.tensor([v[..., 2].min(), v[..., 2].max()])
                            for v in pred_bbox3D
                        ]
                    ).to(device)
                    gt_Ks = gt_K.view(1, 3).expand(len(meshes_unitcube3d), 3)
                    meshes_world = transform_meshes_to_camera_coord_system(
                        meshes=meshes_unitcube3d,
                        boxes=boxes.tensor,
                        zranges=zranges,
                        Ks=gt_Ks,
                        imsize=image_size,
                        adjust_cs=True,
                    )
            assert prediction["instances"].image_size[0] == image_height
            assert prediction["instances"].image_size[1] == image_width

            # box iou
            boxiou = pairwise_iou(boxes, faux_gt_targets)

            # filter predictions with iou > filter_iou
            valid_pred_ids = boxiou > filter_iou

            shape_metrics_unitcube3d = compare_meshes(
                meshes_unitcube3d, gt_mesh_unitcube3d, reduce=False
            )
            shape_metrics = shape_metrics_unitcube3d

            # sort predictions in descending order
            scores_sorted, idx_sorted = torch.sort(scores, descending=True)

            for pred_id in range(num_img_preds):
                # remember we only evaluate the preds that have overlap more than
                # iou_filter with the ground truth prediction
                if valid_pred_ids[idx_sorted[pred_id], 0] == 0:
                    continue
                # map to dataset category id
                pred_label = reverse_id_mapping[labels[idx_sorted[pred_id]].item()]
                pred_biou = boxiou[idx_sorted[pred_id]].item()
                pred_score = scores[idx_sorted[pred_id]].view(1).to(device)
                # note that metrics returns f1 in % (=x100)
                pred_f1 = shape_metrics[F1_TARGET][idx_sorted[pred_id]].item() / 100.0

                # box
                tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
                if (
                    (pred_label == gt_label)
                    and (pred_biou > iou_thresh)
                    and (original_id not in box_covered)
                ):
                    tpfp[0] = 1
                    box_covered.append(original_id)
                box_apscores[pred_label].append(pred_score)
                box_aplabels[pred_label].append(tpfp)

                # mesh
                tpfp = torch.tensor([0], dtype=torch.uint8, device=device)
                if (
                    (pred_label == gt_label)
                    and (pred_f1 > iou_thresh)
                    and (original_id not in mesh_covered)
                ):
                    tpfp[0] = 1
                    mesh_covered.append(original_id)
                mesh_apscores[pred_label].append(pred_score)
                mesh_aplabels[pred_label].append(tpfp)
            mesh_normal[pred_label].append(
                shape_metrics["AbsNormalConsistency"][idx_sorted[0]].item()
            )
            mesh_chamfer[pred_label].append(
                shape_metrics["Chamfer-L2"][idx_sorted[0]].item()
            )
            mesh_f1[pred_label].append(
                shape_metrics[F1_TARGET][idx_sorted[0]].item() / 100.0
            )

    # check things for eval
    # assert npos.sum() == len(dataset.dataset["annotations"])
    # convert to tensors
    pix3d_metrics = {}
    boxap, meshap = 0.0, 0.0
    chamfer, normal, f1 = 0.0, 0.0, 0.0
    valid = 0.0
    for cat_id in cat_ids:
        cat_name = dataset.loadCats([cat_id])[0]["name"]
        if npos[cat_id] == 0:
            continue
        valid += 1

        cat_box_ap = VOCap.compute_ap(
            torch.cat(box_apscores[cat_id]),
            torch.cat(box_aplabels[cat_id]),
            npos[cat_id],
        )
        boxap += cat_box_ap
        pix3d_metrics["box_ap@%s - %s" % (iou_thresh, cat_name)] = cat_box_ap

        cat_mesh_ap = VOCap.compute_ap(
            torch.cat(mesh_apscores[cat_id]),
            torch.cat(mesh_aplabels[cat_id]),
            npos[cat_id],
        )
        meshap += cat_mesh_ap
        pix3d_metrics["mesh_ap@%s - %s" % (iou_thresh, cat_name)] = cat_mesh_ap

        # Additional 3D metrics
        if len(mesh_chamfer[cat_id]) > 0:
            cat_chamfer = sum(mesh_chamfer[cat_id]) / len(mesh_chamfer[cat_id])
            chamfer += cat_chamfer
            pix3d_metrics["chamfer - %s" % cat_name] = cat_chamfer
            cat_normal = sum(mesh_normal[cat_id]) / len(mesh_normal[cat_id])
            normal += cat_normal
            pix3d_metrics["normal - %s" % cat_name] = cat_normal
            cat_f1 = sum(mesh_f1[cat_id]) / len(mesh_normal[cat_id])
            f1 += cat_f1
            pix3d_metrics["f1_03 - %s" % cat_name] = cat_f1

    pix3d_metrics["box_ap@%s" % iou_thresh] = boxap / valid
    pix3d_metrics["mesh_ap@%s" % iou_thresh] = meshap / valid
    pix3d_metrics["chamfer"] = chamfer / valid
    pix3d_metrics["normal"] = normal / valid
    pix3d_metrics["f1_03"] = f1 / valid

    return pix3d_metrics


def transform_meshes_to_camera_coord_system(
    meshes, boxes, zranges, Ks, imsize, adjust_cs=True
):
    device = meshes.device
    new_verts, new_faces = [], []
    h, w = imsize
    im_size = torch.tensor([w, h], device=device).view(1, 2)
    assert len(meshes) == len(zranges)
    for i in range(len(meshes)):
        verts, faces = meshes.get_mesh_verts_faces(i)
        if verts.numel() == 0:
            verts, faces = ico_sphere(level=3, device=device).get_mesh_verts_faces(0)
        assert not torch.isnan(verts).any()
        assert not torch.isnan(faces).any()
        roi = boxes[i].view(1, 4)
        zrange = zranges[i].view(1, 2)
        K = Ks[i].view(1, 3)
        cub3D = shape_utils.box2D_to_cuboid3D(zrange, K, roi, im_size)
        txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)

        # image to camera coords
        if adjust_cs:
            verts[:, 0] = -verts[:, 0]
            verts[:, 1] = -verts[:, 1]

        # transform to destination size
        xz = verts[:, [0, 2]]
        yz = verts[:, [1, 2]]
        pxz = txz.inverse(xz.view(1, -1, 2)).squeeze(0)
        pyz = tyz.inverse(yz.view(1, -1, 2)).squeeze(0)
        verts = torch.stack([pxz[:, 0], pyz[:, 0], pxz[:, 1]], dim=1).to(
            device, dtype=torch.float32
        )

        new_verts.append(verts)
        new_faces.append(faces)

    return Meshes(verts=new_verts, faces=new_faces)


def load_unique_meshes(json_file, model_root):
    with PathManager.open(json_file, "r") as f:
        anns = json.load(f)["annotations"]
    # find unique models
    unique_models = []
    for obj in anns:
        model_type = obj["model"]
        if model_type not in unique_models:
            unique_models.append(model_type)
    # read unique models
    object_models = {}
    for model in unique_models:
        with PathManager.open(os.path.join(model_root, model), "rb") as f:
            mesh = load_obj(f, load_textures=False)
        object_models[model] = [mesh[0], mesh[1].verts_idx]
    return object_models
