# Copyright (c) Meta Platforms, Inc. and affiliates
import copy
import json
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import DatasetMapper, MetadataCatalog, detection_utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints
from detectron2.utils.file_io import PathManager
from joblib import Memory
from pytorch3d.io import load_obj

import cubercnn.util.shape as shape_utils
from cubercnn.structures import MeshInstances


class DatasetMapper3D(DatasetMapper):
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        dataset_names=None,
    ):

        super().__init__(
            is_train,
            augmentations=augmentations,
            image_format=image_format,
            use_instance_mask=use_instance_mask,
            use_keypoint=use_keypoint,
            instance_mask_format=instance_mask_format,
            keypoint_hflip_indices=keypoint_hflip_indices,
            precomputed_proposal_topk=precomputed_proposal_topk,
            recompute_boxes=recompute_boxes,
        )
        _all_meshes = {}
        for dataset_name in dataset_names:
            # MetadataCatalog registered in datasets/shapenet.py
            json_file = MetadataCatalog.get(dataset_name).json_file
            model_root = MetadataCatalog.get(dataset_name).image_root
            # Load models
            print("Loading models from {}...".format(dataset_name))
            time_start = time.time()
            cachedir = f"cache/{dataset_name}"
            mem = Memory(cachedir)
            load_unique_meshes_cached = mem.cache(load_unique_meshes)
            dataset_mesh_models = load_unique_meshes_cached(json_file, model_root)
            _all_meshes.update(dataset_mesh_models)
            print(
                f"Unique objects loaded: {len(dataset_mesh_models)} in {time.time() - time_start}s"
            )
        self._all_meshes = copy.deepcopy(_all_meshes)
        del _all_meshes

    def __call__(self, dataset_dict):

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = detection_utils.read_image(
            dataset_dict["file_name"], format=self.image_format
        )
        detection_utils.check_image_size(dataset_dict, image)

        transforms = None

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # no need for additoinal processing at inference
        if not self.is_train:
            for anno in dataset_dict["annotations"]:
                anno["area"] = anno.get("area", 0)
            return dataset_dict

        if "annotations" in dataset_dict:

            dataset_id = dataset_dict["dataset_id"]
            K = np.array(dataset_dict["K"])

            unknown_categories = self.dataset_id_to_unknown_cats[dataset_id]

            # transform and pop off annotations
            annos = []
            for obj in dataset_dict.pop("annotations"):
                if obj.get("iscrowd", 0) == 0:
                    obj["mesh"] = self._all_meshes[obj["model"]]
                    annos.append(transform_instance_annotations(obj, transforms, K=K))

            # convert to instance format
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(
                instances
            )

        return dataset_dict


def _process_mesh(mesh, transforms, R=None, t=None):
    # clone mesh
    verts, faces = mesh
    # transform vertices to camera coordinate system
    verts = shape_utils.transform_verts(verts, R, t)

    if transforms is None:
        return verts, faces

    assert all(
        isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
        for t in transforms.transforms
    )
    for t in transforms.transforms:
        if isinstance(t, T.HFlipTransform):
            verts[:, 0] = -verts[:, 0]
        elif isinstance(t, T.ResizeTransform):
            verts = t.apply_coords(verts)
        elif isinstance(t, T.NoOpTransform):
            pass
        else:
            raise ValueError("Transform {} not recognized".format(t))
    return verts, faces


"""
Cached for mirroring annotations
"""
_M1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
_M2 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])


def load_unique_meshes(json_file, model_root, normalize=False):
    """
    Loading meshes.
    Note: Before, we had options to center and normalize the meshes. This was removed, since it might entice people
    to not do it beforehand (which might lead to issues).
    Args:
        json_file:
        model_root:
        normalize:

    Returns:

    """
    with PathManager.open(json_file, "r") as f:
        anns = json.load(f)["annotations"]
    # find unique models
    unique_models = []
    for obj in anns:
        model_type = obj["model"]
        if model_type not in unique_models:
            unique_models.append(model_type)
    # Check if parcel3d dataset (needed before, since p3d needed to be centered)
    # read unique models
    object_models = {}
    for model in unique_models:
        verts, faces = load_mesh(Path(os.path.join(model_root, model)))
        object_models[model] = [verts, faces]
    return object_models


def load_mesh(path: Path):
    with PathManager.open(path.as_posix(), "rb") as f:
        mesh = load_obj(f, load_textures=False)
    return [mesh[0], mesh[1].verts_idx]


def transform_instance_annotations(annotation, transforms, *, K):

    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)

    # bbox is 1d (per-instance bounding box)
    bbox = BoxMode.convert(
        annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS
    )
    if transforms is not None:
        bbox = transforms.apply_box(np.array([bbox]))[0]

    annotation["bbox"] = bbox
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    # load mesh
    if annotation.get("mesh", None) is None:
        annotation["mesh"] = load_mesh(Path(annotation["model"]))

    if annotation["center_cam"][2] != 0:

        # project the 3D box annotation XYZ_3D to screen
        point3D = annotation["center_cam"]
        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        if transforms is not None:
            # apply coords transforms to 2D box
            annotation["center_cam_proj"][0:2] = transforms.apply_coords(
                point2D[np.newaxis][:, :2]
            )[0].tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]

        if annotation["ignore"]:
            # all keypoints marked as not visible
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:

            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        # in place
        if transforms is not None:
            transforms.apply_coords(keypoints[:, :2])
        annotation["keypoints"] = keypoints.tolist()

        if transforms is not None:
            # manually apply mirror for pose
            for transform in transforms:

                # horrizontal flip?
                if isinstance(transform, T.HFlipTransform):

                    pose = _M1 @ np.array(annotation["pose"]) @ _M2
                    annotation["pose"] = pose.tolist()
                    annotation["R_cam"] = pose.tolist()

        # transform Mesh
        # cs_transform = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]).float()
        annotation["mesh"] = _process_mesh(
            annotation["mesh"],
            transforms,
            R=torch.tensor(annotation["model_R"]),
            t=torch.tensor(annotation["model_t"]),
            # R=cs_transform @ torch.tensor(annotation["model_R"]),
            # t=cs_transform @ torch.tensor(annotation["model_t"]),
        )

    return annotation


def annotations_to_instances(annos, image_size, unknown_categories):

    # init
    target = Instances(image_size)

    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor(
        [int(obj["category_id"]) for obj in annos], dtype=torch.int64
    )
    target.gt_boxes = Boxes(
        [
            BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            for obj in annos
        ]
    )
    target.gt_boxes3D = torch.FloatTensor(
        [
            anno["center_cam_proj"] + anno["dimensions"] + anno["center_cam"]
            for anno in annos
        ]
    )
    target.gt_poses = torch.FloatTensor([anno["pose"] for anno in annos])

    n = len(target.gt_classes)

    # do keypoints?
    target.gt_keypoints = Keypoints(
        torch.FloatTensor([anno["keypoints"] for anno in annos])
    )

    gt_unknown_category_mask = torch.zeros(max(unknown_categories) + 1, dtype=bool)
    gt_unknown_category_mask[torch.tensor(list(unknown_categories))] = True

    # include available category indices as tensor with GTs
    target.gt_unknown_category_mask = gt_unknown_category_mask.unsqueeze(0).repeat(
        [n, 1]
    )

    # Load GT for Meshes
    if len(annos) and "K" in annos[0]:
        K = [torch.tensor(obj["K"]) for obj in annos]
        target.gt_K = torch.stack(K, dim=0)
    if len(annos) and "mesh" in annos[0]:
        meshes = [obj["mesh"] for obj in annos]
        target.gt_meshes = MeshInstances(meshes)

    return target
