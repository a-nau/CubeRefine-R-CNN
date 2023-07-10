# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple

import torch
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures import Meshes

import cubercnn.util.shape as shape_utils

"""
We mainly add the function normalize_mesh in the script for view rendering in the master's thesis.
"""

DEFAULT_BBOX3D_FACES = torch.tensor(
    [
        [0, 2, 3],
        [1, 2, 3],  # front
        [4, 6, 7],
        [5, 6, 7],  # back
        [0, 2, 4],
        [6, 2, 4],  # top
        [3, 1, 7],
        [5, 1, 7],  # bottom
        [5, 6, 1],
        [2, 6, 1],  # left
        [0, 3, 4],
        [7, 3, 4],  # right
    ],
    dtype=torch.int64,
)


def normalize_mesh(mesh):
    """
    This function normalize the mesh to fit in a unit sphere, so we can use it for views rendering.
    :param mesh: Meshes object in Pytorch3D
    :return:
    """

    bbox = mesh.get_bounding_boxes()
    long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]
    scale = 1.0 / long_edge
    center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2.0
    num_vertices = mesh.verts_list()[0].shape[0]
    offset = -center.expand(num_vertices, -1)
    mesh = mesh.offset_verts(offset).scale_verts(scale)
    return mesh


def normalize_mesh_from_verts_and_faces(verts, faces):
    """
    This function normalize the mesh to fit in a unit sphere, so we can use it for views rendering.
    :param mesh: Meshes object in Pytorch3D
    :return:
    """
    verts = torch.unsqueeze(verts, dim=0)
    faces = torch.unsqueeze(faces, dim=0)
    mesh = Meshes(verts=verts, faces=faces)
    bbox = mesh.get_bounding_boxes()
    long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]
    scale = 1.0 / long_edge
    center = (bbox[:, :, 0] + bbox[:, :, 1]) / 2.0
    num_vertices = mesh.verts_list()[0].shape[0]
    offset = -center.expand(num_vertices, len(mesh), 3).reshape(-1, 3)
    mesh = mesh.offset_verts(offset).scale_verts(scale)
    return mesh


def batch_crop_meshes_within_box(meshes, boxes, Ks):
    """
    Batched version of :func:`crop_mesh_within_box`.

    We have a bbox2d, that converted into 3D becomes a frustrum (in world space). We then compute the projective
    transformations to convert this frustrum into a unit cube. These transformations are applied to the GT mesh
    vertices, i.e. we obtain a transformed GT mesh that is in coordinates relative to the 2D bbox.
    ? The transformed mesh is adjusted for the fact that objects become smaller, as they recede from the camera.

    Args:
        meshes (MeshInstances): store N meshes for an image
        boxes (Tensor): store N boxes corresponding to the meshes.
        Ks (Tensor): store N camera matrices

    Returns:
        Meshes: A Meshes structure of N meshes where N is the number of
                predicted boxes for this image.
    """
    device = boxes.device
    im_sizes = Ks[:, 1:] * 2.0
    verts = torch.stack([mesh[0] for mesh in meshes], dim=0)
    zranges = torch.stack([verts[:, :, 2].min(1)[0], verts[:, :, 2].max(1)[0]], dim=1)
    # Convert 2D bbox to 3D space (frustrum) that 2D objects could maximally occupy
    cub3D = shape_utils.box2D_to_cuboid3D(zranges, Ks, boxes.clone(), im_sizes)
    # Compute projective transformation that converts 3D frustrum to unit cube
    txz, tyz = shape_utils.cuboid3D_to_unitbox3D(cub3D)
    x, y, z = verts.split(1, dim=2)
    xz = torch.cat([x, z], dim=2)
    yz = torch.cat([y, z], dim=2)
    # Apply projective transformations to vertex coordinates
    pxz = txz(xz)
    pyz = tyz(yz)
    new_verts = torch.stack([pxz[:, :, 0], pyz[:, :, 0], pxz[:, :, 1]], dim=2)

    # align to image
    # flip x, y
    new_verts[:, :, 0] = -new_verts[:, :, 0]
    new_verts[:, :, 1] = -new_verts[:, :, 1]

    verts_list = [new_verts[i] for i in range(boxes.shape[0])]
    faces_list = [mesh[1] for mesh in meshes]

    return Meshes(verts=verts_list, faces=faces_list).to(device=device)


def gt_bbox3d_as_init(
    proposals: List[Instances], level=4, device=None, use_bbox_gt=True
):
    meshes = {"verts": [], "faces": []}
    for proposals_per_image in proposals:
        image_meshes = []
        keypoints3d_per_image = proposals_per_image.gt_keypoints3d
        keypoints3d = keypoints3d_per_image[0]  # GT keypoints are always the same
        mesh = gt_bbox3d_as_init_per_instance(keypoints3d, level, device)
        for i in range(keypoints3d_per_image.size(dim=0)):
            image_meshes.append(mesh.verts_list() + mesh.faces_list())
        bboxes = (
            proposals_per_image.gt_boxes.tensor
            if use_bbox_gt
            else proposals_per_image.proposal_boxes.tensor
        )
        image_meshes = batch_crop_meshes_within_box(
            MeshInstances(image_meshes), bboxes, proposals_per_image.gt_K
        )
        meshes["verts"].extend(image_meshes.verts_list())
        meshes["faces"].extend(image_meshes.faces_list())
    meshes = Meshes(**meshes)
    return meshes


def gt_bbox3d_as_init_per_batch(verts, faces, level=1, device=None):
    """
    Load bbox3d from GT. N x 8 x 3 Input
    Subdivide it's mesh.
    Crop it within the bbox2d.
    Use as Voxelbranch output for test purposes.
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        pass
    else:
        mesh = gt_bbox3d_as_init_per_batch(verts, faces, level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()
        faces = mesh.faces_list()
    return Meshes(verts=verts, faces=faces)


def gt_bbox3d_as_init_per_instance(keypoints, level=1, device=None):
    """
    Load bbox3d from GT.
    Subdivide it's mesh.
    Crop it within the bbox2d.
    Use as Voxelbranch output for test purposes.
    """
    if device is None:
        device = torch.device("cpu")
    if level < 0:
        raise ValueError("level must be >= 0.")
    if level == 0:
        verts = keypoints
        faces = DEFAULT_BBOX3D_FACES.to(device)
    else:
        mesh = gt_bbox3d_as_init_per_instance(keypoints, level - 1, device)
        subdivide = SubdivideMeshes()
        mesh = subdivide(mesh)
        verts = mesh.verts_list()[0]
        faces = mesh.faces_list()[0]
    return Meshes(verts=[verts], faces=[faces])


class MeshInstances:
    """
    Class to hold meshes of varying topology to interface with Instances
    """

    def __init__(self, meshes):
        assert isinstance(meshes, list)
        assert torch.is_tensor(meshes[0][0])
        assert torch.is_tensor(meshes[0][1])
        self.data = meshes

    def to(self, device):
        to_meshes = [(mesh[0].to(device), mesh[1].to(device)) for mesh in self]
        return MeshInstances(to_meshes)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_data = [self.data[item]]
        else:
            # advanced indexing on a single dimension
            selected_data = []
            if isinstance(item, torch.Tensor) and (
                item.dtype == torch.uint8 or item.dtype == torch.bool
            ):
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_data.append(self.data[i])
        return MeshInstances(selected_data)

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}) ".format(len(self))
        return s
