# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
from detectron2.utils.file_io import PathManager
from scipy import io as sio

from cubercnn.util.projtransform import ProjectiveTransform


def cuboid3D_to_unitbox3D(cub3D):
    device = cub3D.device
    dst = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    dst = dst.view(1, 4, 2).expand(cub3D.shape[0], -1, -1)
    # for (x,z) plane
    txz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack(
                [cub3D[:, 0, 0], cub3D[:, 4, 0]], dim=1
            ),  # x3D_min at zmin, zmin
            torch.stack(
                [cub3D[:, 0, 1], cub3D[:, 4, 0]], dim=1
            ),  # x3D_max at zmin, zmin
            torch.stack(
                [cub3D[:, 2, 0], cub3D[:, 4, 1]], dim=1
            ),  # x3D_min at zmax, zmax
            torch.stack(
                [cub3D[:, 2, 1], cub3D[:, 4, 1]], dim=1
            ),  # x3D_max at zmax, zmax
        ],
        dim=1,
    )
    # src, dst both B x N(=4) x 2, map src(frustum) to dst(unit box)
    if not txz.estimate(src, dst):
        raise ValueError("Estimate failed")
    # for (y,z) plane
    tyz = ProjectiveTransform()
    src = torch.stack(
        [
            torch.stack([cub3D[:, 1, 0], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 1, 1], cub3D[:, 4, 0]], dim=1),
            torch.stack([cub3D[:, 3, 0], cub3D[:, 4, 1]], dim=1),
            torch.stack([cub3D[:, 3, 1], cub3D[:, 4, 1]], dim=1),
        ],
        dim=1,
    )
    if not tyz.estimate(src, dst):
        raise ValueError("Estimate failed")
    return txz, tyz


def box2D_to_cuboid3D(zranges, Ks, boxes, im_sizes):
    """
    Given a bbox2d and a min and max z-value, compute a 3D frustrum that contains the whole 3D space that objects in
    the 2D bbox could occupy
    Args: (Note: all args are for a list of meshes)
        zranges: min and max z-value of the corresponding GT mesh
        Ks: GT camera matrix
        boxes: 2D bounding boxes
        im_sizes: image size

    Returns:

    """
    # 3D box in camera space (not sure to be the bounding box, which is smallest)
    device = boxes.device
    if boxes.shape[0] != Ks.shape[0] != zranges.shape[0]:
        raise ValueError("Ks, boxes and zranges must have the same batch dimension")
    if zranges.shape[1] != 2:
        raise ValueError("zrange must have two entries per example")
    w, h = im_sizes.t()
    # here sx is focal length
    sx, px, py = Ks.t()
    sy = sx
    x1, y1, x2, y2 = boxes.t()  # bbox is handled correctly
    # transform 2d box from image coordinates to world coordinates (in camera space, unit pixel)
    # origin (image space): (0, 0) -> center (px, py)
    x1 = w - 1 - x1 - px
    y1 = h - 1 - y1 - py
    x2 = w - 1 - x2 - px
    y2 = h - 1 - y2 - py

    cub3D = torch.zeros((boxes.shape[0], 5, 2), device=device, dtype=torch.float32)
    for i in range(2):
        z = zranges[:, i]  # min / max z
        x3D_min = x2 * z / sx
        x3D_max = x1 * z / sx
        y3D_min = y2 * z / sy
        y3D_max = y1 * z / sy
        cub3D[:, i * 2 + 0, 0] = x3D_min
        cub3D[:, i * 2 + 0, 1] = x3D_max
        cub3D[:, i * 2 + 1, 0] = y3D_min
        cub3D[:, i * 2 + 1, 1] = y3D_max
    cub3D[:, 4, 0] = zranges[:, 0]
    cub3D[:, 4, 1] = zranges[:, 1]
    # for a single instance
    # [[x3D_min(zmin), x3D_max(zmin)], [y3D_min(zmin), y3D_max(zmin)],
    #  [x3D_min(zmax), x3D_max(zmax)], [y3D_min(zmax), y3D_max(zmax)], [zmin, zmax]]
    return cub3D


def transform_verts(verts, R, t):
    """
    Transforms verts with rotation R and translation t
    Inputs:
        - verts (tensor): of shape (N, 3)
        - R (tensor): of shape (3, 3) or None
        - t (tensor): of shape (3,) or None
    Outputs:
        - rotated_verts (tensor): of shape (N, 3)
    """
    rot_verts = verts.clone().t()
    if R is not None:
        assert R.dim() == 2
        assert R.size(0) == 3 and R.size(1) == 3
        rot_verts = torch.mm(R, rot_verts)
    if t is not None:
        assert t.dim() == 1
        assert t.size(0) == 3
        rot_verts = rot_verts + t.unsqueeze(1)
    rot_verts = rot_verts.t()  # transpose
    return rot_verts


def read_voxel(voxelfile):
    """
    We modify this function to accept both the .mat and .binvox format voxel files.

    Reads voxel and transforms it in the form of verts
    """
    shapenet = True
    if ".binvox" in voxelfile:
        verts = read_binvox_coords(open(voxelfile, "rb")).numpy()
    elif ".mat" in voxelfile:
        shapenet = False
        voxel = sio.loadmat(voxelfile)["voxel"]
        voxel = np.rot90(voxel, k=3, axes=(1, 2))
        verts = np.argwhere(voxel > 0).astype(np.float32, copy=False)
    else:
        raise NotImplementedError("Invalid voxel format!")

    # centering and normalization
    min = np.min(verts, axis=0)
    max = np.max(verts, axis=0)
    # ! check this again!
    verts = verts - (max - min) / 2  # centering
    scale1 = float(
        np.max(max - min)
    )  # longest side is scaled to 1 (same as meshes) (2)
    # scale2 = np.sqrt(np.max(np.sum(verts ** 2, axis=1))) * 2  # scale to fit inside unit cube (3)
    # scale3 = np.linalg.norm(max - min)  # difference in each direction is used (1,4)
    verts /= scale1  # scale
    verts = torch.tensor(verts, dtype=torch.float32)

    return verts


def read_binvox_coords(f, dtype=torch.float32):
    """
    This function in shapenet/utils/binvox_torch.py of the original project Mesh R-CNN

    Read a binvox file and return the indices of all nonzero voxels.

    This matches the behavior of binvox_rw.read_as_coord_array
    (https://github.com/dimatura/binvox-rw-py/blob/public/binvox_rw.py#L153)
    but this implementation uses torch rather than numpy, and is more efficient
    due to improved vectorization.

    I think that binvox_rw.read_as_coord_array actually has a bug; when converting
    linear indices into three-dimensional indices, they use floating-point
    division instead of integer division. We can reproduce their incorrect
    implementation by passing integer_division=False.

    Args:
      f (BinaryIO): A file pointer to the binvox file to read
      dtype: Datatype of the output tensor.

    Returns:
      coords (tensor): A tensor of shape (N, 3) where N is the number of nonzero voxels,
           and coords[i] = (x, y, z) gives the index of the ith nonzero voxel. If the
           voxel grid has shape (V, V, V) then we have 0 <= x, y, z < V.
    """
    size, translation, scale = _read_binvox_header(f)
    storage = torch.ByteStorage.from_buffer(f.read())
    data = torch.tensor([], dtype=torch.uint8)
    data.set_(source=storage)
    vals, counts = data[::2], data[1::2]
    idxs = _compute_idxs(vals, counts)
    x_idxs = idxs / (size * size)
    zy_idxs = idxs % (size * size)
    z_idxs = zy_idxs / size
    y_idxs = zy_idxs % size
    coords = torch.stack([x_idxs, y_idxs, z_idxs], dim=1)
    return coords.to(dtype)


def _compute_idxs(vals, counts):

    """
    This function in shapenet/utils/binvox_torch.py of the original project Mesh R-CNN
    Fast vectorized version of index computation"""
    # Consider an example where:
    # vals   = [0, 1, 0, 1, 1]
    # counts = [2, 3, 3, 2, 1]
    #
    # These values of counts and vals mean that the dense binary grid is:
    # [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    #
    # So the nonzero indices we want to return are:
    # [2, 3, 4, 8, 9, 10]

    # After the cumsum we will have:
    # end_idxs = [2, 5, 8, 10, 11]
    end_idxs = counts.cumsum(dim=0)

    # After masking and computing start_idx we have:
    # end_idxs   = [5, 10, 11]
    # counts     = [3,  2,  1]
    # start_idxs = [2,  8, 10]
    mask = vals == 1
    end_idxs = end_idxs[mask]
    counts = counts[mask].to(end_idxs)
    start_idxs = end_idxs - counts

    # We initialize delta as:
    # [2, 1, 1, 1, 1, 1]
    delta = torch.ones(counts.sum().item(), dtype=torch.int64)
    delta[0] = start_idxs[0]

    # We compute pos = [3, 5], val = [3, 0]; then delta is
    # [2, 1, 1, 4, 1, 1]
    pos = counts.cumsum(dim=0)[:-1]
    val = start_idxs[1:] - end_idxs[:-1]
    delta[pos] += val

    # A final cumsum gives the idx we want: [2, 3, 4, 8, 9, 10]
    idxs = delta.cumsum(dim=0)
    return idxs


def _read_binvox_header(f):

    """
    This function in shapenet/utils/binvox_torch.py of the original project Mesh R-CNN
    """

    # First line of the header should be "#binvox 1"
    line = f.readline().strip()
    if line != b"#binvox 1":
        raise ValueError("Invalid header (line 1)")

    # Second line of the header should be "dim [int] [int] [int]"
    # and all three int should be the same
    line = f.readline().strip()
    if not line.startswith(b"dim "):
        raise ValueError("Invalid header (line 2)")
    dims = line.split(b" ")
    try:
        dims = [int(d) for d in dims[1:]]
    except ValueError:
        raise ValueError("Invalid header (line 2)")
    if len(dims) != 3 or dims[0] != dims[1] or dims[0] != dims[2]:
        raise ValueError("Invalid header (line 2)")
    size = dims[0]

    # Third line of the header should be "translate [float] [float] [float]"
    line = f.readline().strip()
    if not line.startswith(b"translate "):
        raise ValueError("Invalid header (line 3)")
    translation = line.split(b" ")
    if len(translation) != 4:
        raise ValueError("Invalid header (line 3)")
    try:
        translation = tuple(float(t) for t in translation[1:])
    except ValueError:
        raise ValueError("Invalid header (line 3)")

    # Fourth line of the header should be "scale [float]"
    line = f.readline().strip()
    if not line.startswith(b"scale "):
        raise ValueError("Invalid header (line 4)")
    line = line.split(b" ")
    if not len(line) == 2:
        raise ValueError("Invalid header (line 4)")
    scale = float(line[1])

    # Fifth line of the header should be "data"
    line = f.readline().strip()
    if not line == b"data":
        raise ValueError("Invalid header (line 5)")

    return size, translation, scale
