# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import OrderedDict

import einops
import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.registry import Registry
from pytorch3d.loss import chamfer_distance, mesh_edge_loss
from pytorch3d.ops import (
    GraphConv,
    SubdivideMeshes,
    sample_points_from_meshes,
    vert_align,
)
from pytorch3d.ops.graph_conv import gather_scatter, gather_scatter_python  # for 0N-GCN
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    ChebConv,
    DNAConv,
    EdgeCNN,
    EGConv,
    FeaStConv,
    FiLMConv,
    GATv2Conv,
    GCNConv,
)
from torch_geometric.nn import GraphConv as GraphConvPyG  # GPSConv,
from torch_geometric.nn import ResGatedGraphConv, SAGEConv, SuperGATConv

from cubercnn.structures.mesh import MeshInstances, batch_crop_meshes_within_box

ROI_MESH_HEAD_REGISTRY = Registry("ROI_MESH_HEAD")


def mesh_rcnn_loss(
    pred_meshes,
    instances,
    Ks,
    loss_weights=None,
    gt_num_samples=5000,
    pred_num_samples=5000,
    gt_coord_thresh=None,
):
    """
    Compute the mesh prediction loss defined in the Mesh R-CNN paper.

    Args:
        pred_meshes (list of Meshes): A list of K Meshes. Each entry contains B meshes,
            where B is the total number of predicted meshes in all images.
            K is the number of refinements
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the pred_meshes.
            The ground-truth labels (class, box, mask, ...) associated with each instance
            are stored in fields.
        Ks: camera matrix adjusted for scaled bbox, per image
        loss_weights (dict): Contains the weights for the different losses, e.g.
            loss_weights = {'champfer': 1.0, 'normals': 0.0, 'edge': 0.2}
        gt_num_samples (int): The number of points to sample from gt meshes
        pred_num_samples (int): The number of points to sample from predicted meshes
        gt_coord_thresh (float): A threshold value over which the batch is ignored
    Returns:
        mesh_loss (Tensor): A scalar tensor containing the loss.
    """
    if not isinstance(pred_meshes, list):
        raise ValueError("Expecting a list of Meshes")

    assert sum([len(i) for i in instances]) == len(
        pred_meshes[0]
    ), "Number of instances doesn't match mesh predictions"
    assert len({len(i) for i in pred_meshes}) == 1, "pred_meshes have different lengths"
    assert len(instances) == len(Ks)

    gt_verts, gt_faces = [], []
    for instances_per_image, K_per_image in zip(instances, Ks):
        if len(instances_per_image) == 0:
            continue

        gt_mesh_per_image = batch_crop_meshes_within_box(
            instances_per_image.gt_meshes,
            instances_per_image.proposal_boxes_scaled.tensor,
            einops.repeat(K_per_image, "c -> n c", n=len(instances_per_image)),
        ).to(device=pred_meshes[0].device)
        gt_verts.extend(gt_mesh_per_image.verts_list())
        gt_faces.extend(gt_mesh_per_image.faces_list())

    if len(gt_verts) == 0:
        return None, None

    gt_meshes = Meshes(verts=gt_verts, faces=gt_faces)
    gt_valid = gt_meshes.valid  # num_faces > 0
    gt_sampled_verts, gt_sampled_normals = sample_points_from_meshes(
        gt_meshes, num_samples=gt_num_samples, return_normals=True
    )

    all_loss_chamfer = []
    all_loss_normals = []
    all_loss_edge = []
    for pred_mesh in pred_meshes:
        pred_sampled_verts, pred_sampled_normals = sample_points_from_meshes(
            pred_mesh, num_samples=pred_num_samples, return_normals=True
        )
        # behave as a mask for valid
        weights = (pred_mesh.valid * gt_valid).to(dtype=torch.float32)
        # chamfer loss
        loss_chamfer, loss_normals = chamfer_distance(
            pred_sampled_verts,
            gt_sampled_verts,
            x_normals=pred_sampled_normals,
            y_normals=gt_sampled_normals,
            weights=weights,
        )

        # chamfer loss
        loss_chamfer = loss_chamfer * loss_weights["chamfer"]
        all_loss_chamfer.append(loss_chamfer)
        # normal loss
        loss_normals = loss_normals * loss_weights["normals"]
        all_loss_normals.append(loss_normals)
        # mesh edge regularization
        loss_edge = mesh_edge_loss(pred_mesh)
        loss_edge = loss_edge * loss_weights["edge"]
        all_loss_edge.append(loss_edge)

    loss_chamfer = sum(all_loss_chamfer)
    loss_normals = sum(all_loss_normals)
    loss_edge = sum(all_loss_edge)

    # if the rois are bad, the target verts can be arbitrarily large
    # causing exploding gradients. If this is the case, ignore the batch
    if (
        gt_coord_thresh and gt_sampled_verts.abs().max() > gt_coord_thresh
    ):  # max abs x/y/z coordinate > 5.0
        loss_chamfer = loss_chamfer * 0.0
        loss_normals = loss_normals * 0.0
        loss_edge = loss_edge * 0.0

    return loss_chamfer, loss_normals, loss_edge, gt_meshes


def mesh_rcnn_inference(pred_meshes, pred_instances):
    """
    Return the predicted mesh for each predicted instance

    Args:
        pred_meshes (Meshes): A class of Meshes containing B meshes, where B is
            the total number of predictions in all images.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. Each Instances must have field "pred_classes".

    Returns:
        None. pred_instances will contain an extra "pred_meshes" field storing the meshes
    """
    num_boxes_per_image = [len(i) for i in pred_instances]
    pred_meshes = pred_meshes.split(num_boxes_per_image)

    for pred_mesh, instances in zip(pred_meshes, pred_instances):
        # NOTE do not save the Meshes object; pickle dumps become inefficient
        if pred_mesh.isempty():
            continue
        verts_list = pred_mesh.verts_list()
        faces_list = pred_mesh.faces_list()
        instances.pred_meshes = MeshInstances(
            [(v, f) for (v, f) in zip(verts_list, faces_list)]
        )


class ZNGraphConv(nn.Module):
    """We re-implement zero-neighbor graph convolution layer proposed in GEOMetrics based on Pytorch3D functions."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: str = "normal",
        with_neighbor: float = 0.5,
        directed: bool = False,
    ):
        """
        Args:
            input_dim: Number of input features per vertex.
            output_dim: Number of output features per vertex.
            init: Weight initialization method. Can be one of ['zero', 'normal'].
            directed: Bool indicating if edges in the graph are directed.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.support_dim = int(output_dim * with_neighbor)
        self.directed = directed
        self.w0 = nn.Linear(input_dim, output_dim)
        self.w1 = nn.Linear(input_dim, self.support_dim)

        if init == "normal":
            nn.init.normal_(self.w0.weight, mean=0, std=0.01)
            nn.init.normal_(self.w1.weight, mean=0, std=0.01)
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        elif init == "zero":
            self.w0.weight.data.zero_()
            self.w1.weight.data.zero_()
        else:
            raise ValueError('Invalid GraphConv initialization "%s"' % init)

    def forward(self, verts, edges):
        """
        Args:
            verts: FloatTensor of shape (V, input_dim) where V is the number of
                vertices and input_dim is the number of input features
                per vertex. input_dim has to match the input_dim specified
                in __init__.
            edges: LongTensor of shape (E, 2) where E is the number of edges
                where each edge has the indices of the two vertices which
                form the edge.

        Returns:
            out: FloatTensor of shape (V, output_dim) where output_dim is the
            number of output features per vertex.
        """
        if verts.is_cuda != edges.is_cuda:
            raise ValueError("verts and edges tensors must be on the same device.")
        if verts.shape[0] == 0:
            # empty graph.
            return verts.sum() * 0.0

        verts_w0 = self.w0(verts)  # (V, output_dim)
        verts_w1 = self.w1(verts)  # (V, support_dim)

        if torch.cuda.is_available() and verts.is_cuda and edges.is_cuda:
            neighbor_sums = gather_scatter(verts_w1, edges, self.directed)
        else:
            neighbor_sums = gather_scatter_python(
                verts_w1, edges, self.directed
            )  # (V, support_dim)

        device, dtype = neighbor_sums.device, neighbor_sums.dtype
        verts_num = neighbor_sums.shape[0]

        zero_neighbor = torch.zeros(
            [verts_num, self.output_dim - self.support_dim], dtype=dtype, device=device
        )

        neighbor_sums = torch.cat((neighbor_sums, zero_neighbor), dim=1)

        # Add neighbor features to each vertex's features.
        out = verts_w0 + neighbor_sums
        return out

    def __repr__(self):
        Din, Dout, directed = self.input_dim, self.output_dim, self.directed
        return "ZNGraphConv(%d -> %d, directed=%r)" % (Din, Dout, directed)


class ResGraphConv(nn.Module):

    """We reimplement the residual GCN based on the Mesh R-CNN paper (described in ShapeNet subsection)."""

    def __init__(self, hidden_dim, init="normal"):
        super(ResGraphConv, self).__init__()

        self.gconv1 = GraphConv(hidden_dim, hidden_dim, init=init, directed=False)
        self.gconv2 = GraphConv(hidden_dim, hidden_dim, init=init, directed=False)

    def forward(self, input_feats, edges_packed):
        vert_feats = F.relu(self.gconv1(input_feats, edges_packed), inplace=True)
        vert_feats = F.relu(self.gconv2(vert_feats, edges_packed), inplace=True)

        return (input_feats + vert_feats) * 0.5


class MeshRefinementStage(nn.Module):
    def __init__(
        self,
        img_feat_dim,
        vert_feat_dim,
        hidden_dim,
        stage_depth,
        zn_neighbor=None,
        conv_type="Graph",
        gconv_init="normal",
    ):
        """
        Modified to be compatible to 0N-GCN and residual GCN.

        Args:
          img_feat_dim: Dimension of features we will get from vert_align
          vert_feat_dim: Dimension of vert_feats we will receive from the
                        previous stage; can be 0
          hidden_dim: Output dimension for graph-conv layers
          stage_depth: Number of graph-conv layers to use
          gconv_init: How to initialize graph-conv layers
        """
        super(MeshRefinementStage, self).__init__()

        self.conv_type = conv_type
        self.residual = conv_type == "ResGraph"
        self.PYG_CONVS = [
            "GCN",
            "FeaSt",  # not good
            "GATv2",
            "GATv2e",
            "DNA",  # doesn't work
            "SuperGAT",
            "ResGatedGraph",
            "EG",
            "FiLM",
            "GPS",
            "SAGE",
            "GraphPyG",
            "EdgeCNN",
            "Cheb",
        ]

        # fc layer to reduce feature dimension
        verts_pos_dim = 3
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        # deform layer
        if self.residual:
            self.linear_project = nn.Linear(
                hidden_dim + vert_feat_dim + verts_pos_dim, hidden_dim
            )
            nn.init.normal_(self.linear_project.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.linear_project.bias, 0)
            self.verts_offset = GraphConv(
                hidden_dim, verts_pos_dim, init=gconv_init, directed=False
            )
        else:
            self.verts_offset = nn.Linear(hidden_dim + verts_pos_dim, verts_pos_dim)

        # graph convs
        self.gconvs = nn.ModuleList()
        for i in range(stage_depth):
            if i == 0:
                input_dim = hidden_dim + vert_feat_dim + verts_pos_dim  # 128 + 128 + 3
            else:
                input_dim = hidden_dim + verts_pos_dim

            if self.conv_type == "Graph":
                gconv = GraphConv(
                    input_dim, hidden_dim, init=gconv_init, directed=False
                )
            elif self.residual:
                gconv = ResGraphConv(hidden_dim, init=gconv_init)
            elif self.conv_type == "GCN":
                gconv = GCNConv(input_dim, hidden_dim)
            elif self.conv_type == "FeaSt":
                gconv = FeaStConv(input_dim, hidden_dim)
            elif self.conv_type == "GATv2":
                gconv = GATv2Conv(input_dim, hidden_dim)
            elif self.conv_type == "GATv2e":
                gconv = GATv2Conv(
                    input_dim,
                    hidden_dim,
                    heads=2,
                    dropout=0.2,
                    concat=False,
                )
            elif self.conv_type == "DNA":
                gconv = DNAConv(input_dim, hidden_dim)
            elif self.conv_type == "SuperGAT":
                gconv = SuperGATConv(input_dim, hidden_dim)
            elif self.conv_type == "ResGatedGraph":
                gconv = ResGatedGraphConv(input_dim, hidden_dim)
            elif self.conv_type == "FiLM":
                gconv = FiLMConv(input_dim, hidden_dim)
            elif self.conv_type == "EG":
                gconv = EGConv(input_dim, hidden_dim)
            elif self.conv_type == "SAGE":
                gconv = SAGEConv(input_dim, hidden_dim)
            elif self.conv_type == "GraphPyG":
                gconv = GraphConvPyG(input_dim, hidden_dim)
            elif self.conv_type == "EdgeCNN":
                gconv = EdgeCNN(input_dim, hidden_dim, 1)
            elif self.conv_type == "Cheb":
                gconv = ChebConv(input_dim, hidden_dim, K=10)
            elif self.conv_type == "ZNG" and zn_neighbor > 0:
                gconv = ZNGraphConv(
                    input_dim,
                    hidden_dim,
                    init=gconv_init,
                    with_neighbor=zn_neighbor,
                    directed=False,
                )
            else:
                raise ValueError(f"Unknown graph conv type {self.conv_type}")
            self.gconvs.append(gconv)

        # initialization
        nn.init.normal_(self.bottleneck.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.bottleneck.bias, 0)

        if not self.residual:
            nn.init.zeros_(self.verts_offset.weight)
            nn.init.constant_(self.verts_offset.bias, 0)

    def forward(self, x, mesh, vert_feats=None):  # hidden_dim = 128
        # vert_feats: embedding from before (if available) -> we also use img_feats at the new (moved) position
        img_feats = vert_align(x, mesh, return_packed=True, padding_mode="border")
        img_feats = F.relu(
            self.bottleneck(img_feats)
        )  # 256 -> hidden_dim; reduce feature dimension of image features
        if vert_feats is None:  # first refinement stage
            # hidden_dim+3; add vertex positions to image features, when no vert_feats available
            vert_feats = torch.cat((img_feats, mesh.verts_packed()), dim=1)
        else:  # subsequent refinement stages
            # hidden_dim*2+3; use vert_feats and img_feats and add vertex positions
            vert_feats = torch.cat((vert_feats, img_feats, mesh.verts_packed()), dim=1)

        if self.residual:
            vert_feats = F.relu(self.linear_project(vert_feats))

        for graph_conv in self.gconvs:
            if self.residual:
                vert_feats = graph_conv(vert_feats, mesh.edges_packed())
            else:
                if self.conv_type in self.PYG_CONVS:
                    vert_feats_nopos = F.relu(
                        graph_conv(vert_feats, mesh.edges_packed().T)
                    )
                else:
                    vert_feats_nopos = F.relu(
                        graph_conv(vert_feats, mesh.edges_packed())
                    )
                vert_feats = torch.cat((vert_feats_nopos, mesh.verts_packed()), dim=1)

        offset_args = [vert_feats]
        if self.residual:
            offset_args.append(
                mesh.edges_packed()
            )  # also add positions for prediction, not only vert_feats
            vert_feats_nopos = vert_feats

        # hidden_dim+3->3; use all features to predict offset only
        deform = torch.tanh(self.verts_offset(*offset_args))
        mesh = mesh.offset_verts(deform)  # apply offset to mesh
        return (
            mesh,
            vert_feats_nopos,
        )  # return updated mesh and current vert_features for next refinement stage


@ROI_MESH_HEAD_REGISTRY.register()
class MeshRCNNGraphConvHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine layers.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MeshRCNNGraphConvHead, self).__init__()

        # fmt: off
        num_stages         = cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES
        num_graph_convs    = cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS  # per stage
        graph_conv_dim     = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM
        graph_conv_init    = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT
        zn_neighbor        = cfg.MODEL.ROI_MESH_HEAD.ZN_GCONV
        conv_type          = cfg.MODEL.ROI_MESH_HEAD.CONV_TYPE
        input_channels     = input_shape.channels
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(num_stages):
            vert_feat_dim = 0 if i == 0 else graph_conv_dim
            stage = MeshRefinementStage(
                input_channels,
                vert_feat_dim,
                graph_conv_dim,
                num_graph_convs,
                zn_neighbor=zn_neighbor,
                conv_type=conv_type,
                gconv_init=graph_conv_init,
            )
            self.stages.append(stage)

    def forward(self, x, mesh):
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]

        meshes = []
        vert_feats = None  # first stage has no input vert_feats yet
        for stage in self.stages:
            mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
            meshes.append(mesh)
        return meshes


# subdivide layer is the pixel2mesh version unpooling layer for upsampling
@ROI_MESH_HEAD_REGISTRY.register()
class MeshRCNNGraphConvSubdHead(nn.Module):
    """
    A mesh head with vert align, graph conv layers and refine and subdivide layers.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super(MeshRCNNGraphConvSubdHead, self).__init__()

        # fmt: off
        self.num_stages    = cfg.MODEL.ROI_MESH_HEAD.NUM_STAGES
        num_graph_convs    = cfg.MODEL.ROI_MESH_HEAD.NUM_GRAPH_CONVS  # per stage
        graph_conv_dim     = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_DIM
        graph_conv_init    = cfg.MODEL.ROI_MESH_HEAD.GRAPH_CONV_INIT
        input_channels     = input_shape.channels
        # fmt: on

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            vert_feat_dim = 0 if i == 0 else graph_conv_dim
            stage = MeshRefinementStage(
                input_channels,
                vert_feat_dim,
                graph_conv_dim,
                num_graph_convs,
                gconv_init=graph_conv_init,
            )
            self.stages.append(stage)

    def forward(self, x, mesh):
        if x.numel() == 0 or mesh.isempty():
            return [Meshes(verts=[], faces=[])]

        meshes = []
        vert_feats = None
        for i, stage in enumerate(self.stages):
            mesh, vert_feats = stage(x, mesh, vert_feats=vert_feats)
            meshes.append(mesh)
            if i < self.num_stages - 1:
                subdivide = SubdivideMeshes()
                mesh, vert_feats = subdivide(mesh, feats=vert_feats)
        return meshes


def build_mesh_head(cfg, input_shape):
    name = cfg.MODEL.ROI_MESH_HEAD.NAME
    return ROI_MESH_HEAD_REGISTRY.get(name)(cfg, input_shape)
