# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.
"""

from torch import nn
import torch.nn.functional as F
from .util import (
    Hourglass, kp2gaussian, make_coordinate_grid, AntiAliasInterpolation2d,
    ResBlock3d, UpBlock3d, SameBlock2d, DownBlock3d,
    GridSample3DEquivalent
)
from .dense_motion import DenseMotionNetwork


def deform_input(inp, deformation):
    bs, d_in, h_in, w_in, _ = deformation.shape
    _, _, d, h, w = inp.shape

    if d_in != d or h_in != h or w_in != w:
        # Create a grid for the target size to resample the deformation field
        identity_grid = make_coordinate_grid((d, h, w), ref=inp).view(1, d, h, w, 3)
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1)

        # Reshape deformation to be a (N, C, D, H, W) tensor for sampling
        deformation_as_input = deformation.permute(0, 4, 1, 2, 3)

        # Use our ONNX-safe grid_sample to resize the deformation field
        resized_deformation = GridSample3DEquivalent(align_corners=False)(
            deformation_as_input, identity_grid
        )

        # Permute back to the original (N, D, H, W, C) format
        deformation = resized_deformation.permute(0, 2, 3, 4, 1)

    return GridSample3DEquivalent(align_corners=False)(inp, deformation)


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,
        block_expansion,
        max_features,
        num_down_blocks,
        reshape_channel,
        estimate_occlusion_map=False,
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,
                feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
            )
        else:
            self.dense_motion_network = None

        self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    def forward(self, feature_3d, kp_driving, kp_source):
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']  # Bx16x64x64x3
            out = deform_input(feature_3d, deformation)  # Bx32x16x64x64

            bs, c, d, h, w = out.shape  # Bx32x16x64x64
            out = out.view(bs, c * d, h, w)  # -> Bx512x64x64
            out = self.third(out)  # -> Bx256x64x64
            out = self.fourth(out)  # -> Bx256x64x64

            if self.flag_use_occlusion_map and (occlusion_map is not None):
                out = out * occlusion_map

        ret_dct = {
            'occlusion_map': occlusion_map,
            'deformation': deformation,
            'out': out,
        }

        return ret_dct
