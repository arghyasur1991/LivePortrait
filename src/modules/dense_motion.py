# coding: utf-8

"""
The module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
"""

from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, Conv3DEquivalent, BatchNorm3DEquivalent, make_coordinate_grid, kp2gaussian, GridSample3DEquivalent


class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, num_kp, feature_channel, reshape_depth, compress, estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)  # ~60+G

        self.mask = Conv3DEquivalent(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)  # 65G! NOTE: computation cost is large
        self.compress = Conv3DEquivalent(feature_channel, compress, kernel_size=1)  # 0.8G
        self.norm = BatchNorm3DEquivalent(compress, affine=True)
        self.grid_sample = GridSample3DEquivalent(padding_mode='zeros', align_corners=False)
        self.num_kp = num_kp
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

    def create_sparse_motions(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 4, 16, 64, 64)
        identity_grid = make_coordinate_grid((d, h, w), ref=kp_source)  # (16, 64, 64, 3)
        # Reshape to (3, d, h, w) to have 4 dimensions, then add batch dimension
        identity_grid = identity_grid.permute(3, 0, 1, 2)  # (3, d=16, h=64, w=64)
        identity_grid = identity_grid.unsqueeze(0)  # (1, 3, d=16, h=64, w=64) - 5D

        # Work directly with flattened coordinates to avoid 6D tensors
        # kp_driving: (bs, num_kp, 3) -> (bs, num_kp*3)
        kp_driving_flat = kp_driving.view(bs, -1)  # (bs, num_kp*3) - 2D
        kp_source_flat = kp_source.view(bs, -1)    # (bs, num_kp*3) - 2D

        # Reshape to broadcast: (bs, num_kp*3, 1, 1, 1)
        kp_driving_bc = kp_driving_flat.view(bs, -1, 1, 1, 1)  # (bs, num_kp*3, 1, 1, 1) - 5D
        kp_source_bc = kp_source_flat.view(bs, -1, 1, 1, 1)    # (bs, num_kp*3, 1, 1, 1) - 5D

        # Expand identity_grid and compute coordinate_grid
        identity_grid_expanded = identity_grid.repeat(1, self.num_kp, 1, 1, 1)  # (1, num_kp*3, d, h, w) - 5D
        coordinate_grid = identity_grid_expanded - kp_driving_bc  # (bs, num_kp*3, d, h, w) - 5D

        # Calculate driving_to_source
        driving_to_source = coordinate_grid + kp_source_bc  # (bs, num_kp*3, d, h, w) - 5D

        # adding background feature - identity_grid for background
        identity_grid_bg = identity_grid.repeat(bs, 1, 1, 1, 1)  # (bs, 3, d, h, w) - 5D
        sparse_motions = torch.cat([identity_grid_bg, driving_to_source], dim=1)  # (bs, 3+num_kp*3, d, h, w) - 5D
        return sparse_motions

    def create_deformed_feature(self, feature, sparse_motions):
        bs, c, d, h, w = feature.shape

        # Process each keypoint separately to avoid 6D tensors
        deformed_features_list = []

        # Background (identity) feature
        background_motion = sparse_motions[:, :3, :, :, :]  # (bs, 3, d, h, w) - 5D
        background_motion_for_sample = background_motion.permute(0, 2, 3, 4, 1)  # (bs, d, h, w, 3) - 5D
        background_deformed = self.grid_sample(feature, background_motion_for_sample)
        deformed_features_list.append(background_deformed)

        # Process each keypoint
        for kp_idx in range(self.num_kp):
            start_idx = 3 + kp_idx * 3
            end_idx = 3 + (kp_idx + 1) * 3
            kp_motion = sparse_motions[:, start_idx:end_idx, :, :, :]  # (bs, 3, d, h, w) - 5D
            kp_motion_for_sample = kp_motion.permute(0, 2, 3, 4, 1)  # (bs, d, h, w, 3) - 5D
            kp_deformed = self.grid_sample(feature, kp_motion_for_sample)
            deformed_features_list.append(kp_deformed)

        # Concatenate all deformed features: (bs, (num_kp+1)*c, d, h, w) - 5D
        sparse_deformed = torch.cat(deformed_features_list, dim=1)
        return sparse_deformed

    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        # feature is now (bs, (num_kp+1)*c, d, h, w) - need to extract spatial dimensions
        bs = feature.shape[0]
        spatial_size = feature.shape[2:]  # (d=16, h=64, w=64)

        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w) - 5D
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w) - 5D
        heatmap = gaussian_driving - gaussian_source  # (bs, num_kp, d, h, w) - 5D

        # adding background feature
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.dtype).to(heatmap.device)
        heatmap = torch.cat([zeros, heatmap], dim=1)  # (bs, 1+num_kp, d, h, w) - 5D
        return heatmap

    def forward(self, feature, kp_driving, kp_source):
        bs, _, d, h, w = feature.shape  # (bs, 32, 16, 64, 64)

        feature = self.compress(feature)  # (bs, 4, 16, 64, 64)
        feature = self.norm(feature)  # (bs, 4, 16, 64, 64)
        feature = F.relu(feature)  # (bs, 4, 16, 64, 64)

        out_dict = dict()

        # 1. deform 3d feature
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, (1+num_kp)*3, d, h, w) - 5D
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)  # (bs, (1+num_kp)*c, d, h, w) - 5D

        # 2. Create heatmap representations
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)  # (bs, 1+num_kp, d, h, w) - 5D

        # Prepare input by processing each keypoint+background separately to avoid 6D tensors
        c = deformed_feature.shape[1] // (self.num_kp + 1)  # channels per keypoint
        input_list = []

        for i in range(self.num_kp + 1):
            # Extract heatmap and deformed feature for this keypoint/background
            heatmap_i = heatmap[:, i:i+1, :, :, :]  # (bs, 1, d, h, w) - 5D
            deformed_i = deformed_feature[:, i*c:(i+1)*c, :, :, :]  # (bs, c, d, h, w) - 5D
            # Concatenate: (bs, 1+c, d, h, w) - 5D
            input_i = torch.cat([heatmap_i, deformed_i], dim=1)
            input_list.append(input_i)

        # Concatenate all: (bs, (1+num_kp)*(c+1), d, h, w) - 5D
        input = torch.cat(input_list, dim=1)

        prediction = self.hourglass(input)

        mask = self.mask(prediction)
        mask = F.softmax(mask, dim=1)  # (bs, 1+num_kp, d=16, h=64, w=64) - 5D
        out_dict['mask'] = mask

        # Calculate deformation by processing each component separately
        deformation_components = []
        for i in range(self.num_kp + 1):
            # Extract motion and mask for this component
            motion_i = sparse_motion[:, i*3:(i+1)*3, :, :, :]  # (bs, 3, d, h, w) - 5D
            mask_i = mask[:, i:i+1, :, :, :]  # (bs, 1, d, h, w) - 5D
            # Multiply: (bs, 3, d, h, w) - 5D
            weighted_motion_i = motion_i * mask_i
            deformation_components.append(weighted_motion_i)

        # Sum all components: (bs, 3, d, h, w) - 5D
        deformation_temp = sum(deformation_components)
        deformation = deformation_temp.permute(0, 2, 3, 4, 1)  # (bs, d, h, w, 3) - 5D

        out_dict['deformation'] = deformation

        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape
            prediction_reshape = prediction.view(bs, -1, h, w)  # (bs, channels*d, h, w) - 4D
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # (bs, 1, h, w) - 4D
            out_dict['occlusion_map'] = occlusion_map

        return out_dict
