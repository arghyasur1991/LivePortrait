# coding: utf-8

"""
utility functions and classes to handle feature extraction and model loading
"""

import os
import os.path as osp
import torch
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull # pylint: disable=E0401,E0611
from typing import Union
import cv2

from ..modules.spade_generator import SPADEDecoder
from ..modules.warping_network import WarpingNetwork
from ..modules.motion_extractor import MotionExtractor
from ..modules.appearance_feature_extractor import AppearanceFeatureExtractor
from ..modules.stitching_retargeting_network import StitchingRetargetingNetwork


def tensor_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """transform torch.Tensor into numpy.ndarray"""
    if isinstance(data, torch.Tensor):
        return data.data.cpu().numpy()
    return data

def calc_motion_multiplier(
    kp_source: Union[np.ndarray, torch.Tensor],
    kp_driving_initial: Union[np.ndarray, torch.Tensor]
) -> float:
    """calculate motion_multiplier based on the source image and the first driving frame"""
    kp_source_np = tensor_to_numpy(kp_source)
    kp_driving_initial_np = tensor_to_numpy(kp_driving_initial)

    source_area = ConvexHull(kp_source_np.squeeze(0)).volume
    driving_area = ConvexHull(kp_driving_initial_np.squeeze(0)).volume
    motion_multiplier = np.sqrt(source_area) / np.sqrt(driving_area)
    # motion_multiplier = np.cbrt(source_area) / np.cbrt(driving_area)

    return motion_multiplier

def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return osp.join(osp.dirname(filepath), basename(filepath))


def is_image(file_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)


def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2device(dct: dict, device):
    for key in dct:
        if isinstance(dct[key], torch.Tensor):
            dct[key] = dct[key].to(device)
        else:
            dct[key] = torch.tensor(dct[key]).to(device)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def adapt_conv3d_to_conv2d_weights(state_dict, model_state_dict):
    """
    Adapt 3D convolution weights to 2D convolution weights using mathematically sound approaches.

    For 3D conv weights [out_c, in_c, kd, kh, kw] -> 2D conv weights [out_c, in_c, kh, kw]:

    1. For kernel_depth=1: Direct squeeze
    2. For kernel_depth=3: Use center slice with neighbor averaging
    3. For kernel_depth=7: Use Gaussian-weighted combination
    """
    adapted_state_dict = OrderedDict()

    for key, checkpoint_param in state_dict.items():
        if key in model_state_dict:
            model_param = model_state_dict[key]
            checkpoint_shape = checkpoint_param.shape
            model_shape = model_param.shape

            # Check if this is a conv weight that needs adaptation
            if (len(checkpoint_shape) == 5 and len(model_shape) == 4 and
                checkpoint_shape[:2] == model_shape[:2] and
                checkpoint_shape[3:] == model_shape[2:]):

                depth_size = checkpoint_shape[2]

                if depth_size == 1:
                    # Direct squeeze for depth=1
                    adapted_param = checkpoint_param.squeeze(2)
                    method = "squeeze"

                elif depth_size == 3:
                    # For 3x3x3 kernels, use center slice enhanced with neighbor information
                    # This preserves the central learned pattern while incorporating depth context
                    center_slice = checkpoint_param[:, :, 1, :, :]  # Center slice
                    prev_slice = checkpoint_param[:, :, 0, :, :]    # Previous slice
                    next_slice = checkpoint_param[:, :, 2, :, :]    # Next slice

                    # Weighted combination: center gets most weight, neighbors provide context
                    # Weights: [0.2, 0.6, 0.2] - center-focused but depth-aware
                    adapted_param = 0.6 * center_slice + 0.2 * prev_slice + 0.2 * next_slice
                    method = "weighted_center"

                elif depth_size == 7:
                    # For 7x7x7 kernels, use Gaussian-weighted combination
                    # Create Gaussian weights centered on middle slice
                    center = depth_size // 2
                    sigma = depth_size / 6.0  # Spread across ~99% of kernel

                    depth_indices = torch.arange(depth_size, dtype=checkpoint_param.dtype, device=checkpoint_param.device)
                    gaussian_weights = torch.exp(-0.5 * ((depth_indices - center) / sigma) ** 2)
                    gaussian_weights = gaussian_weights / gaussian_weights.sum()

                    # Apply weighted combination across depth dimension
                    adapted_param = torch.zeros_like(checkpoint_param[:, :, 0, :, :])
                    for d_idx in range(depth_size):
                        adapted_param += gaussian_weights[d_idx] * checkpoint_param[:, :, d_idx, :, :]

                    method = "gaussian_weighted"

                else:
                    # For other sizes, use enhanced center slice approach
                    center = depth_size // 2
                    center_slice = checkpoint_param[:, :, center, :, :]

                    # If possible, average with immediate neighbors for better approximation
                    if depth_size >= 3:
                        neighbor_weight = 0.15
                        main_weight = 1.0 - 2 * neighbor_weight

                        adapted_param = main_weight * center_slice
                        if center > 0:
                            adapted_param += neighbor_weight * checkpoint_param[:, :, center-1, :, :]
                        if center < depth_size - 1:
                            adapted_param += neighbor_weight * checkpoint_param[:, :, center+1, :, :]
                    else:
                        adapted_param = center_slice

                    method = "enhanced_center"

                adapted_state_dict[key] = adapted_param
                print(f"Adapted {key}: {checkpoint_shape} -> {adapted_param.shape} (method: {method})")

            # Check if this is a conv bias (should match directly)
            elif checkpoint_shape == model_shape:
                adapted_state_dict[key] = checkpoint_param

            else:
                print(f"Warning: Could not adapt {key}: checkpoint shape {checkpoint_shape} vs model shape {model_shape}")
                # Skip mismatched parameters rather than forcing them
                continue
        else:
            # Key not in model, skip it
            print(f"Warning: Key {key} not found in model, skipping")

    # Check for missing model parameters
    missing_keys = set(model_state_dict.keys()) - set(adapted_state_dict.keys())
    if missing_keys:
        print(f"Warning: {len(missing_keys)} model parameters not found in checkpoint, will use default initialization")
        for key in list(missing_keys)[:5]:  # Show first 5
            print(f"  Missing: {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")

    return adapted_state_dict


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'warping_module':
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load checkpoint
    checkpoint_state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

    # Special handling for warping_module to adapt 3D->2D weights
    if model_type == 'warping_module':
        model_state_dict = model.state_dict()
        adapted_state_dict = adapt_conv3d_to_conv2d_weights(checkpoint_state_dict, model_state_dict)
        model.load_state_dict(adapted_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint_state_dict)

    model.eval()
    return model


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def is_square_video(video_path):
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()
    # if width != height:
        # gr.Info(f"Uploaded video is not square, force do crop (driving) to be True")

    return width == height

def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
