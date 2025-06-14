#!/usr/bin/env python3
"""
ONNX Inference Script for LivePortrait
This script performs inference using only ONNX models, no PyTorch models.

Usage:
    conda activate LivePortrait
    python onnx_inference.py --source assets/examples/source/s6.jpg --driving assets/examples/driving/d0.mp4
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.video import images2video, get_fps, add_audio_to_video, has_audio_stream
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, load_video, resize_to_limit, dump, load
from src.utils.helper import mkdir, basename, is_video, is_template, remove_suffix, is_image, calc_motion_multiplier
from src.utils.filter import smooth
from src.utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from src.utils.camera import headpose_pred_to_degree
from src.utils.rprint import rlog as log

# NumPy versions of camera functions for ONNX compatibility
def headpose_pred_to_degree_numpy(pred: np.ndarray) -> np.ndarray:
    """
    NumPy version of headpose_pred_to_degree for ONNX compatibility
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        idx_tensor = np.arange(0, 66, dtype=np.float32)
        pred_softmax = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)  # softmax
        degree = np.sum(pred_softmax * idx_tensor, axis=1) * 3 - 97.5
        return degree

    return pred


def get_rotation_matrix_numpy(pitch_: np.ndarray, yaw_: np.ndarray, roll_: np.ndarray) -> np.ndarray:
    """
    NumPy version of get_rotation_matrix for ONNX compatibility
    the input is in degree
    """
    PI = np.pi

    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    if pitch.ndim == 1:
        pitch = pitch[:, None]
    if yaw.ndim == 1:
        yaw = yaw[:, None]
    if roll.ndim == 1:
        roll = roll[:, None]

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = np.ones([bs, 1])
    zeros = np.zeros([bs, 1])
    x, y, z = pitch, yaw, roll

    rot_x = np.concatenate([
        ones, zeros, zeros,
        zeros, np.cos(x), -np.sin(x),
        zeros, np.sin(x), np.cos(x)
    ], axis=1).reshape([bs, 3, 3])

    rot_y = np.concatenate([
        np.cos(y), zeros, np.sin(y),
        zeros, ones, zeros,
        -np.sin(y), zeros, np.cos(y)
    ], axis=1).reshape([bs, 3, 3])

    rot_z = np.concatenate([
        np.cos(z), -np.sin(z), zeros,
        np.sin(z), np.cos(z), zeros,
        zeros, zeros, ones
    ], axis=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return np.transpose(rot, (0, 2, 1))  # transpose


class ONNXLivePortraitInference:
    def __init__(self,
                 model_dir: str = "./onnx_models",
                 device: str = "cpu",
                 use_int8: bool = False,
                 providers: Optional[List] = None):
        """
        Initialize ONNX LivePortrait inference

        Args:
            model_dir: Directory containing ONNX models
            device: Device to run inference on ('cpu', 'cuda')
            use_int8: Whether to use INT8 quantized models
            providers: ONNX Runtime providers to use
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.use_int8 = use_int8

        # Set up ONNX Runtime providers
        if providers is None:
            if device == 'cuda':
                self.providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                self.providers = ['CPUExecutionProvider']
        else:
            self.providers = providers

        # Set session options for optimization
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if use_int8:
            print("Enabling INT8 optimizations in ONNX Runtime...")
            self.session_options.add_session_config_entry('session.intra_op_num_threads', '0')
            self.session_options.add_session_config_entry('session.inter_op_num_threads', '0')

        print(f"Using ONNX providers: {self.providers}")

        # Initialize configurations (needed for cropping and other utilities)
        self.inference_cfg = InferenceConfig()
        self.crop_cfg = CropConfig()
        self.cropper = Cropper(crop_cfg=self.crop_cfg)

        # Load all ONNX models
        self.load_onnx_models()

        print("✓ ONNX LivePortrait inference initialized successfully")

    def get_model_path(self, model_name: str, model_type: str = "human") -> str:
        """Get path to ONNX model"""
        suffix = "_int8.onnx" if self.use_int8 else ".onnx"

        if model_type == "animal":
            return str(self.model_dir / "animal" / f"{model_name}{suffix}")
        else:
            return str(self.model_dir / f"{model_name}{suffix}")

    def load_onnx_models(self):
        """Load all ONNX models"""
        print("Loading ONNX models...")

        try:
            # Load Appearance Feature Extractor (F)
            f_path = self.get_model_path("appearance_feature_extractor")
            print(f"Loading Appearance Feature Extractor from {f_path}")
            self.appearance_extractor_session = ort.InferenceSession(
                f_path, self.session_options, providers=self.providers
            )

            # Load Motion Extractor (M)
            m_path = self.get_model_path("motion_extractor")
            print(f"Loading Motion Extractor from {m_path}")
            self.motion_extractor_session = ort.InferenceSession(
                m_path, self.session_options, providers=self.providers
            )

            # Load Warping Network (W)
            w_path = self.get_model_path("warping_network")
            print(f"Loading Warping Network from {w_path}")
            self.warping_network_session = ort.InferenceSession(
                w_path, self.session_options, providers=self.providers
            )

            # Load SPADE Generator (G)
            g_path = self.get_model_path("spade_generator")
            print(f"Loading SPADE Generator from {g_path}")
            self.spade_generator_session = ort.InferenceSession(
                g_path, self.session_options, providers=self.providers
            )

            # Load Stitching/Retargeting Networks (S) - optional
            self.stitching_sessions = {}
            stitching_dir = self.model_dir / "stitching"
            if stitching_dir.exists():
                for network_name in ['stitching', 'lip', 'eye']:
                    network_path = stitching_dir / f"{network_name}.onnx"
                    if network_path.exists():
                        print(f"Loading {network_name} network from {network_path}")
                        self.stitching_sessions[network_name] = ort.InferenceSession(
                            str(network_path), self.session_options, providers=self.providers
                        )

            print("✓ All ONNX models loaded successfully")

        except Exception as e:
            print(f"❌ Failed to load ONNX models: {e}")
            raise

    def prepare_source(self, img: np.ndarray) -> np.ndarray:
        """Prepare source image for inference"""
        h, w = img.shape[:2]
        if h != 256 or w != 256:
            x = cv2.resize(img, (256, 256))
        else:
            x = img.copy()

        if x.ndim == 3:
            x = x[np.newaxis].astype(np.float32) / 255.0  # HxWx3 -> 1xHxWx3, normalized to 0~1
        elif x.ndim == 4:
            x = x.astype(np.float32) / 255.0  # BxHxWx3, normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')

        x = np.clip(x, 0, 1)  # clip to 0~1
        x = x.transpose(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        return x

    def extract_feature_3d(self, x: np.ndarray) -> np.ndarray:
        """Extract 3D appearance features using ONNX model"""
        input_name = self.appearance_extractor_session.get_inputs()[0].name
        output = self.appearance_extractor_session.run(None, {input_name: x})
        return output[0]

    def get_kp_info(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """Get keypoint information using ONNX model"""
        input_name = self.motion_extractor_session.get_inputs()[0].name
        outputs = self.motion_extractor_session.run(None, {input_name: x})

        # Outputs are: ['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp']
        output_names = [out.name for out in self.motion_extractor_session.get_outputs()]

        kp_info = {}
        for i, name in enumerate(output_names):
            kp_info[name] = outputs[i]

        # Process outputs to match PyTorch version format
        bs = kp_info['kp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree_numpy(kp_info['pitch'])
        kp_info['yaw'] = headpose_pred_to_degree_numpy(kp_info['yaw'])
        kp_info['roll'] = headpose_pred_to_degree_numpy(kp_info['roll'])
        kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3

        return kp_info

    def transform_keypoint(self, kp_info: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform keypoints with pose, shift, and expression deformation"""
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']
        t, exp = kp_info['t'], kp_info['exp']
        scale = kp_info['scale']

        bs = kp.shape[0]
        if kp.ndim == 2:
            num_kp = kp.shape[0] // 3
            kp = kp.reshape(1, num_kp, 3)

        # Get rotation matrix
        rot_matrix = get_rotation_matrix_numpy(pitch, yaw, roll)  # (bs, 3, 3)

        # Apply transformations
        kp_transformed = kp @ rot_matrix.transpose(0, 2, 1)  # (bs, k, 3)
        kp_transformed += exp
        kp_transformed = kp_transformed * scale[..., None, None]
        kp_transformed[..., :2] += t[..., None, :2]

        return kp_transformed.reshape(bs, -1)

    def warp_decode(self, feature_3d: np.ndarray, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """Warp features and decode using ONNX models"""
        # Use warping network
        input_names = [inp.name for inp in self.warping_network_session.get_inputs()]

        # The warping network from our export expects only feature_3d as input
        # (since we created a simplified version due to 5D grid sampling limitations)
        if len(input_names) == 1:
            # Simplified warping network - only takes feature_3d
            inputs = {input_names[0]: feature_3d}
        else:
            # Full warping network - takes feature_3d, kp_driving, kp_source
            inputs = {
                input_names[0]: feature_3d,
                input_names[1]: kp_driving,
                input_names[2]: kp_source
            }

        warping_outputs = self.warping_network_session.run(None, inputs)
        warped_feature = warping_outputs[0]

        # Use SPADE generator
        spade_input_name = self.spade_generator_session.get_inputs()[0].name
        spade_output = self.spade_generator_session.run(None, {spade_input_name: warped_feature})

        return spade_output[0]

    def retarget_eye(self, kp_source: np.ndarray, eye_close_ratio: float) -> np.ndarray:
        """Retarget eye using ONNX model if available"""
        if 'eye' not in self.stitching_sessions:
            return kp_source

        # Prepare input: kp_source + eye_close_ratio
        bs = kp_source.shape[0]
        kp_flat = kp_source.reshape(bs, -1)  # (bs, 63)
        eye_input = np.concatenate([kp_flat, np.array([[eye_close_ratio]] * bs),
                                   np.array([[eye_close_ratio]] * bs),
                                   np.array([[eye_close_ratio]] * bs)], axis=1)  # (bs, 66)

        input_name = self.stitching_sessions['eye'].get_inputs()[0].name
        output = self.stitching_sessions['eye'].run(None, {input_name: eye_input})

        return output[0].reshape(bs, -1, 3)

    def retarget_lip(self, kp_source: np.ndarray, lip_close_ratio: float) -> np.ndarray:
        """Retarget lip using ONNX model if available"""
        if 'lip' not in self.stitching_sessions:
            return kp_source

        # Prepare input: kp_source + lip_close_ratio
        bs = kp_source.shape[0]
        kp_flat = kp_source.reshape(bs, -1)  # (bs, 63)
        lip_input = np.concatenate([kp_flat, np.array([[lip_close_ratio]] * bs),
                                   np.array([[lip_close_ratio]] * bs)], axis=1)  # (bs, 65)

        input_name = self.stitching_sessions['lip'].get_inputs()[0].name
        output = self.stitching_sessions['lip'].run(None, {input_name: lip_input})

        return output[0].reshape(bs, -1, 3)

    def stitching(self, kp_source: np.ndarray, kp_driving: np.ndarray) -> np.ndarray:
        """Stitch keypoints using ONNX model if available"""
        if 'stitching' not in self.stitching_sessions:
            return kp_driving

        # Prepare input: concatenate kp_source and kp_driving
        bs = kp_source.shape[0]
        kp_source_flat = kp_source.reshape(bs, -1)  # (bs, 63)
        kp_driving_flat = kp_driving.reshape(bs, -1)  # (bs, 63)
        stitching_input = np.concatenate([kp_source_flat, kp_driving_flat], axis=1)  # (bs, 126)

        input_name = self.stitching_sessions['stitching'].get_inputs()[0].name
        output = self.stitching_sessions['stitching'].run(None, {input_name: stitching_input})

        # Output includes delta_tx, delta_ty, plus keypoints
        delta_tx_ty = output[0][:, :2]  # (bs, 2)
        kp_stitched = output[0][:, 2:].reshape(bs, -1, 3)  # (bs, 21, 3)

        # Apply translation
        kp_stitched[..., :2] += delta_tx_ty[..., None, :]

        return kp_stitched

    def parse_output(self, out: np.ndarray) -> np.ndarray:
        """Parse network output to image"""
        out = np.transpose(out, (0, 2, 3, 1))  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)
        out = (out * 255).astype(np.uint8)
        return out[0]  # Remove batch dimension

    def calc_ratio(self, lmk_lst: List[np.ndarray]) -> Tuple[List[float], List[float]]:
        """Calculate eye and lip close ratios from landmarks"""
        c_d_eyes_lst = []
        c_d_lip_lst = []

        for i, lmk in enumerate(lmk_lst):
            # The retargeting functions expect landmarks in shape (batch, num_points, 2)
            # NOT flattened coordinates. They use point indices to access specific landmarks
            if lmk.ndim == 2 and lmk.shape[1] == 2:
                # Add batch dimension: (num_points, 2) -> (1, num_points, 2)
                lmk_batch = lmk[None, ...]  # Shape: (1, 203, 2)
            else:
                # Already has batch dimension or different format
                lmk_batch = lmk

            c_d_eyes = calc_eye_close_ratio(lmk_batch)
            c_d_lip = calc_lip_close_ratio(lmk_batch)
            c_d_eyes_lst.append(c_d_eyes)
            c_d_lip_lst.append(c_d_lip)

        return c_d_eyes_lst, c_d_lip_lst

    def make_motion_template(self, I_lst: np.ndarray, c_eyes_lst: List[float], c_lip_lst: List[float], output_fps: int = 25) -> Dict:
        """Create motion template using ONNX models"""
        n_frames = I_lst.shape[0]
        template_dct = {
            'n_frames': n_frames,
            'output_fps': output_fps,
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in range(n_frames):
            print(f"Processing frame {i+1}/{n_frames}")

            # Get keypoint info using ONNX model
            I_i = I_lst[i:i+1]  # Keep batch dimension
            x_i_info = self.get_kp_info(I_i)
            x_s = self.transform_keypoint(x_i_info)

            # Get rotation matrix
            R_i = get_rotation_matrix_numpy(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])

            item_dct = {
                'scale': x_i_info['scale'].astype(np.float32),
                'R': R_i.astype(np.float32),
                'exp': x_i_info['exp'].astype(np.float32),
                't': x_i_info['t'].astype(np.float32),
                'kp': x_i_info['kp'].astype(np.float32),
                'x_s': x_s.astype(np.float32),
            }

            template_dct['motion'].append(item_dct)
            template_dct['c_eyes_lst'].append(c_eyes_lst[i])
            template_dct['c_lip_lst'].append(c_lip_lst[i])

        return template_dct

    def execute(self, args):
        """Main execution function"""
        log("Starting ONNX LivePortrait inference...")

        ######## Load source input ########
        flag_is_source_video = False
        source_fps = None

        if is_image(args.source):
            flag_is_source_video = False
            img_rgb = load_image_rgb(args.source)
            img_rgb = resize_to_limit(img_rgb, self.inference_cfg.source_max_dim, self.inference_cfg.source_division)
            log(f"Load source image from {args.source}")
            source_rgb_lst = [img_rgb]
        elif is_video(args.source):
            flag_is_source_video = True
            source_rgb_lst = load_video(args.source)
            source_rgb_lst = [resize_to_limit(img, self.inference_cfg.source_max_dim, self.inference_cfg.source_division) for img in source_rgb_lst]
            source_fps = int(get_fps(args.source))
            log(f"Load source video from {args.source}, FPS is {source_fps}")
        else:
            raise Exception(f"Unknown source format: {args.source}")

        ######## Process driving info ########
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None

        if flag_load_from_template:
            log(f"Load from template: {args.driving}")
            driving_template_dct = load(args.driving)
            c_d_eyes_lst = driving_template_dct.get('c_eyes_lst', driving_template_dct.get('c_d_eyes_lst', []))
            c_d_lip_lst = driving_template_dct.get('c_lip_lst', driving_template_dct.get('c_d_lip_lst', []))
            driving_n_frames = driving_template_dct['n_frames']
            flag_is_driving_video = True if driving_n_frames > 1 else False

            if flag_is_source_video and flag_is_driving_video:
                n_frames = min(len(source_rgb_lst), driving_n_frames)
            elif flag_is_source_video and not flag_is_driving_video:
                n_frames = len(source_rgb_lst)
            else:
                n_frames = driving_n_frames

            output_fps = driving_template_dct.get('output_fps', self.inference_cfg.output_fps)
            log(f'The FPS of template: {output_fps}')

        elif os.path.exists(args.driving):
            if is_video(args.driving):
                flag_is_driving_video = True
                output_fps = int(get_fps(args.driving))
                log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
                driving_rgb_lst = load_video(args.driving)
            elif is_image(args.driving):
                flag_is_driving_video = False
                driving_img_rgb = load_image_rgb(args.driving)
                output_fps = 25
                log(f"Load driving image from {args.driving}")
                driving_rgb_lst = [driving_img_rgb]
            else:
                raise Exception(f"{args.driving} is not a supported type!")

            # Make motion template using ONNX models
            log("Start making driving motion template...")
            driving_n_frames = len(driving_rgb_lst)

            if flag_is_source_video and flag_is_driving_video:
                n_frames = min(len(source_rgb_lst), driving_n_frames)
                driving_rgb_lst = driving_rgb_lst[:n_frames]
            elif flag_is_source_video and not flag_is_driving_video:
                n_frames = len(source_rgb_lst)
            else:
                n_frames = driving_n_frames

            # Crop driving video
            if self.inference_cfg.flag_crop_driving_video or not is_video(args.driving):
                ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                if len(ret_d["frame_crop_lst"]) != n_frames and flag_is_driving_video:
                    n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(frame, (256, 256)) for frame in driving_rgb_crop_lst]
            else:
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(frame, (256, 256)) for frame in driving_rgb_lst]

            c_d_eyes_lst, c_d_lip_lst = self.calc_ratio(driving_lmk_crop_lst)

            # Prepare driving frames for ONNX inference
            I_d_lst = np.array([self.prepare_source(frame) for frame in driving_rgb_crop_256x256_lst])
            I_d_lst = np.squeeze(I_d_lst, axis=1)  # Remove extra dimension

            driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)

            wfp_template = remove_suffix(args.driving) + '.pkl'
            dump(wfp_template, driving_template_dct)
            log(f"Dump motion template to {wfp_template}")
        else:
            raise Exception(f"{args.driving} does not exist!")

        if not flag_is_driving_video:
            c_d_eyes_lst = c_d_eyes_lst * n_frames
            c_d_lip_lst = c_d_lip_lst * n_frames

        ######## Process source and generate results ########
        I_p_lst = []

        if flag_is_source_video:
            log("Processing source video...")
            source_rgb_lst = source_rgb_lst[:n_frames]

            if self.inference_cfg.flag_do_crop:
                ret_s = self.cropper.crop_source_video(source_rgb_lst, self.crop_cfg)
                log(f'Source video is cropped, {len(ret_s["frame_crop_lst"])} frames are processed.')
                if len(ret_s["frame_crop_lst"]) != n_frames:
                    n_frames = min(n_frames, len(ret_s["frame_crop_lst"]))
                source_rgb_crop_lst = ret_s['frame_crop_lst']
            else:
                source_rgb_crop_lst = [cv2.resize(frame, (256, 256)) for frame in source_rgb_lst]

            # Extract source features once (first frame)
            source_prepared = self.prepare_source(source_rgb_crop_lst[0])
            source_feature_3d = self.extract_feature_3d(source_prepared)
            source_kp_info = self.get_kp_info(source_prepared)
            source_kp = source_kp_info['kp'].reshape(1, -1, 3)

        else:
            # Single source image
            if self.inference_cfg.flag_do_crop:
                ret_s = self.cropper.crop_source_image(source_rgb_lst[0], self.crop_cfg)
                source_rgb_crop = ret_s['img_crop']
            else:
                source_rgb_crop = cv2.resize(source_rgb_lst[0], (256, 256))

            source_prepared = self.prepare_source(source_rgb_crop)
            source_feature_3d = self.extract_feature_3d(source_prepared)
            source_kp_info = self.get_kp_info(source_prepared)
            source_kp = source_kp_info['kp'].reshape(1, -1, 3)

        # Generate animated frames
        log("Generating animated frames...")
        for i in range(n_frames):
            if i % 10 == 0:
                log(f"Processing frame {i+1}/{n_frames}")

            # Get driving motion
            motion = driving_template_dct['motion'][i]
            driving_kp = motion['kp'].reshape(1, -1, 3)

            # Apply retargeting if enabled
            if self.inference_cfg.flag_eye_retargeting:
                driving_kp = self.retarget_eye(driving_kp, c_d_eyes_lst[i])
            if self.inference_cfg.flag_lip_retargeting:
                driving_kp = self.retarget_lip(driving_kp, c_d_lip_lst[i])
            if self.inference_cfg.flag_stitching:
                driving_kp = self.stitching(source_kp, driving_kp)

            # Generate frame
            out = self.warp_decode(source_feature_3d, source_kp.reshape(1, -1), driving_kp.reshape(1, -1))
            out_img = self.parse_output(out)
            I_p_lst.append(out_img)

        # Save results
        wfp_out = f"{remove_suffix(args.source)}_onnx_{basename(args.driving)}.mp4"
        log(f"Saving result to {wfp_out}")

        images2video(I_p_lst, wfp=wfp_out, fps=output_fps)

        # Add audio if available
        if has_audio_stream(args.driving):
            log("Adding audio to output video...")
            wfp_out_with_audio = wfp_out.replace('.mp4', '_with_audio.mp4')
            add_audio_to_video(wfp_out, args.driving, wfp_out_with_audio)
            log(f"Final result with audio: {wfp_out_with_audio}")

        log("✅ ONNX inference completed!")


def main():
    parser = argparse.ArgumentParser(description='ONNX LivePortrait Inference')
    parser.add_argument('--source', type=str, required=True,
                       help='Source image or video path')
    parser.add_argument('--driving', type=str, required=True,
                       help='Driving video or image path')
    parser.add_argument('--model_dir', type=str, default='./onnx_models',
                       help='Directory containing ONNX models')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--use_int8', action='store_true',
                       help='Use INT8 quantized models')
    parser.add_argument('--flag_eye_retargeting', action='store_true',
                       help='Enable eye retargeting')
    parser.add_argument('--flag_lip_retargeting', action='store_true',
                       help='Enable lip retargeting')
    parser.add_argument('--flag_stitching', action='store_true', default=True,
                       help='Enable stitching')

    args = parser.parse_args()

    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory {args.model_dir} does not exist!")
        print("Please run export_to_onnx.py first to generate ONNX models.")
        return

    try:
        # Initialize ONNX inference
        onnx_inference = ONNXLivePortraitInference(
            model_dir=args.model_dir,
            device=args.device,
            use_int8=args.use_int8
        )

        # Set inference flags
        onnx_inference.inference_cfg.flag_eye_retargeting = args.flag_eye_retargeting
        onnx_inference.inference_cfg.flag_lip_retargeting = args.flag_lip_retargeting
        onnx_inference.inference_cfg.flag_stitching = args.flag_stitching

        # Run inference
        start_time = time.time()
        onnx_inference.execute(args)
        end_time = time.time()

        print(f"Total inference time: {end_time - start_time:.2f} seconds")

    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
