#!/usr/bin/env python3

"""
Hybrid Debug Inference - Mix and match PyTorch vs ONNX models
This allows us to isolate which model component is failing
"""

import os
import os.path as osp
import numpy as np
import cv2
import torch
import onnxruntime as ort
from typing import Dict

# Add src to path
import sys
sys.path.insert(0, 'src')

from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.io import load_image_rgb
from src.utils.cropper import Cropper
from src.live_portrait_wrapper import LivePortraitWrapper

# Import ONNX utilities
from onnx_inference import ONNXLivePortraitPipeline


class HybridDebugPipeline:
    """
    Hybrid pipeline that can use PyTorch or ONNX for each component:
    - appearance_extractor: pytorch/onnx
    - motion_extractor: pytorch/onnx
    - warping_spade: pytorch/onnx
    - stitching: pytorch/onnx
    """

    def __init__(self,
                 inference_cfg: InferenceConfig,
                 crop_cfg: CropConfig,
                 appearance_mode="pytorch",    # pytorch/onnx
                 motion_mode="pytorch",        # pytorch/onnx
                 warping_mode="pytorch",       # pytorch/onnx
                 stitching_mode="pytorch"):    # pytorch/onnx

        self.inference_cfg = inference_cfg
        self.crop_cfg = crop_cfg

        self.appearance_mode = appearance_mode
        self.motion_mode = motion_mode
        self.warping_mode = warping_mode
        self.stitching_mode = stitching_mode

        print(f"🧪 Hybrid Debug Pipeline Configuration:")
        print(f"   Appearance Extractor: {appearance_mode}")
        print(f"   Motion Extractor: {motion_mode}")
        print(f"   Warping/SPADE: {warping_mode}")
        print(f"   Stitching: {stitching_mode}")

        # Initialize PyTorch pipeline
        self.pytorch_pipeline = LivePortraitWrapper(
            inference_cfg=inference_cfg
        )

        # Initialize ONNX pipeline
        self.onnx_pipeline = ONNXLivePortraitPipeline(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg,
            model_dir="./onnx_models"
        )

        # Initialize cropper
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def prepare_source(self, img: np.ndarray) -> np.ndarray:
        """Prepare source image - always use ONNX version for consistency"""
        return self.onnx_pipeline.prepare_source(img)

    def extract_feature_3d(self, x):
        """Extract appearance features using selected mode"""
        if self.appearance_mode == "pytorch":
            print("🔥 Using PyTorch for appearance extraction")
            if isinstance(x, np.ndarray):
                # Convert to torch and let pytorch_pipeline handle device placement
                x = torch.from_numpy(x).to(self.pytorch_pipeline.device)
            return self.pytorch_pipeline.extract_feature_3d(x)
        else:
            print("🤖 Using ONNX for appearance extraction")
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            return self.onnx_pipeline.extract_feature_3d(x)

    def get_kp_info(self, x):
        """Extract motion/keypoints using selected mode"""
        if self.motion_mode == "pytorch":
            print("🔥 Using PyTorch for motion extraction")
            if isinstance(x, np.ndarray):
                # Convert to torch and let pytorch_pipeline handle device placement
                x = torch.from_numpy(x).to(self.pytorch_pipeline.device)
            return self.pytorch_pipeline.get_kp_info(x)
        else:
            print("🤖 Using ONNX for motion extraction")
            if isinstance(x, torch.Tensor):
                x = x.cpu().numpy()
            return self.onnx_pipeline.get_kp_info(x)

    def transform_keypoint(self, kp_info):
        """Transform keypoints - use the same mode as motion extraction"""
        if self.motion_mode == "pytorch":
            return self.pytorch_pipeline.transform_keypoint(kp_info)
        else:
            return self.onnx_pipeline.transform_keypoint(kp_info)

    def warp_decode(self, feature_3d, kp_source, kp_driving):
        """Warp and decode using selected mode"""
        if self.warping_mode == "pytorch":
            print("🔥 Using PyTorch for warping/generation")
            # Convert to torch if needed
            if isinstance(feature_3d, np.ndarray):
                feature_3d = torch.from_numpy(feature_3d).to(self.pytorch_pipeline.device)
            if isinstance(kp_source, np.ndarray):
                kp_source = torch.from_numpy(kp_source).to(self.pytorch_pipeline.device)
            if isinstance(kp_driving, np.ndarray):
                kp_driving = torch.from_numpy(kp_driving).to(self.pytorch_pipeline.device)
            return self.pytorch_pipeline.warp_decode(feature_3d, kp_source, kp_driving)
        else:
            print("🤖 Using ONNX for warping/generation")
            # Convert to numpy if needed
            if isinstance(feature_3d, torch.Tensor):
                feature_3d = feature_3d.cpu().numpy()
            if isinstance(kp_source, torch.Tensor):
                kp_source = kp_source.cpu().numpy()
            if isinstance(kp_driving, torch.Tensor):
                kp_driving = kp_driving.cpu().numpy()
            return self.onnx_pipeline.warp_decode(feature_3d, kp_source, kp_driving)

    def stitching(self, kp_source, kp_driving):
        """Apply stitching using selected mode"""
        if self.stitching_mode == "pytorch":
            print("🔥 Using PyTorch for stitching")
            # Convert to torch if needed
            if isinstance(kp_source, np.ndarray):
                kp_source = torch.from_numpy(kp_source).to(self.pytorch_pipeline.device)
            if isinstance(kp_driving, np.ndarray):
                kp_driving = torch.from_numpy(kp_driving).to(self.pytorch_pipeline.device)
            return self.pytorch_pipeline.stitching(kp_source, kp_driving)
        else:
            print("🤖 Using ONNX for stitching")
            # Convert to numpy if needed
            if isinstance(kp_source, torch.Tensor):
                kp_source = kp_source.cpu().numpy()
            if isinstance(kp_driving, torch.Tensor):
                kp_driving = kp_driving.cpu().numpy()
            return self.onnx_pipeline.stitching(kp_source, kp_driving)

    def parse_output(self, out):
        """Parse output to image"""
        if isinstance(out, torch.Tensor):
            # PyTorch output
            out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = np.clip(out, 0, 1)
            out = (out * 255).astype(np.uint8)
        else:
            # ONNX output
            out = self.onnx_pipeline.parse_output(out)
        return out

    def execute_simple_test(self, source_path: str, driving_path: str):
        """Execute a simple test with two images"""
        print(f"\n🧪 Testing with source: {source_path}")
        print(f"🧪 Testing with driving: {driving_path}")

        # Load images
        source_img = load_image_rgb(source_path)
        driving_img = load_image_rgb(driving_path)

        # Resize and prepare
        source_img = cv2.resize(source_img, (256, 256))
        driving_img = cv2.resize(driving_img, (256, 256))

        source_prepared = self.prepare_source(source_img)
        driving_prepared = self.prepare_source(driving_img)

        print(f"Source shape: {source_prepared.shape}")
        print(f"Driving shape: {driving_prepared.shape}")

        try:
            # Step 1: Extract appearance features
            print("\n🔍 Step 1: Extracting appearance features...")
            source_feature_3d = self.extract_feature_3d(source_prepared)
            print(f"✓ Appearance features extracted: {type(source_feature_3d)} {getattr(source_feature_3d, 'shape', 'no shape')}")

            # Step 2: Extract keypoints
            print("\n🔍 Step 2: Extracting keypoints...")
            source_kp_info = self.get_kp_info(source_prepared)
            driving_kp_info = self.get_kp_info(driving_prepared)

            # Transform keypoints
            if self.motion_mode == "pytorch":
                source_kp = self.transform_keypoint(source_kp_info)
                driving_kp = self.transform_keypoint(driving_kp_info)
                # Reshape for pytorch format
                if isinstance(source_kp, torch.Tensor):
                    source_kp = source_kp.reshape(1, -1)
                    driving_kp = driving_kp.reshape(1, -1)
            else:
                source_kp = source_kp_info['kp'].reshape(1, -1, 3)
                driving_kp = driving_kp_info['kp'].reshape(1, -1, 3)

            print(f"✓ Source keypoints: {type(source_kp)} {getattr(source_kp, 'shape', 'no shape')}")
            print(f"✓ Driving keypoints: {type(driving_kp)} {getattr(driving_kp, 'shape', 'no shape')}")

            # Step 3: Apply stitching if enabled
            if self.inference_cfg.flag_stitching:
                print("\n🔍 Step 3: Applying stitching...")
                original_driving_kp = driving_kp.clone() if isinstance(driving_kp, torch.Tensor) else driving_kp.copy()
                stitched_kp = self.stitching(source_kp, driving_kp)

                # Check if stitching changed anything
                if isinstance(stitched_kp, torch.Tensor) and isinstance(original_driving_kp, torch.Tensor):
                    diff = torch.abs(stitched_kp - original_driving_kp).mean().item()
                elif isinstance(stitched_kp, np.ndarray) and isinstance(original_driving_kp, np.ndarray):
                    diff = np.abs(stitched_kp - original_driving_kp).mean()
                else:
                    diff = "type_mismatch"

                print(f"✓ Stitching applied, difference from original: {diff}")
                driving_kp = stitched_kp
            else:
                print("\n🔍 Step 3: Skipping stitching (disabled)")

            # Step 4: Generate image
            print("\n🔍 Step 4: Generating image...")
            if self.warping_mode == "pytorch":
                # For PyTorch, need correct shapes
                if isinstance(source_kp, np.ndarray):
                    source_kp_flat = torch.from_numpy(source_kp.reshape(1, -1)).to(self.pytorch_pipeline.device)
                    driving_kp_flat = torch.from_numpy(driving_kp.reshape(1, -1)).to(self.pytorch_pipeline.device)
                else:
                    source_kp_flat = source_kp.reshape(1, -1)
                    driving_kp_flat = driving_kp.reshape(1, -1)
            else:
                # For ONNX, need correct shapes
                if isinstance(source_kp, torch.Tensor):
                    source_kp_flat = source_kp.cpu().numpy().reshape(1, -1)
                    driving_kp_flat = driving_kp.cpu().numpy().reshape(1, -1)
                else:
                    source_kp_flat = source_kp.reshape(1, -1)
                    driving_kp_flat = driving_kp.reshape(1, -1)

            output = self.warp_decode(source_feature_3d, source_kp_flat, driving_kp_flat)
            print(f"✓ Image generated: {type(output)} {getattr(output, 'shape', 'no shape')}")

            # Step 5: Parse output
            print("\n🔍 Step 5: Parsing output...")
            final_img = self.parse_output(output)
            print(f"✓ Final image: {final_img.shape}, dtype: {final_img.dtype}")

            # Save result
            config_name = f"app{self.appearance_mode[0]}_mot{self.motion_mode[0]}_warp{self.warping_mode[0]}_stitch{self.stitching_mode[0]}"
            output_path = f"hybrid_debug_{config_name}.jpg"
            cv2.imwrite(output_path, cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR))
            print(f"✅ Result saved to: {output_path}")

            return True

        except Exception as e:
            print(f"❌ Error in pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_all_combinations():
    """Test different combinations of PyTorch/ONNX models"""

    source_path = "assets/examples/source/s6.jpg"
    driving_path = "assets/examples/source/s9.jpg"

    # Define test configurations
    configs = [
        # Start with all PyTorch (should work)
        ("pytorch", "pytorch", "pytorch", "pytorch"),

        # Replace one component at a time with ONNX
        ("onnx", "pytorch", "pytorch", "pytorch"),      # appearance -> ONNX
        ("pytorch", "onnx", "pytorch", "pytorch"),      # motion -> ONNX
        ("pytorch", "pytorch", "onnx", "pytorch"),      # warping -> ONNX
        ("pytorch", "pytorch", "pytorch", "onnx"),      # stitching -> ONNX

        # All ONNX (this should reproduce the issue)
        ("onnx", "onnx", "onnx", "onnx"),
    ]

    inference_cfg = InferenceConfig()
    inference_cfg.flag_stitching = True  # Enable stitching for testing

    crop_cfg = CropConfig()

    results = {}

    for i, (app_mode, mot_mode, warp_mode, stitch_mode) in enumerate(configs):
        config_name = f"app{app_mode[0]}_mot{mot_mode[0]}_warp{warp_mode[0]}_stitch{stitch_mode[0]}"
        print(f"\n{'='*60}")
        print(f"🧪 Test {i+1}/{len(configs)}: {config_name}")
        print(f"{'='*60}")

        try:
            pipeline = HybridDebugPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg,
                appearance_mode=app_mode,
                motion_mode=mot_mode,
                warping_mode=warp_mode,
                stitching_mode=stitch_mode
            )

            success = pipeline.execute_simple_test(source_path, driving_path)
            results[config_name] = "✅ SUCCESS" if success else "❌ FAILED"

        except Exception as e:
            print(f"❌ Failed to initialize pipeline: {e}")
            results[config_name] = f"❌ INIT_FAILED: {str(e)[:50]}"

    # Print summary
    print(f"\n{'='*60}")
    print("🧪 HYBRID DEBUG RESULTS SUMMARY")
    print(f"{'='*60}")
    for config, result in results.items():
        print(f"{config:<25} | {result}")

    print(f"\n💡 Analysis Tips:")
    print("- If all PyTorch works but any ONNX component fails, that component has issues")
    print("- Compare output images to see which component breaks the animation")
    print("- Look for the first ONNX component that causes problems")


if __name__ == "__main__":
    test_all_combinations()
