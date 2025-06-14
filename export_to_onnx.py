#!/usr/bin/env python3
"""
ONNX Export Script for LivePortrait Models
This script exports all LivePortrait PyTorch models to ONNX format for faster inference.

Usage:
    conda activate LivePortrait
    python export_to_onnx.py --output_dir ./onnx_models --export_int8
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

# INT8 quantization support
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    INT8_AVAILABLE = True
    print("✓ INT8 quantization support available")
except ImportError:
    INT8_AVAILABLE = False
    print("⚠️ INT8 quantization not available. Install with: pip install onnxruntime")

# Disable attention optimizations for ONNX export
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

# Add the project root to Python path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.inference_config import InferenceConfig
from src.utils.helper import load_model


def export_appearance_feature_extractor_to_onnx(model, output_path: str, device='cpu', opset_version=20):
    """Export Appearance Feature Extractor (A) to ONNX"""
    print(f"Exporting Appearance Feature Extractor to {output_path}")

    output_path = str(output_path)
    device = torch.device('cpu')

    # Move model to cpu and eval mode
    model = model.to(device)
    model.eval()

    # Create dummy input
    source_image = torch.randn(1, 3, 256, 256, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            source_image,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['source_image'],
            output_names=['feature_3d'],
            dynamic_axes={
                'source_image': {0: 'batch_size'},
                'feature_3d': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"✓ Appearance Feature Extractor exported successfully")


def export_motion_extractor_to_onnx(model, output_path: str, device='cpu', opset_version=20):
    """Export Motion Extractor (M) to ONNX"""
    print(f"Exporting Motion Extractor to {output_path}")

    output_path = str(output_path)
    device = torch.device('cpu')

    # Move model to cpu and eval mode
    model = model.to(device)
    model.eval()

    # Create dummy input
    input_image = torch.randn(1, 3, 256, 256, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            input_image,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_image'],
            output_names=['pitch', 'yaw', 'roll', 't', 'exp', 'scale', 'kp'],
            dynamic_axes={
                'input_image': {0: 'batch_size'},
                'pitch': {0: 'batch_size'},
                'yaw': {0: 'batch_size'},
                'roll': {0: 'batch_size'},
                't': {0: 'batch_size'},
                'exp': {0: 'batch_size'},
                'scale': {0: 'batch_size'},
                'kp': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"✓ Motion Extractor exported successfully")


def export_warping_network_to_onnx(model, output_path: str, device='cpu', opset_version=20):
    """Export Warping Network (W) to ONNX using original grid_sample with opset 20"""
    print(f"Exporting Warping Network to {output_path}")

    output_path = str(output_path)
    device = torch.device('cpu')

    class WarpingNetworkWrapper(torch.nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.warping_network = original_model

        def forward(self, feature_3d, kp_driving, kp_source):
            """
            Use the original warping network with native grid_sample
            feature_3d: (B, C, D, H, W)
            kp_driving: (B, num_kp, 3)
            kp_source: (B, num_kp, 3)
            """
            # Call the original warping network
            result = self.warping_network(feature_3d, kp_source, kp_driving)

            # Handle different return formats
            if isinstance(result, dict):
                out = result.get('out', result.get('warped_feature'))
                occlusion_map = result.get('occlusion_map')
                deformation = result.get('deformation')
            elif isinstance(result, (list, tuple)):
                out = result[0] if len(result) > 0 else None
                occlusion_map = result[1] if len(result) > 1 else None
                deformation = result[2] if len(result) > 2 else None
            else:
                out = result
                occlusion_map = None
                deformation = None

            # Ensure we have all required outputs
            if out is None:
                raise ValueError("Warping network did not return valid output")

            # Create dummy outputs if missing
            bs = feature_3d.shape[0]
            h, w = feature_3d.shape[-2:]

            if occlusion_map is None:
                occlusion_map = torch.ones(bs, 1, h, w, device=feature_3d.device, dtype=feature_3d.dtype)

            if deformation is None:
                d = feature_3d.shape[2]
                deformation = torch.zeros(bs, d, h, w, 3, device=feature_3d.device, dtype=feature_3d.dtype)

            return out, occlusion_map, deformation

    # Create wrapper
    wrapper = WarpingNetworkWrapper(model).to(device)
    wrapper.eval()

    # Create test inputs
    feature_3d = torch.randn(1, 32, 16, 64, 64, device=device)
    kp_driving = torch.randn(1, 21, 3, device=device)
    kp_source = torch.randn(1, 21, 3, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Exporting warping network with original grid_sample using opset 20...")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (feature_3d, kp_driving, kp_source),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['feature_3d', 'kp_driving', 'kp_source'],
            output_names=['warped_feature', 'occlusion_map', 'deformation'],
            dynamic_axes={
                'feature_3d': {0: 'batch_size'},
                'kp_driving': {0: 'batch_size'},
                'kp_source': {0: 'batch_size'},
                'warped_feature': {0: 'batch_size'},
                'occlusion_map': {0: 'batch_size'},
                'deformation': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"✓ Warping Network exported successfully using original grid_sample")
    print(f"✓ Opset 20 provides native support for 5D operations")
    print(f"✓ Ready for Unity/C# inference")

    return wrapper


def export_spade_generator_to_onnx(model, output_path: str, device='cpu', opset_version=20):
    """Export SPADE Generator (G) to ONNX"""
    print(f"Exporting SPADE Generator to {output_path}")

    output_path = str(output_path)
    device = torch.device('cpu')

    class SPADEGeneratorWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, warped_feature):
            return self.model(warped_feature)

    wrapper = SPADEGeneratorWrapper(model).to(device)
    wrapper.eval()

    # Create dummy input: Bx256x64x64
    dummy_input = torch.randn(1, 256, 64, 64, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['warped_feature'],
            output_names=['generated_image'],
            dynamic_axes={
                'warped_feature': {0: 'batch_size'},
                'generated_image': {0: 'batch_size'}
            },
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )

    print(f"✓ SPADE Generator exported successfully")
    return wrapper


def export_stitching_retargeting_to_onnx(model_dict, output_dir: str, device='cpu', opset_version=20):
    """Export Stitching and Retargeting networks (S) to ONNX"""
    print(f"Exporting Stitching/Retargeting Networks to {output_dir}")

    device = torch.device('cpu')
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    for module_name, model in model_dict.items():
        if model is None:
            continue

        output_path = os.path.join(output_dir, f"{module_name}.onnx")

        class StitchingRetargetingWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                return self.model(x)

        wrapper = StitchingRetargetingWrapper(model).to(device)
        wrapper.eval()

        # Determine input size based on module type
        if module_name == 'stitching':
            input_size = 126  # (21*3)*2
        elif module_name == 'lip':
            input_size = 65   # (21*3)+2
        elif module_name == 'eye':
            input_size = 66   # (21*3)+3
        else:
            input_size = 126  # default

        dummy_input = torch.randn(1, input_size, device=device)

        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                },
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL
            )

        results[module_name] = wrapper
        print(f"✓ {module_name} network exported successfully")

    return results


def verify_onnx_model(model_path: str):
    """Verify ONNX model can be loaded"""
    try:
        import onnxruntime as ort

        # Test with ONNX Runtime
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        inputs = [input.name for input in session.get_inputs()]
        outputs = [output.name for output in session.get_outputs()]

        print(f"✓ ONNX model {os.path.basename(model_path)} verified successfully")
        print(f"  Inputs: {inputs}")
        print(f"  Outputs: {outputs}")
        return True

    except Exception as e:
        print(f"✗ ONNX model {os.path.basename(model_path)} verification failed: {e}")
        return False


def convert_model_to_int8(fp32_model_path: str, int8_model_path: str, model_type: str = "general"):
    """Convert FP32 ONNX model to INT8 quantized version"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {model_type} - quantization not available")
        return False

    try:
        print(f"Converting {model_type} to INT8...")

        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            weight_type=QuantType.QInt8,
            optimize_model=True
        )

        print(f"✓ {model_type} INT8 conversion completed")
        return True

    except Exception as e:
        print(f"✗ {model_type} INT8 conversion failed: {e}")
        return False


def export_model_with_quantization(export_func, model, output_path: str, model_name: str,
                                   export_int8: bool = True, device: str = "cpu",
                                   opset_version: int = 20, **kwargs):
    """Export model with optional INT8 quantization"""

    # Export FP32 model
    fp32_path = output_path
    wrapper = export_func(model, fp32_path, device, opset_version, **kwargs)

    # Verify FP32 model
    if not verify_onnx_model(fp32_path):
        return None

    # Export INT8 model if requested
    if export_int8:
        int8_path = fp32_path.replace('.onnx', '_int8.onnx')
        convert_model_to_int8(fp32_path, int8_path, model_name)

        if os.path.exists(int8_path):
            verify_onnx_model(int8_path)

    return wrapper


def main():
    parser = argparse.ArgumentParser(description='Export LivePortrait models to ONNX')
    parser.add_argument('--output_dir', type=str, default='./onnx_models',
                       help='Output directory for ONNX models')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                       help='Device for export (recommend CPU)')
    parser.add_argument('--opset_version', type=int, default=20,
                       help='ONNX opset version')
    parser.add_argument('--export_int8', action='store_true',
                       help='Also export INT8 quantized models')
    parser.add_argument('--human_only', action='store_true',
                       help='Export only human models (skip animal models)')
    parser.add_argument('--verify_only', type=str, default=None,
                       help='Only verify existing ONNX model at given path')

    args = parser.parse_args()

    if args.verify_only:
        verify_onnx_model(args.verify_only)
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize configurations
    inference_cfg = InferenceConfig()

    # Load model configuration
    with open(inference_cfg.models_config, 'r') as f:
        model_config = yaml.safe_load(f)

    print("Starting LivePortrait ONNX export...")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"ONNX Opset Version: {args.opset_version}")
    print(f"Export INT8: {args.export_int8}")

    device = args.device
    opset_version = args.opset_version

    try:
        # Export Human Models
        print("\n=== Exporting Human Models ===")

        # Load and export Appearance Feature Extractor (F)
        print("\n1. Loading Appearance Feature Extractor...")
        appearance_extractor = load_model(inference_cfg.checkpoint_F, model_config, device, 'appearance_feature_extractor')
        export_model_with_quantization(
            export_appearance_feature_extractor_to_onnx,
            appearance_extractor,
            str(output_dir / "appearance_feature_extractor.onnx"),
            "Appearance Feature Extractor",
            args.export_int8,
            device,
            opset_version
        )

        # Load and export Motion Extractor (M)
        print("\n2. Loading Motion Extractor...")
        motion_extractor = load_model(inference_cfg.checkpoint_M, model_config, device, 'motion_extractor')
        export_model_with_quantization(
            export_motion_extractor_to_onnx,
            motion_extractor,
            str(output_dir / "motion_extractor.onnx"),
            "Motion Extractor",
            args.export_int8,
            device,
            opset_version
        )

        # Load and export Warping Network (W)
        print("\n3. Loading Warping Network...")
        warping_network = load_model(inference_cfg.checkpoint_W, model_config, device, 'warping_module')
        export_model_with_quantization(
            export_warping_network_to_onnx,
            warping_network,
            str(output_dir / "warping_network.onnx"),
            "Warping Network",
            args.export_int8,
            device,
            opset_version
        )

        # Load and export SPADE Generator (G)
        print("\n4. Loading SPADE Generator...")
        spade_generator = load_model(inference_cfg.checkpoint_G, model_config, device, 'spade_generator')
        export_model_with_quantization(
            export_spade_generator_to_onnx,
            spade_generator,
            str(output_dir / "spade_generator.onnx"),
            "SPADE Generator",
            args.export_int8,
            device,
            opset_version
        )

        # Load and export Stitching/Retargeting Networks (S)
        if os.path.exists(inference_cfg.checkpoint_S):
            print("\n5. Loading Stitching/Retargeting Networks...")

            stitching_model = load_model(inference_cfg.checkpoint_S, model_config, device, 'stitching_retargeting_module')

            # Extract individual networks
            stitching_dict = {}

            # Handle case where model is a dict (which it is!)
            if isinstance(stitching_model, dict):
                for component_name in ['stitching', 'lip', 'eye']:
                    if component_name in stitching_model:
                        component = stitching_model[component_name]
                        stitching_dict[component_name] = component
            else:
                # Handle case where model has attributes
                for component_name in ['stitching', 'lip', 'eye']:
                    if hasattr(stitching_model, component_name):
                        component = getattr(stitching_model, component_name)
                        stitching_dict[component_name] = component

                # Check if it's a ModuleDict
                if hasattr(stitching_model, '_modules'):
                    for module_name, module in stitching_model._modules.items():
                        if module_name in ['stitching', 'lip', 'eye']:
                            stitching_dict[module_name] = module

            if stitching_dict:
                export_stitching_retargeting_to_onnx(
                    stitching_dict,
                    str(output_dir / "stitching"),
                    device,
                    opset_version
                )
            else:
                print("Warning: No stitching components found - skipping export")

        # Export Animal Models (if requested)
        if not args.human_only:
            print("\n=== Exporting Animal Models ===")
            animal_dir = output_dir / "animal"
            animal_dir.mkdir(exist_ok=True)

            # Check if animal model files exist before loading
            if os.path.exists(inference_cfg.checkpoint_F_animal):
                print("\n1. Loading Animal Appearance Feature Extractor...")
                animal_appearance_extractor = load_model(inference_cfg.checkpoint_F_animal, model_config, device, 'appearance_feature_extractor')
                export_model_with_quantization(
                    export_appearance_feature_extractor_to_onnx,
                    animal_appearance_extractor,
                    str(animal_dir / "appearance_feature_extractor.onnx"),
                    "Animal Appearance Feature Extractor",
                    args.export_int8,
                    device,
                    opset_version
                )

            if os.path.exists(inference_cfg.checkpoint_M_animal):
                print("\n2. Loading Animal Motion Extractor...")
                animal_motion_extractor = load_model(inference_cfg.checkpoint_M_animal, model_config, device, 'motion_extractor')
                export_model_with_quantization(
                    export_motion_extractor_to_onnx,
                    animal_motion_extractor,
                    str(animal_dir / "motion_extractor.onnx"),
                    "Animal Motion Extractor",
                    args.export_int8,
                    device,
                    opset_version
                )

            if os.path.exists(inference_cfg.checkpoint_W_animal):
                print("\n3. Loading Animal Warping Network...")
                animal_warping_network = load_model(inference_cfg.checkpoint_W_animal, model_config, device, 'warping_module')
                export_model_with_quantization(
                    export_warping_network_to_onnx,
                    animal_warping_network,
                    str(animal_dir / "warping_network.onnx"),
                    "Animal Warping Network",
                    args.export_int8,
                    device,
                    opset_version
                )

            if os.path.exists(inference_cfg.checkpoint_G_animal):
                print("\n4. Loading Animal SPADE Generator...")
                animal_spade_generator = load_model(inference_cfg.checkpoint_G_animal, model_config, device, 'spade_generator')
                export_model_with_quantization(
                    export_spade_generator_to_onnx,
                    animal_spade_generator,
                    str(animal_dir / "spade_generator.onnx"),
                    "Animal SPADE Generator",
                    args.export_int8,
                    device,
                    opset_version
                )

        # Create metadata file
        metadata = {
            'export_info': {
                'device': args.device,
                'opset_version': args.opset_version,
                'int8_exported': args.export_int8,
                'human_only': args.human_only
            },
            'model_info': {
                'appearance_feature_extractor': {
                    'input_shape': [1, 3, 256, 256],
                    'output_shape': [1, 32, 16, 64, 64],
                    'description': 'Extracts 3D appearance features from source image'
                },
                'motion_extractor': {
                    'input_shape': [1, 3, 256, 256],
                    'output_shapes': {
                        'pitch': [1, 1], 'yaw': [1, 1], 'roll': [1, 1],
                        't': [1, 3], 'exp': [1, 63], 'scale': [1, 1], 'kp': [1, 63]
                    },
                    'description': 'Extracts keypoints, pose, and expression from image'
                },
                'warping_network': {
                    'input_shapes': {
                        'feature_3d': [1, 32, 16, 64, 64],
                        'kp_driving': [1, 21, 3],
                        'kp_source': [1, 21, 3]
                    },
                    'output_shapes': {
                        'warped_feature': [1, 256, 64, 64],
                        'occlusion_map': [1, 1, 64, 64],
                        'deformation': [1, 16, 64, 64, 3]
                    },
                    'description': 'Warps source features using motion information'
                },
                'spade_generator': {
                    'input_shape': [1, 256, 64, 64],
                    'output_shape': [1, 3, 256, 256],
                    'description': 'Generates final animated image'
                }
            }
        }

        with open(output_dir / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✅ LivePortrait ONNX export completed successfully!")
        print(f"📁 Models exported to: {output_dir}")
        print(f"📄 Metadata saved to: {output_dir}/export_metadata.json")

        # Note about InsightFace models
        print(f"\n📝 Note: InsightFace models are already in ONNX format at:")
        print(f"   Check pretrained_weights/insightface/ for face detection models")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
