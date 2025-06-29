#!/usr/bin/env python3
"""
ONNX Export and Processing Script for LivePortrait Models
This script loads existing ONNX models, optimizes them, and converts to different precisions.
Added support for exporting warping_spade directly from PyTorch modules.
"""

import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import json
from pathlib import Path
import shutil
from onnxsim import simplify
import sys
import torch.nn.functional as F

# Add src path for PyTorch export
sys.path.append('src')

# INT8 quantization support
try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.calibrate import CalibrationDataReader
    from onnxruntime.quantization import quantize_static, CalibrationMethod
    INT8_AVAILABLE = True
    print("✓ INT8 quantization support available")
except ImportError:
    INT8_AVAILABLE = False
    print("⚠️ INT8 quantization not available. Install with: pip install onnxruntime")

from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.transformers.fusion_options import FusionOptions
from onnxruntime.transformers.optimizer import optimize_model

class CorrectedWarpingSpadeWrapper(nn.Module):
    """
    Corrected wrapper for warping_spade export from PyTorch
    Uses correct input order: (feature_3d, kp_driving, kp_source)
    """

    def __init__(self, warping_module, spade_generator):
        super().__init__()
        self.warping_module = warping_module
        self.spade_generator = spade_generator

    def forward(self, feature_3d, kp_driving, kp_source):
        """
        Correct parameter order matching original warping_spade.onnx
        """
        # Call warping module with correct parameter mapping
        ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)

        # SPADE decode
        final_image = self.spade_generator(feature=ret_dct['out'])

        return final_image

class CoreMLFriendlyWarpingSpadeWrapper(nn.Module):
    """
    A wrapper for the warping_spade model that replaces problematic
    operations with CoreML-friendly equivalents before exporting to ONNX.
    """
    def __init__(self, warping_module, spade_generator):
        super().__init__()
        self.warping_module = warping_module
        self.spade_generator = spade_generator

        # Replace AvgPool2d with a Conv2d-based equivalent
        self._replace_avg_pool(self.warping_module)

    def _replace_avg_pool(self, module):
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.AvgPool2d):
                # We need to get the in_channels of the module that CONTAINS the AvgPool2d
                # This is a bit tricky, so we'll make an assumption it's the parent's in_channels
                # A more robust way might be needed if the structure is complex.
                # For now, this is a placeholder. Let's find a better way.

                # Let's try to get it from the previous conv layer if possible
                # This is still not robust.
                # The best way is to know the architecture, which we do.
                # The avg_pool is in down_blocks, which have conv layers.

                # This is a hacky way to get the in_channels.
                # It assumes the avg pool is part of a block that has a 'conv' attribute.
                if hasattr(module, 'conv') and hasattr(module.conv, 'in_channels'):
                     in_channels = module.conv.in_channels
                else:
                    # Fallback for other structures. This may need adjustment.
                    # Let's assume the input to the avg_pool has the same channels as the output.
                    # This is often true. We can't know for sure without tracing.
                    # Let's stick with a simpler approach for now.
                    # The wrapper will be specific to this model's known architecture.
                    # Let's remove this dynamic approach and hardcode for simplicity and robustness
                    # within this specific context.
                    pass # We will handle this in the forward pass of the main module.

            elif len(list(child_module.children())) > 0:
                self._replace_avg_pool(child_module)

    def forward(self, feature_3d, kp_driving, kp_source):
        # We will apply CoreML-friendly operations here.
        # This requires re-implementing the forward passes of the sub-modules.

        # --- Warping Module ---
        # The original warping module uses AvgPool which we need to replace.
        # It's better to replace the nn.AvgPool2d layers in the model definition itself,
        # but let's try to do it here for now.

        # This is becoming too complex. Let's revert to a simpler, more direct approach
        # by fixing the ONNX graph, but this time, let's be more careful.
        # The PyTorch-level modifications are too invasive for this script.

        # Let's go back to the ONNX-level fixes, but do it right.
        # I will remove this and go back.
        pass

def apply_coreml_compatibility_fixes(model_path):
    """Apply graph transformations to make the ONNX model CoreML-friendly."""
    print(f"🔧 Applying CoreML compatibility fixes to {model_path}...")
    try:
        model = onnx.load(model_path)

        # Run shape inference first to populate value_info
        try:
            model = onnx.shape_inference.infer_shapes(model)
            print("  - Ran shape inference to ensure all tensor shapes are available.")
        except Exception as e:
            print(f"  - Warning: Shape inference failed: {e}")

        graph = model.graph
        new_graph_nodes = []

        for node in graph.node:
            if node.op_type == 'AveragePool':
                print(f"  ✓ Replacing AveragePool '{node.name}' with Conv.")

                kernel_shape, strides, pads = None, [1, 1], [0, 0, 0, 0]
                for attr in node.attribute:
                    if attr.name == 'kernel_shape': kernel_shape = attr.ints
                    elif attr.name == 'strides': strides = attr.ints
                    elif attr.name == 'pads': pads = attr.ints

                if not kernel_shape:
                    new_graph_nodes.append(node); continue

                input_tensor_name = node.input[0]
                channels = None
                for vi in list(graph.value_info) + list(graph.input):
                    if vi.name == input_tensor_name:
                        shape = vi.type.tensor_type.shape.dim
                        if len(shape) > 1 and shape[1].dim_value > 0:
                            channels = shape[1].dim_value; break

                if not channels:
                    new_graph_nodes.append(node); continue

                print(f"  - Inferred {channels} channels for grouped convolution.")

                k = np.zeros((channels, 1, *kernel_shape), dtype=np.float32)
                k.fill(1.0 / np.prod(kernel_shape))

                conv_kernel_name = node.name + "_kernel"
                conv_kernel_init = onnx.numpy_helper.from_array(k, name=conv_kernel_name)
                graph.initializer.append(conv_kernel_init)

                new_graph_nodes.append(onnx.helper.make_node(
                    'Conv', [node.input[0], conv_kernel_name], node.output,
                    name=node.name + "_conv", strides=strides, pads=pads, group=channels
                ))

            elif node.op_type == 'Resize':
                print(f"  ✓ Modifying Resize node '{node.name}' for CoreML compatibility.")
                attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                attrs.update({'mode': 'nearest', 'coordinate_transformation_mode': 'asymmetric'})
                attrs.pop('cubic_coeff_a', None)
                new_graph_nodes.append(onnx.helper.make_node(
                    'Resize', node.input, node.output, name=node.name, **attrs
                ))
            else:
                new_graph_nodes.append(node)

        del graph.node[:]
        graph.node.extend(new_graph_nodes)

        # --- FIX: Targeted INT64 to INT32 casting ---
        # Only cast for ops that need it for CoreML, like Gather and Concat.
        # Reshape and Unsqueeze (axes) need INT64.
        initializers_to_remove = []
        initializers_to_add = []
        for node in graph.node:
            if node.op_type in ['Gather', 'Concat']:
                for input_name in node.input:
                    for tensor in graph.initializer:
                        if tensor.name == input_name and tensor.data_type == onnx.TensorProto.INT64:
                            if tensor not in initializers_to_remove:
                                print(f"  - Converting initializer '{tensor.name}' to INT32 for {node.op_type} node.")
                                int64_data = onnx.numpy_helper.to_array(tensor)
                                int32_data = int64_data.astype(np.int32)
                                new_tensor = onnx.numpy_helper.from_array(int32_data, name=tensor.name)
                                initializers_to_add.append(new_tensor)
                                initializers_to_remove.append(tensor)

        for tensor in initializers_to_remove:
            graph.initializer.remove(tensor)
        graph.initializer.extend(initializers_to_add)

        onnx.checker.check_model(model)
        onnx.save(model, model_path)
        print(f"✓ CoreML compatibility fixes applied successfully.")
        return True
    except Exception as e:
        print(f"✗ Failed to apply CoreML fixes: {e}")
        import traceback
        traceback.print_exc()
        return False

def export_warping_spade_from_pytorch(output_path):
    """Export warping_spade directly from PyTorch modules with corrected parameter order"""
    print(f"🔧 Exporting warping_spade from PyTorch modules...")

    try:
        from src.config.inference_config import InferenceConfig
        from src.live_portrait_wrapper import LivePortraitWrapper

        # Load PyTorch modules
        cfg = InferenceConfig()
        cfg.flag_force_cpu = True
        wrapper = LivePortraitWrapper(inference_cfg=cfg)

        print("✅ PyTorch modules loaded")

        # Create corrected wrapper
        pytorch_wrapper = CorrectedWarpingSpadeWrapper(
            warping_module=wrapper.warping_module,
            spade_generator=wrapper.spade_generator
        )

        pytorch_wrapper.eval()

        # Fixed input shapes for maximum optimization
        with torch.no_grad():
            feature_3d = torch.randn(1, 32, 16, 64, 64)
            kp_driving = torch.randn(1, 21, 3)  # Correct order
            kp_source = torch.randn(1, 21, 3)

            # Test
            test_output = pytorch_wrapper(feature_3d, kp_driving, kp_source)
            print(f"✅ Test output shape: {test_output.shape}")

        # Export to ONNX with corrected parameter order
        sample_inputs = (feature_3d, kp_driving, kp_source)

        torch.onnx.export(
            pytorch_wrapper,
            sample_inputs,
            output_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['feature_3d', 'kp_driving', 'kp_source'],  # Corrected order
            output_names=['out'],
            # Fixed shapes for maximum optimization - no dynamic axes
        )

        print(f"✅ PyTorch export successful")

        # Optimize the exported model
        model = onnx.load(output_path)
        onnx.checker.check_model(model)

        print(f"🔧 Optimizing exported model...")
        model_simp, check = simplify(
            model,
            check_n=5,
            perform_optimization=True,
            overwrite_input_shapes={
                'feature_3d': [1, 32, 16, 64, 64],
                'kp_driving': [1, 21, 3],
                'kp_source': [1, 21, 3]
            }
        )

        if check:
            onnx.save(model_simp, output_path)
            print(f"✅ PyTorch export and optimization complete")
            if not apply_coreml_compatibility_fixes(output_path):
                 print("❌ CoreML compatibility fixes failed.")
                 return False

            print(f"📊 Final model re-loaded and checked.")
            return True
        else:
            print(f"❌ Optimization failed")
            return False

    except Exception as e:
        print(f"❌ PyTorch export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@torch.no_grad()
def tune_model(
    model_path: str,
    model_type: str,
    fp16: bool
):
    """Optimize ONNX model using ONNX Runtime transformers"""
    model_dir = os.path.dirname(model_path)

    # Set optimization options based on model type
    optimization_options = FusionOptions(model_type)

    # Disable problematic optimizations for LivePortrait models
    optimization_options.enable_group_norm = False
    optimization_options.enable_nhwc_conv = False
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_bias_splitgelu = False
    optimization_options.enable_bias_add = False
    optimization_options.enable_skip_layer_norm = model_type not in ["warping", "spade"]
    optimization_options.enable_gelu = model_type not in ["warping", "spade"]

    optimizer = optimize_model(
        input=model_path,
        model_type=model_type,
        opt_level=0,
        optimization_options=optimization_options,
        use_gpu=False,
        only_onnxruntime=False
    )

    if fp16:
        optimizer.convert_float_to_float16(
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=['RandomNormalLike']
        )

    optimizer.topological_sort()

    # Handle external data file cleanup
    data_location = f"{model_path}.data"
    if os.path.exists(data_location):
        os.remove(data_location)

    onnx.save_model(
        optimizer.model,
        model_path,
        save_as_external_data=False,
        all_tensors_to_one_file=True,
        location=None,
        convert_attribute=False,
    )

def process_existing_onnx_model(input_path, output_path, model_name, model_type="general"):
    """Process an existing ONNX model by copying and optimizing it"""
    print(f"Processing {model_name} from {input_path}")

    # Convert paths to strings
    input_path = str(input_path)
    output_path = str(output_path)

    try:
        # Copy the model to output location
        shutil.copy2(input_path, output_path)

        # Copy external data file if it exists
        input_data = input_path + ".data"
        if os.path.exists(input_data):
            output_data = output_path + ".data"
            shutil.copy2(input_data, output_data)

        # Optimize the model
        tune_model(output_path, model_type, fp16=False)

        # Apply post-processing optimizations
        model = onnx.load(output_path)
        model_simp, check = simplify(model)

        # Save original model as backup
        shutil.copy(output_path, output_path + ".original")
        onnx.save(model_simp, output_path)

        # Apply CoreML fixes if it's the warping_spade model
        if "warping_spade" in output_path:
            apply_coreml_compatibility_fixes(output_path)

        print(f"✓ {model_name} processed successfully")
        return True

    except Exception as e:
        print(f"✗ Failed to process {model_name}: {e}")
        return False

def convert_model_to_fp16(fp32_model_path, fp16_model_path, model_type="general"):
    """Convert FP32 ONNX model to FP16"""
    try:
        print(f"Converting {fp32_model_path} to FP16...")

        # Copy the FP32 model first
        shutil.copy(fp32_model_path, fp16_model_path)

        # Copy external data file if exists
        fp32_data = fp32_model_path + ".data"
        if os.path.exists(fp32_data):
            fp16_data = fp16_model_path + ".data"
            shutil.copy(fp32_data, fp16_data)

        # Apply FP16 conversion using tune_model
        tune_model(fp16_model_path, model_type, fp16=True)

        # Apply post-processing optimizations
        model = onnx.load(fp16_model_path)
        model_simp, check = simplify(model)

        # Save original model as backup
        shutil.copy(fp16_model_path, fp16_model_path + ".original")
        onnx.save(model_simp, fp16_model_path)

        # Apply CoreML fixes if it's the warping_spade model
        if "warping_spade" in fp16_model_path:
            apply_coreml_compatibility_fixes(fp16_model_path)

        print(f"✓ FP16 model saved to {fp16_model_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to FP16: {e}")
        return False

def convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8 using static quantization with QDQ format"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False

    try:
        print(f"Converting {fp32_model_path} to INT8 using static QDQ quantization...")

        from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat
        from onnxruntime.quantization.calibrate import CalibrationDataReader

        class DummyCalibrationDataReader(CalibrationDataReader):
            def __init__(self, model_path):
                self.model_path = model_path
                self.data_generated = False

                # Load model to get input shapes and types
                model = onnx.load(model_path)
                self.input_names = [inp.name for inp in model.graph.input]
                self.input_shapes = {}
                self.input_types = {}

                for inp in model.graph.input:
                    # Get shape
                    shape = []
                    for dim in inp.type.tensor_type.shape.dim:
                        if dim.dim_value > 0:
                            shape.append(dim.dim_value)
                        else:
                            # Use realistic defaults for dynamic dimensions
                            shape.append(1)  # Default batch size

                    self.input_shapes[inp.name] = shape

                    # Get data type
                    elem_type = inp.type.tensor_type.elem_type
                    if elem_type == onnx.TensorProto.FLOAT:
                        self.input_types[inp.name] = np.float32
                    elif elem_type == onnx.TensorProto.INT64:
                        self.input_types[inp.name] = np.int64
                    elif elem_type == onnx.TensorProto.INT32:
                        self.input_types[inp.name] = np.int32
                    else:
                        self.input_types[inp.name] = np.float32

            def get_next(self):
                if not self.data_generated:
                    self.data_generated = True
                    calibration_data = {}
                    for name, shape in self.input_shapes.items():
                        dtype = self.input_types[name]
                        if dtype == np.int64 or dtype == np.int32:
                            calibration_data[name] = np.zeros(shape, dtype=dtype)
                        else:
                            calibration_data[name] = np.random.randn(*shape).astype(dtype)
                    return calibration_data
                else:
                    return None

        calibration_reader = DummyCalibrationDataReader(fp32_model_path)

        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100  # > 100MB threshold

        quantize_static(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            calibration_data_reader=calibration_reader,
            quant_format=QuantFormat.QDQ,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            use_external_data_format=use_external_data,
            calibrate_method=CalibrationMethod.MinMax
        )

        print(f"✓ INT8 QDQ model saved to {int8_model_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8 QDQ: {e}")
        return False

def convert_model_to_int8(fp32_model_path, int8_model_path, model_type="general"):
    """Convert FP32 ONNX model to INT8"""
    if not INT8_AVAILABLE:
        print(f"Skipping INT8 conversion for {fp32_model_path} - onnxruntime quantization not available")
        return False

    try:
        print(f"Converting {fp32_model_path} to INT8...")

        # Try static QDQ quantization first
        if convert_model_to_int8_static_qdq(fp32_model_path, int8_model_path, model_type):
            return True

        print(f"Static QDQ quantization failed, trying dynamic quantization...")

        # Fallback to dynamic quantization
        model_size = os.path.getsize(fp32_model_path)
        use_external_data = model_size > 1024 * 1024 * 100

        quantize_dynamic(
            model_input=fp32_model_path,
            model_output=int8_model_path,
            weight_type=QuantType.QInt8,
            use_external_data_format=use_external_data
        )

        print(f"✓ INT8 model saved to {int8_model_path}")
        return True

    except Exception as e:
        print(f"✗ Failed to convert {fp32_model_path} to INT8: {e}")
        return False

def verify_onnx_model(onnx_path, input_shapes=None):
    """Verify the exported ONNX model"""
    print(f"Verifying ONNX model: {onnx_path}")

    try:
        # For large models with external data, skip protobuf check
        file_size = os.path.getsize(onnx_path)
        is_large_model = file_size > 100 * 1024 * 1024  # > 100MB

        if not is_large_model:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

        # Create ONNX Runtime session to verify it can load
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')

        session = ort.InferenceSession(onnx_path, providers=providers)

        print(f"✓ ONNX model {onnx_path} is valid")
        print(f"  Input names: {[inp.name for inp in session.get_inputs()]}")
        print(f"  Output names: {[out.name for out in session.get_outputs()]}")

        return True

    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        return False

def cleanup_export_directory(output_dir):
    """Clean up export directory, keeping only .onnx, .onnx.data, and config.json files"""
    print(f"\n🧹 Cleaning up export directory: {output_dir}")

    output_path = Path(output_dir)
    if not output_path.exists():
        return

    kept_files = []
    removed_files = []

    for file_path in output_path.iterdir():
        if file_path.is_file():
            filename = file_path.name

            # Keep these files
            if (filename.endswith('.onnx') or
                filename.endswith('.onnx.data') or
                filename == 'onnx_config.json'):
                kept_files.append(filename)
            else:
                # Remove everything else
                try:
                    file_path.unlink()
                    removed_files.append(filename)
                except Exception as e:
                    print(f"⚠️ Failed to remove {filename}: {e}")

    print(f"✓ Kept {len(kept_files)} essential files: {', '.join(kept_files)}")
    if removed_files:
        print(f"🗑️ Removed {len(removed_files)} temporary files")
    else:
        print("📝 No temporary files to remove")

def process_model_with_precisions(input_path, base_path, model_name, model_type, export_fp32=True, export_fp16=False, export_int8=False, from_pytorch=False):
    """Process model in multiple precision formats"""
    success_count = 0
    base_path_str = str(base_path)

    # Create precision-specific paths
    fp32_path = base_path_str
    fp16_path = base_path_str.replace('.onnx', '_fp16.onnx')
    int8_path = base_path_str.replace('.onnx', '_int8.onnx')

    # Process FP32 model first (base model)
    if export_fp32 or export_fp16 or export_int8:
        try:
            print(f"\n=== Processing {model_name} (FP32) ===")

            # Check if we should export from PyTorch
            if from_pytorch and model_name == "warping_spade":
                print(f"🔧 Exporting {model_name} from PyTorch modules...")
                if export_warping_spade_from_pytorch(fp32_path):
                    print(f"✅ PyTorch export successful")
                else:
                    print(f"❌ PyTorch export failed")
                    return 0
            else:
                # Use existing ONNX processing
                if not process_existing_onnx_model(input_path, fp32_path, model_name, model_type):
                    print(f"✗ {model_name} FP32 processing failed")
                    return 0

            if verify_onnx_model(fp32_path):
                if export_fp32:
                    success_count += 1
                    print(f"✓ {model_name} FP32 processing successful")

                # Convert to FP16 if requested
                if export_fp16:
                    if convert_model_to_fp16(fp32_path, fp16_path, model_type):
                        if verify_onnx_model(fp16_path):
                            success_count += 1
                            print(f"✓ {model_name} FP16 conversion successful")
                        else:
                            print(f"✗ {model_name} FP16 model verification failed")
                    else:
                        print(f"✗ {model_name} FP16 conversion failed")

                # Convert to INT8 if requested
                if export_int8:
                    if convert_model_to_int8(fp32_path, int8_path, model_type):
                        if verify_onnx_model(int8_path):
                            success_count += 1
                            print(f"✓ {model_name} INT8 quantization successful")
                        else:
                            print(f"✗ {model_name} INT8 model verification failed")
                    else:
                        print(f"✗ {model_name} INT8 quantization failed")

                # Remove FP32 if not requested (was only needed for conversion)
                if not export_fp32 and os.path.exists(fp32_path):
                    os.remove(fp32_path)
                    # Also remove external data file if exists
                    fp32_data = fp32_path + ".data"
                    if os.path.exists(fp32_data):
                        os.remove(fp32_data)
            else:
                print(f"✗ {model_name} FP32 model verification failed")
        except Exception as e:
            print(f"✗ Failed to process {model_name}: {e}")

    return success_count

def main():
    parser = argparse.ArgumentParser(description="Process LivePortrait ONNX models")
    parser.add_argument("--weights_dir", default="./weights",
                       help="Input directory containing ONNX models")
    parser.add_argument("--output_dir", default="./models/onnx",
                       help="Output directory for processed ONNX models")
    parser.add_argument("--models", nargs="+",
                       choices=["appearance_feature_extractor", "motion_extractor", "warping_spade", "landmark",
                               "stitching", "stitching_eye", "stitching_lip", "det_10g", "2d106det", "all"],
                       default=["all"], help="Models to process")
    parser.add_argument("--precision", choices=["all", "fp32", "fp16", "floating", "int8"],
                       default="floating", help="Precision to export: all (fp32+fp16+int8), fp32, fp16, floating (fp32+fp16), int8")
    parser.add_argument("--from-pytorch", action="store_true",
                       help="Export warping_spade from PyTorch modules instead of existing ONNX (corrects parameter order)")

    args = parser.parse_args()

    # Determine which precisions to export
    export_fp32 = args.precision in ["all", "fp32", "floating"]
    export_fp16 = args.precision in ["all", "fp16", "floating"]
    export_int8 = args.precision in ["all", "int8"] and INT8_AVAILABLE

    if args.precision in ["all", "int8"] and not INT8_AVAILABLE:
        print("⚠️ INT8 quantization requested but onnxruntime quantization not available")
        export_int8 = False

    # Print precision configuration
    precisions = []
    if export_fp32: precisions.append("FP32")
    if export_fp16: precisions.append("FP16")
    if export_int8: precisions.append("INT8")
    print(f"📝 Processing models in precision(s): {', '.join(precisions)}")

    if getattr(args, 'from_pytorch', False):
        print(f"🔧 PyTorch export enabled for warping_spade (corrected parameter order)")

    # Create output directory
    output_dir = Path(args.output_dir)

    # Clean directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    weights_dir = Path(args.weights_dir)
    print(f"Loading models from: {weights_dir}")
    print(f"Output directory: {output_dir}")

    # Define model mappings
    model_mappings = {
        "appearance_feature_extractor": {
            "file": "appearance_feature_extractor.onnx",
            "type": "encoder"
        },
        "motion_extractor": {
            "file": "motion_extractor.onnx",
            "type": "encoder"
        },
        "warping_spade": {
            "file": "warping_spade.onnx",
            "type": "spade"
        },
        "landmark": {
            "file": "landmark.onnx",
            "type": "landmark"
        },
        "stitching": {
            "file": "stitching.onnx",
            "type": "stitching"
        },
        "stitching_eye": {
            "file": "stitching_eye.onnx",
            "type": "stitching"
        },
        "stitching_lip": {
            "file": "stitching_lip.onnx",
            "type": "stitching"
        },
        "det_10g": {
            "file": "det_10g_fixed.onnx",  # Use the fixed version
            "type": "detector"
        },
        "2d106det": {
            "file": "2d106det.onnx",
            "type": "detector"
        }
    }

    models_to_process = args.models
    if "all" in models_to_process:
        models_to_process = list(model_mappings.keys())

    try:
        success_count = 0

        for model_name in models_to_process:
            if model_name not in model_mappings:
                print(f"⚠️ Unknown model: {model_name}")
                continue

            model_info = model_mappings[model_name]
            input_path = weights_dir / model_info["file"]
            output_path = output_dir / f"{model_name}.onnx"

            from_pytorch = getattr(args, 'from_pytorch', False) and model_name == "warping_spade"

            if from_pytorch:
                print(f"🔧 Will export {model_name} from PyTorch modules")
            elif not input_path.exists():
                print(f"⚠️ Model file not found: {input_path}")
                continue

            try:
                success_count += process_model_with_precisions(
                    input_path, output_path, model_name, model_info["type"],
                    export_fp32, export_fp16, export_int8, from_pytorch=from_pytorch
                )
            except Exception as e:
                print(f"Failed to process {model_name}: {e}")

        print(f"\n✓ Successfully processed {success_count} model variants")
        print(f"Output directory: {output_dir}")

        # Save configuration
        config = {
            "source_weights_dir": str(weights_dir),
            "processed_models": models_to_process,
            "precision": args.precision,
            "from_pytorch": getattr(args, 'from_pytorch', False),
            "precisions_processed": {
                "fp32": export_fp32,
                "fp16": export_fp16,
                "int8": export_int8
            },
            "int8_available": INT8_AVAILABLE,
            "success_count": success_count,
            "model_mappings": model_mappings
        }

        config_path = output_dir / "onnx_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuration saved to: {config_path}")

        # Clean up temporary files
        cleanup_export_directory(output_dir)

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
