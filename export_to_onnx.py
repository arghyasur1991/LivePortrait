#!/usr/bin/env python3
"""
ONNX Export and Processing Script for LivePortrait Models
This script loads existing ONNX models, optimizes them, and converts to different precisions.
"""

import os
import torch
import onnx
import onnxruntime as ort
import numpy as np
import argparse
import json
from pathlib import Path
import shutil
from onnxsim import simplify

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

def process_model_with_precisions(input_path, base_path, model_name, model_type, export_fp32=True, export_fp16=False, export_int8=False):
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
            if process_existing_onnx_model(input_path, fp32_path, model_name, model_type):
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
            else:
                print(f"✗ {model_name} FP32 processing failed")
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

            if not input_path.exists():
                print(f"⚠️ Model file not found: {input_path}")
                continue

            try:
                success_count += process_model_with_precisions(
                    input_path, output_path, model_name, model_info["type"],
                    export_fp32, export_fp16, export_int8
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
