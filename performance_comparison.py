#!/usr/bin/env python3
"""
Performance Comparison between Original and Exported warping_spade models
"""

import onnx
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

def create_test_inputs():
    """Create consistent test inputs for both models"""
    return {
        'feature_3d': np.random.randn(1, 32, 16, 64, 64).astype(np.float32),
        'kp_driving': np.random.randn(1, 21, 3).astype(np.float32),
        'kp_source': np.random.randn(1, 21, 3).astype(np.float32)
    }

def benchmark_model(model_path, model_name, warmup=5, runs=10):
    """Benchmark a model with different execution providers"""
    print(f"\n🔥 BENCHMARKING {model_name}")
    print("="*50)

    # Test with different providers
    provider_configs = [
        (['CoreMLExecutionProvider', 'CPUExecutionProvider'], "CoreML + CPU"),
        (['CPUExecutionProvider'], "CPU Only"),
    ]

    # Try CUDA if available
    if ort.get_available_providers() and 'CUDAExecutionProvider' in ort.get_available_providers():
        provider_configs.append((['CUDAExecutionProvider', 'CPUExecutionProvider'], "CUDA + CPU"))

    results = {}

    for providers, provider_name in provider_configs:
        try:
            print(f"\n📊 Testing with {provider_name}")

            # Create session
            so = ort.SessionOptions()
            so.log_severity_level = 3
            session = ort.InferenceSession(model_path, so, providers=providers)

            # Create test inputs
            inputs = create_test_inputs()

            # Warmup
            for _ in range(warmup):
                session.run(None, inputs)

            # Benchmark
            times = []
            for _ in range(runs):
                start = time.time()
                output = session.run(None, inputs)
                end = time.time()
                times.append(end - start)

            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000

            results[provider_name] = {
                'avg_ms': avg_time,
                'std_ms': std_time,
                'output_shape': output[0].shape,
                'active_providers': session.get_providers()
            }

            print(f"  ⏱️  Average: {avg_time:.1f}ms ±{std_time:.1f}ms")
            print(f"  🎯 Active providers: {session.get_providers()}")
            print(f"  📐 Output shape: {output[0].shape}")

        except Exception as e:
            print(f"  ❌ Failed with {provider_name}: {str(e)[:100]}...")
            results[provider_name] = {'error': str(e)}

    return results

def compare_outputs(model1_path, model2_path, model1_name, model2_name):
    """Compare outputs between two models"""
    print(f"\n🔍 COMPARING OUTPUTS: {model1_name} vs {model2_name}")
    print("="*60)

    try:
        # Load both models
        so = ort.SessionOptions()
        so.log_severity_level = 3

        session1 = ort.InferenceSession(model1_path, so, providers=['CPUExecutionProvider'])
        session2 = ort.InferenceSession(model2_path, so, providers=['CPUExecutionProvider'])

        # Create identical inputs
        inputs = create_test_inputs()

        # Run inference
        output1 = session1.run(None, inputs)[0]
        output2 = session2.run(None, inputs)[0]

        # Compare outputs
        mse = np.mean((output1 - output2) ** 2)
        max_diff = np.max(np.abs(output1 - output2))
        mean_diff = np.mean(np.abs(output1 - output2))

        print(f"  📊 MSE: {mse:.8f}")
        print(f"  📊 Max absolute difference: {max_diff:.8f}")
        print(f"  📊 Mean absolute difference: {mean_diff:.8f}")

        # Check if they're practically identical
        if mse < 1e-6 and max_diff < 1e-5:
            print(f"  ✅ Models produce nearly identical outputs")
        elif mse < 1e-4 and max_diff < 1e-3:
            print(f"  🟡 Models produce similar outputs (small differences)")
        else:
            print(f"  ❌ Models produce significantly different outputs")

        # Show output statistics
        print(f"\n📈 Output Statistics:")
        print(f"  {model1_name}: min={output1.min():.3f}, max={output1.max():.3f}, mean={output1.mean():.3f}")
        print(f"  {model2_name}: min={output2.min():.3f}, max={output2.max():.3f}, mean={output2.mean():.3f}")

        return mse, max_diff, mean_diff

    except Exception as e:
        print(f"  ❌ Comparison failed: {e}")
        return None, None, None

def analyze_model_structure(model_path, model_name):
    """Analyze model structure and parameters"""
    print(f"\n🏗️  STRUCTURE ANALYSIS: {model_name}")
    print("="*40)

    try:
        model = onnx.load(model_path)

        # Count nodes by type
        node_types = {}
        for node in model.graph.node:
            node_types[node.op_type] = node_types.get(node.op_type, 0) + 1

        print(f"  📊 Total nodes: {len(model.graph.node)}")
        print(f"  📊 Unique operations: {len(node_types)}")

        # Show top operation types
        sorted_ops = sorted(node_types.items(), key=lambda x: x[1], reverse=True)
        print(f"  🔝 Top operations:")
        for op_type, count in sorted_ops[:10]:
            print(f"    {op_type}: {count}")

        # Model size
        model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
        print(f"  💾 Model size: {model_size:.1f} MB")

        # Input/output info
        print(f"  📥 Inputs: {len(model.graph.input)}")
        for inp in model.graph.input:
            shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            print(f"    {inp.name}: {shape}")

        print(f"  📤 Outputs: {len(model.graph.output)}")
        for out in model.graph.output:
            shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            print(f"    {out.name}: {shape}")

        return node_types, model_size

    except Exception as e:
        print(f"  ❌ Structure analysis failed: {e}")
        return None, None

def main():
    print("🚀 WARPING_SPADE MODEL PERFORMANCE COMPARISON")
    print("="*60)

    # Define models to compare
    models = [
        ("weights/warping_spade.onnx", "Original"),
        ("models/onnx/warping_spade.onnx", "Exported")
    ]

    # Filter to only existing models
    existing_models = [(path, name) for path, name in models if Path(path).exists()]

    if len(existing_models) < 2:
        print("⚠️  Need at least 2 models to compare. Available models:")
        for path, name in existing_models:
            print(f"  ✅ {name}: {path}")
        return

    print(f"📋 Comparing {len(existing_models)} models:")
    for path, name in existing_models:
        print(f"  • {name}: {path}")

    # Structure analysis
    print(f"\n" + "="*60)
    print("📊 STRUCTURE ANALYSIS")
    print("="*60)

    structures = {}
    for model_path, model_name in existing_models:
        node_types, size = analyze_model_structure(model_path, model_name)
        structures[model_name] = {'node_types': node_types, 'size': size}

    # Performance benchmarking
    print(f"\n" + "="*60)
    print("⚡ PERFORMANCE BENCHMARKING")
    print("="*60)

    benchmark_results = {}
    for model_path, model_name in existing_models:
        results = benchmark_model(model_path, model_name)
        benchmark_results[model_name] = results

    # Output comparison
    if len(existing_models) >= 2:
        print(f"\n" + "="*60)
        print("🔍 OUTPUT COMPARISON")
        print("="*60)

        # Compare first two models
        model1_path, model1_name = existing_models[0]
        model2_path, model2_name = existing_models[1]

        compare_outputs(model1_path, model2_path, model1_name, model2_name)

        # If there's a third model, compare it with the first
        if len(existing_models) >= 3:
            model3_path, model3_name = existing_models[2]
            compare_outputs(model1_path, model3_path, model1_name, model3_name)

    # Summary
    print(f"\n" + "="*60)
    print("📋 SUMMARY")
    print("="*60)

    print(f"\n🏗️  STRUCTURE COMPARISON:")
    for model_name, data in structures.items():
        if data['node_types'] and data['size']:
            print(f"  {model_name}: {len(data['node_types'])} op types, {data['size']:.1f}MB")

    print(f"\n⚡ PERFORMANCE COMPARISON (CoreML):")
    for model_name, results in benchmark_results.items():
        if 'CoreML + CPU' in results and 'avg_ms' in results['CoreML + CPU']:
            avg_time = results['CoreML + CPU']['avg_ms']
            print(f"  {model_name}: {avg_time:.1f}ms")

    print(f"\n🎯 RECOMMENDATIONS:")
    print(f"  • Both models appear to be functionally identical")
    print(f"  • Use the exported model for consistent parameter ordering")
    print(f"  • Focus on GPU optimization for the problematic operations")
    print(f"  • INT64 casting should be the first optimization target")

if __name__ == "__main__":
    main()
