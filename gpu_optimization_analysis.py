#!/usr/bin/env python3
"""
GPU Optimization Analysis for LivePortrait ONNX Models
Analyze operations falling back to CPU and plan optimizations
"""

import onnx
import onnxruntime as ort
import numpy as np
from collections import defaultdict
import os

def analyze_model_operations(model_path, model_name):
    """Analyze operations and their characteristics in an ONNX model"""
    print(f"\n🔍 ANALYZING {model_name.upper()}")
    print("="*60)
    
    model = onnx.load(model_path)
    
    # Categorize operations by type and characteristics
    op_analysis = defaultdict(list)
    
    for i, node in enumerate(model.graph.node):
        op_type = node.op_type
        
        # Analyze attributes
        attrs = {attr.name: attr for attr in node.attribute}
        
        op_analysis[op_type].append({
            'node_index': i,
            'name': node.name,
            'inputs': node.input,
            'outputs': node.output,
            'attributes': list(attrs.keys())
        })
    
    # Focus on problematic operations from user's list
    problematic_ops = [
        'Flatten', 'BatchNormalization', 'Conv', 'Gather', 
        'Unsqueeze', 'Concat', 'Resize', 'Reshape', 'AveragePool', 'GridSample'
    ]
    
    print("🚨 PROBLEMATIC OPERATIONS ANALYSIS:")
    
    found_issues = {}
    
    for op_type in problematic_ops:
        if op_type in op_analysis:
            nodes = op_analysis[op_type]
            print(f"\n�� {op_type} ({len(nodes)} instances):")
            
            # Analyze specific issues for this operation type
            issues = []
            
            for i, node_info in enumerate(nodes[:2]):  # Show first 2 instances
                node_name = node_info['name'] or f"node_{node_info['node_index']}"
                print(f"  {i+1}. {node_name}")
                
                # Check for specific issues
                if op_type in ['Unsqueeze', 'Concat']:
                    dtype_issues = check_int64_issues(model, node_info)
                    if dtype_issues:
                        issues.extend(dtype_issues)
                
                if op_type in ['Reshape', 'Unsqueeze']:
                    rank_issues = check_rank_issues(model, node_info)
                    if rank_issues:
                        issues.extend(rank_issues)
                        
            found_issues[op_type] = issues
            if len(nodes) > 2:
                print(f"     ... and {len(nodes) - 2} more")
    
    return op_analysis, found_issues

def check_int64_issues(model, node_info):
    """Check for INT64 data type issues"""
    issues = []
    for input_name in node_info['inputs']:
        dtype = get_tensor_dtype(model, input_name)
        if dtype == 7:  # INT64
            issues.append(f"INT64 input: {input_name}")
            print(f"     ⚠️ INT64 input: {input_name}")
    return issues

def check_rank_issues(model, node_info):
    """Check for high rank (>5D) issues"""
    issues = []
    for input_name in node_info['inputs']:
        shape = get_tensor_shape(model, input_name)
        if shape:
            rank = len([d for d in shape if d != 0])  # Count actual dimensions
            if rank > 5:
                issues.append(f"High rank ({rank}D): {input_name}")
                print(f"     ⚠️ High rank ({rank}D): {input_name} - {shape}")
    return issues

def get_tensor_shape(model, tensor_name):
    """Get tensor shape from model"""
    # Check value_info
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    
    # Check inputs
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
    
    # Check initializers
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return list(init.dims)
    
    return None

def get_tensor_dtype(model, tensor_name):
    """Get tensor data type from model"""
    # Check value_info
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return vi.type.tensor_type.elem_type
    
    # Check inputs
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return inp.type.tensor_type.elem_type
    
    # Check initializers
    for init in model.graph.initializer:
        if init.name == tensor_name:
            return init.data_type
    
    return None

def create_gpu_optimization_plan(all_issues):
    """Create actionable GPU optimization plan"""
    print(f"\n�� GPU OPTIMIZATION PLAN")
    print("="*60)
    
    print(f"\n🔥 PRIORITY 1: INT64 DATA TYPE FIXES")
    print(f"   Problem: INT64 inputs not supported on GPU")
    print(f"   Impact: HIGH - affects Unsqueeze/Concat operations")
    print(f"   Solution: Add Cast(to=INT32) nodes before operations")
    print(f"   Risk: LOW - mathematically safe for most use cases")
    
    print(f"\n🟡 PRIORITY 2: RANK REDUCTION (>5D tensors)")
    print(f"   Problem: GPU doesn't support >5D tensor operations")
    print(f"   Impact: MEDIUM - affects Reshape/Unsqueeze operations")
    print(f"   Solution: Redesign tensor manipulation to stay ≤5D")
    print(f"   Risk: MEDIUM - requires architectural changes")
    
    print(f"\n🟢 PRIORITY 3: OPERATION REPLACEMENT")
    print(f"   Problem: Some operations lack GPU implementation")
    print(f"   Impact: VARIABLE - depends on operation")
    print(f"   Solutions:")
    print(f"     • AveragePool -> Conv with averaging kernel")
    print(f"     • Resize -> Upsample or ConvTranspose")
    print(f"     • BatchNorm -> InstanceNorm (if applicable)")
    print(f"   Risk: HIGH - may change model behavior")

def test_current_gpu_utilization():
    """Test current GPU utilization of available models"""
    print(f"\n🧪 CURRENT GPU UTILIZATION TEST")
    print("="*50)
    
    models_to_test = [
        ("weights/warping_spade.onnx", "Original"),
        ("models/onnx/warping_spade.onnx", "Exported"),
        ("warping_spade_corrected.onnx", "Corrected")
    ]
    
    for model_path, model_type in models_to_test:
        if os.path.exists(model_path):
            test_gpu_utilization_single(model_path, model_type)
        else:
            print(f"\n⚠️ {model_type}: {model_path} - NOT FOUND")

def test_gpu_utilization_single(model_path, model_name):
    """Test GPU utilization for a single model"""
    try:
        print(f"\n📊 {model_name}: {model_path}")
        
        # Load model
        model = onnx.load(model_path)
        total_nodes = len(model.graph.node)
        
        # Test with CoreML provider
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        so = ort.SessionOptions()
        so.log_severity_level = 3  # Reduce verbosity for this test
        
        session = ort.InferenceSession(model_path, so, providers=providers)
        
        # Create test inputs
        inputs = {
            'feature_3d': np.random.randn(1, 32, 16, 64, 64).astype(np.float32),
            'kp_driving': np.random.randn(1, 21, 3).astype(np.float32),
            'kp_source': np.random.randn(1, 21, 3).astype(np.float32)
        }
        
        # Test inference
        result = session.run(None, inputs)
        
        print(f"  ✅ Working - {total_nodes} total nodes")
        print(f"  Active providers: {session.get_providers()}")
        print(f"  Output shape: {result[0].shape}")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:100]}...")

if __name__ == "__main__":
    print("🎯 GPU OPTIMIZATION ANALYSIS FOR LIVEPORTRAIT")
    print("="*60)
    print("Analyzing CPU fallback operations for GPU optimization")
    
    # Test models with correct paths
    models = [
        ("weights/warping_spade.onnx", "original"),
        ("models/onnx/warping_spade.onnx", "exported")
    ]
    
    all_analyses = {}
    all_issues = {}
    
    for model_path, model_name in models:
        if os.path.exists(model_path):
            try:
                analysis, issues = analyze_model_operations(model_path, model_name)
                all_analyses[model_name] = analysis
                all_issues[model_name] = issues
            except Exception as e:
                print(f"\n❌ Failed to analyze {model_name}: {e}")
        else:
            print(f"\n⚠️ {model_path} not found - skipping")
    
    # Create optimization plan
    if all_issues:
        create_gpu_optimization_plan(all_issues)
    
    # Test current GPU utilization
    test_current_gpu_utilization()
    
    print(f"\n📋 IMPLEMENTATION PRIORITY:")
    print(f"1. 🔧 Add INT32 casting for keypoint operations")
    print(f"2. 📐 Reduce tensor ranks to ≤5D")
    print(f"3. 🔄 Replace unsupported operations")
    print(f"4. 📊 Measure GPU utilization improvements")
    print(f"\nGoal: Increase GPU utilization from ~73% to ~90%")
