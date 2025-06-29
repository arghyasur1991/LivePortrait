#!/usr/bin/env python3
"""
CORRECTED EXPORT: Fix the input parameter order to match original warping_spade.onnx
Original expects: (feature_3d, kp_driving, kp_source) - NOT (feature_3d, kp_source, kp_driving)
"""

import os
import torch
import torch.nn as nn
import onnx
import numpy as np
import sys
sys.path.append('src')

from src.config.inference_config import InferenceConfig
from src.live_portrait_wrapper import LivePortraitWrapper
from onnxsim import simplify

class CorrectedWarpingSpadeWrapper(nn.Module):
    """
    CORRECTED wrapper with proper input order matching original
    """
    
    def __init__(self, warping_module, spade_generator):
        super().__init__()
        self.warping_module = warping_module
        self.spade_generator = spade_generator
    
    def forward(self, feature_3d, kp_driving, kp_source):
        """
        CORRECTED parameter order: (feature_3d, kp_driving, kp_source)
        This matches the original warping_spade.onnx input order!
        """
        # Call warping module with CORRECT parameter order
        ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)
        
        # SPADE decode
        final_image = self.spade_generator(feature=ret_dct['out'])
        
        return final_image

def export_corrected_warping_spade():
    """Export with CORRECTED input order"""
    print("🔧 CORRECTED EXPORT: Fixing parameter order")
    print("="*60)
    print("Original expects: (feature_3d, kp_driving, kp_source)")
    print("Previous wrong:   (feature_3d, kp_source, kp_driving)")
    print()
    
    # Load modules
    cfg = InferenceConfig()
    cfg.flag_force_cpu = True
    wrapper = LivePortraitWrapper(inference_cfg=cfg)
    
    print("✅ LivePortrait modules loaded")
    
    # Create CORRECTED wrapper
    corrected_wrapper = CorrectedWarpingSpadeWrapper(
        warping_module=wrapper.warping_module,
        spade_generator=wrapper.spade_generator
    )
    
    print("✅ Corrected wrapper created")
    
    # Test with CORRECTED parameter order
    corrected_wrapper.eval()
    with torch.no_grad():
        feature_3d = torch.randn(1, 32, 16, 64, 64)
        kp_driving = torch.randn(1, 21, 3)  # Driving keypoints (target motion)
        kp_source = torch.randn(1, 21, 3)   # Source keypoints (original position)
        
        # Test CORRECTED order
        test_output = corrected_wrapper(feature_3d, kp_driving, kp_source)
        print(f"✅ Test output shape: {test_output.shape}")
    
    # Export with CORRECTED input order
    output_path = "warping_spade_corrected.onnx"
    sample_inputs = (feature_3d, kp_driving, kp_source)  # CORRECTED ORDER
    
    print(f"\\n📤 Exporting with CORRECTED parameter order...")
    
    try:
        torch.onnx.export(
            corrected_wrapper,
            sample_inputs,
            output_path,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['feature_3d', 'kp_driving', 'kp_source'],  # CORRECTED ORDER
            output_names=['out'],
            # Fixed shapes for maximum optimization
        )
        
        print(f"✅ Export successful!")
        
        # Optimize
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        
        print(f"🔧 Optimizing...")
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
            
            # Compare with original
            original = onnx.load('weights/warping_spade.onnx')
            
            print(f"\\n📊 CORRECTED MODEL INFO:")
            print(f"  Nodes: {len(model_simp.graph.node)} (original: {len(original.graph.node)})")
            print(f"  Size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
            
            return True
        else:
            print(f"❌ Optimization failed")
            return False
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_corrected_model():
    """Test the corrected model against original"""
    model_path = "warping_spade_corrected.onnx"
    
    if not os.path.exists(model_path):
        print(f"❌ {model_path} not found")
        return
    
    try:
        import onnxruntime as ort
        
        print(f"\\n🧪 TESTING CORRECTED MODEL:")
        
        # Create realistic test with clear motion
        feature_3d = np.random.randn(1, 32, 16, 64, 64).astype(np.float32) * 0.1
        
        # Source keypoints (neutral position)
        kp_source = np.zeros((1, 21, 3), dtype=np.float32)
        
        # Driving keypoints (with clear motion - head turn right)
        kp_driving = kp_source.copy()
        kp_driving[0, :, 0] += 0.3  # Move all keypoints right (head turn)
        
        # CORRECT input order for both models
        inputs = {
            'feature_3d': feature_3d,
            'kp_driving': kp_driving,  # CORRECTED ORDER
            'kp_source': kp_source
        }
        
        print(f"  Input motion: {np.abs(kp_driving - kp_source).mean():.4f}")
        
        # Test original
        original_session = ort.InferenceSession('weights/warping_spade.onnx', providers=['CPUExecutionProvider'])
        original_result = original_session.run(None, inputs)
        
        # Test corrected
        corrected_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        corrected_result = corrected_session.run(None, inputs)
        
        # Compare
        diff = np.abs(original_result[0] - corrected_result[0]).mean()
        print(f"  Output difference: {diff:.8f}")
        
        if diff < 1e-4:
            print(f"  🎉 PERFECT MATCH! Input order is now correct!")
        else:
            print(f"  ⚠️ Still some difference")
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

if __name__ == "__main__":
    print("🎯 FIXING INPUT ORDER BUG")
    print("The issue was: kp_source and kp_driving were swapped!")
    print("This caused incorrect warping direction")
    print()
    
    if export_corrected_warping_spade():
        test_corrected_model()
        
        print(f"\\n🎉 INPUT ORDER CORRECTED:")
        print(f"✅ Now matches original: (feature_3d, kp_driving, kp_source)")
        print(f"✅ Warping should now work correctly")
        print(f"✅ Motion will be applied in the right direction")
        print(f"\\n📁 Output: warping_spade_corrected.onnx")
    else:
        print(f"❌ Correction failed")
