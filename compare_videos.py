#!/usr/bin/env python3

import cv2
import numpy as np

def compare_videos():
    """Compare PyTorch vs ONNX video outputs"""

    # Load videos
    pytorch_video = "animations_pytorch/s6--d0.mp4"
    onnx_video = "animations_onnx_opset20_final/s6_onnx_d0.mp4"  # Latest opset 20 version

    cap_pytorch = cv2.VideoCapture(pytorch_video)
    cap_onnx = cv2.VideoCapture(onnx_video)

    print("=== Video Comparison: PyTorch vs ONNX (Opset 20 + Original Grid Sample) ===")

    # Get video properties
    pytorch_frames = int(cap_pytorch.get(cv2.CAP_PROP_FRAME_COUNT))
    onnx_frames = int(cap_onnx.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"PyTorch frames: {pytorch_frames}")
    print(f"ONNX frames: {onnx_frames}")

    # Compare first few frames
    frame_idx = 0
    pytorch_prev_frame = None
    onnx_prev_frame = None

    max_frames_to_check = min(10, pytorch_frames, onnx_frames)

    while frame_idx < max_frames_to_check:
        ret_pytorch, pytorch_frame = cap_pytorch.read()
        ret_onnx, onnx_frame = cap_onnx.read()

        if not ret_pytorch or not ret_onnx:
            break

        # Convert to grayscale for easier comparison
        pytorch_gray = cv2.cvtColor(pytorch_frame, cv2.COLOR_BGR2GRAY)
        onnx_gray = cv2.cvtColor(onnx_frame, cv2.COLOR_BGR2GRAY)

        print(f"\nFrame {frame_idx}:")
        print(f"PyTorch shape: {pytorch_frame.shape}")
        print(f"ONNX shape: {onnx_frame.shape}")

        # Check if frames are changing (animation)
        if pytorch_prev_frame is not None:
            pytorch_diff = np.mean(np.abs(pytorch_gray.astype(float) - pytorch_prev_frame.astype(float)))
            print(f"PyTorch frame change: {pytorch_diff:.2f}")

        if onnx_prev_frame is not None:
            onnx_diff = np.mean(np.abs(onnx_gray.astype(float) - onnx_prev_frame.astype(float)))
            print(f"ONNX frame change: {onnx_diff:.2f}")

        # Compare difference between PyTorch vs ONNX
        if pytorch_frame.shape == onnx_frame.shape:
            frame_diff = np.mean(np.abs(pytorch_gray.astype(float) - onnx_gray.astype(float)))
            print(f"PyTorch vs ONNX difference: {frame_diff:.2f}")

        pytorch_prev_frame = pytorch_gray
        onnx_prev_frame = onnx_gray
        frame_idx += 1

    cap_pytorch.release()
    cap_onnx.release()

    print("\n=== Summary ===")
    print("✅ If frame change values are > 0, animation is working")
    print("✅ If PyTorch vs ONNX difference is reasonable (~5-20), they're producing similar results")
    print("🎯 FINAL TEST: Opset 20 + Original Grid Sample + Updated Runtime")

if __name__ == "__main__":
    compare_videos()
