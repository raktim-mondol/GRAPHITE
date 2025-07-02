#!/usr/bin/env python3
"""
Simple test script for GRAPHITE inference time estimation.
Tests Pipeline 1 with GradCAM and FullGrad methods.
Tests Pipeline 2 (GRAPHITE) with fixed FullGrad method.
"""

from inference_time_estimator import create_estimator


def main():
    print("GRAPHITE Inference Time Analysis")
    print("=" * 50)
    print("Configuration: 5040×5040 images, V100 GPU, FP32")
    print()
    
    estimator = create_estimator()
    
    # Test only GradCAM and FullGrad methods for Pipeline 1
    cam_methods = ['gradcam', 'fullgrad']
    
    print("CAM Method Comparison (Pipeline 1):")
    print("-" * 40)
    for cam in cam_methods:
        p1_result = estimator.estimate_pipeline1_time(cam)
        print(f"{cam:>10}: {p1_result['total_time_ms']:>6.0f} ms")
    
    print()
    print("Pipeline Comparison (FullGrad vs GRAPHITE):")
    print("-" * 40)
    
    # Detailed comparison with FullGrad
    comparison = estimator.compare_pipelines('fullgrad')
    p1_result = estimator.estimate_pipeline1_time('fullgrad')
    graphite_result = estimator.estimate_pipeline2_time()
    
    print(f"Pipeline 1: {comparison['pipeline1_ms']:>6.0f} ms (FullGrad)")
    print(f"GRAPHITE:   {comparison['pipeline2_ms']:>6.0f} ms (FullGrad)")
    print(f"Ratio:      {comparison['complexity_ratio']:>6.1f}x more complex")
    
    print()
    print("GRAPHITE Detailed Breakdown:")
    print("-" * 40)
    print(f"Core inference:    {graphite_result['core_inference_ms']:>6.0f} ms")
    print(f"Multi-level fusion:{graphite_result['multilevel_fusion_ms']:>6.0f} ms")
    print(f"FullGrad CAM:      {graphite_result['fullgrad_cam_ms']:>6.0f} ms")
    print(f"Final fusion:      {graphite_result['final_fusion_ms']:>6.0f} ms")
    print(f"Post-processing:   {graphite_result['post_processing_ms']:>6.0f} ms")
    print(f"Total:             {graphite_result['total_time_ms']:>6.0f} ms")
    
    print()
    print("Pipeline Comparison (GradCAM vs GRAPHITE):")
    print("-" * 40)
    
    # Additional comparison with GradCAM
    comparison_gradcam = estimator.compare_pipelines('gradcam')
    print(f"Pipeline 1: {comparison_gradcam['pipeline1_ms']:>6.0f} ms (GradCAM)")
    print(f"GRAPHITE:   {comparison_gradcam['pipeline2_ms']:>6.0f} ms (FullGrad)")
    print(f"Ratio:      {comparison_gradcam['complexity_ratio']:>6.1f}x more complex")
    
    print()
    print("Recommendations:")
    print("- Use Pipeline 1 for real-time processing (<200ms)")
    print("- Use GRAPHITE for comprehensive research analysis")
    print("- GradCAM: Fastest processing, good for real-time")
    print("- FullGrad: Better quality, moderate speed")
    print("- GRAPHITE: Always uses FullGrad for best fusion quality")


if __name__ == "__main__":
    main() 