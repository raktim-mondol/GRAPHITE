#!/usr/bin/env python3
"""
Simple test script for GRAPHITE inference time estimation.
Tests Pipeline 1 with GradCAM and FullGrad methods.
Tests Pipeline 2 (GRAPHITE) with fixed FullGrad method.
Includes detailed specifications analysis (parameters, FLOPs, memory).
"""

from inference_time_estimator import create_estimator


def main():
    print("GRAPHITE Inference Time Analysis")
    print("=" * 50)
    print("Configuration: 5040×5040 images, V100 GPU, FP32")
    print()
    
    estimator = create_estimator()
    
    # Pipeline specifications comparison
    print("Pipeline Specifications (justifying inference times)")
    print("=" * 50)
    
    specs_comparison = estimator.compare_pipeline_specs('fullgrad')
    
    print("Pipeline 1 (FullGrad) Specifications:")
    print(f"  Parameters:    {specs_comparison['pipeline1']['parameters_m']:.1f} M")
    print(f"  FLOPs:         {specs_comparison['pipeline1']['flops_gflops']:.0f} GFLOPs")
    print(f"  Memory:        {specs_comparison['pipeline1']['memory_gb']:.2f} GB")
    print(f"  Time:          {specs_comparison['pipeline1']['time_ms']:.0f} ms")
    
    print("\nGRAPHITE Specifications:")
    print(f"  Parameters:    {specs_comparison['graphite']['parameters_m']:.1f} M")
    print(f"  FLOPs:         {specs_comparison['graphite']['flops_gflops']:.0f} GFLOPs")
    print(f"  Memory:        {specs_comparison['graphite']['memory_gb']:.2f} GB")
    print(f"  Time:          {specs_comparison['graphite']['time_ms']:.0f} ms")
    
    ratios = specs_comparison['pipeline1_vs_graphite']
    print("\nComplexity Ratios (GRAPHITE vs Pipeline 1):")
    print(f"  Parameters:    {ratios['parameters_ratio']:.1f}x more")
    print(f"  FLOPs:         {ratios['flops_ratio']:.1f}x more")
    print(f"  Memory:        {ratios['memory_ratio']:.1f}x more")
    print(f"  Time:          {ratios['time_ratio']:.1f}x slower")
    
    efficiency = specs_comparison['efficiency_metrics']
    print("\nEfficiency Metrics:")
    print(f"  Pipeline 1:    {efficiency['p1_flops_per_ms']:.1f} GFLOPs/ms")
    print(f"  GRAPHITE:      {efficiency['graphite_flops_per_ms']:.1f} GFLOPs/ms")
    print(f"  Efficiency ratio: {efficiency['p1_flops_per_ms']/efficiency['graphite_flops_per_ms']:.1f}x faster")
    
    print("\n" + "=" * 50)
    
    # Test only GradCAM and FullGrad methods for Pipeline 1
    cam_methods = ['gradcam', 'fullgrad']
    
    print("CAM Method Comparison (Pipeline 1):")
    print("-" * 40)
    for cam in cam_methods:
        p1_result = estimator.estimate_pipeline1_time(cam)
        p1_specs = estimator.get_pipeline1_specs(cam)
        print(f"{cam:>10}: {p1_result['total_time_ms']:>6.0f} ms ({p1_specs['flops_gflops']:>4.0f} GFLOPs)")
    
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
    print(f"MIL inference:     {graphite_result['mil_inference_ms']:>6.0f} ms")
    print(f"HierGAT inference: {graphite_result['hiergat_inference_ms']:>6.0f} ms")
    print(f"MIL attention map: {graphite_result['mil_attention_map_ms']:>6.0f} ms")
    print(f"FullGrad CAM map:  {graphite_result['fullgrad_cam_map_ms']:>6.0f} ms")
    print(f"Multi-level fusion:{graphite_result['multilevel_fusion_map_ms']:>6.0f} ms")
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
    print("Parameter Breakdown:")
    print("-" * 40)
    p1_specs = estimator.get_pipeline1_specs('fullgrad')
    p2_specs = estimator.get_pipeline2_specs()
    
    print("Pipeline 1 Components:")
    print(f"  ResNet18:      {p1_specs['components']['resnet18_params_m']:.1f} M params")
    print(f"  MIL classifier: {p1_specs['components']['mil_classifier_params_m']:.1f} M params")
    print(f"  Total:         {p1_specs['parameters_millions']:.1f} M params")
    
    print("\nGRAPHITE Components:")
    print(f"  ResNet18:      {p2_specs['components']['resnet18_params_m']:.1f} M params")
    print(f"  MIL classifier: {p2_specs['components']['mil_classifier_params_m']:.1f} M params")
    print(f"  HierGAT:       {p2_specs['components']['hiergat_params_m']:.1f} M params")
    print(f"  Total:         {p2_specs['parameters_millions']:.1f} M params")
    
    print()
    print("FLOPs Breakdown:")
    print("-" * 40)
    print("GRAPHITE FLOPs Distribution:")
    print(f"  MIL base:      {p2_specs['components']['mil_flops_gflops']:.0f} GFLOPs")
    print(f"  HierGAT:       {p2_specs['components']['hiergat_flops_gflops']:.0f} GFLOPs")
    print(f"  FullGrad CAM:  {p2_specs['components']['fullgrad_flops_gflops']:.0f} GFLOPs")
    print(f"  Fusion:        {p2_specs['components']['fusion_flops_gflops']:.0f} GFLOPs")
    print(f"  Total:         {p2_specs['flops_gflops']:.0f} GFLOPs")
    
    print()
    print("Recommendations:")
    print("- Use Pipeline 1 for real-time processing (<200ms)")
    print("- Use GRAPHITE for comprehensive research analysis")
    print("- GradCAM: Fastest processing, good for real-time")
    print("- FullGrad: Better quality, moderate speed")
    print("- GRAPHITE: Combines 3 attention maps (MIL + FullGrad + Multi-level)")
    print("- GRAPHITE generates each attention map independently then fuses")
    print(f"- Complexity justified by {ratios['parameters_ratio']:.1f}x params, {ratios['flops_ratio']:.1f}x FLOPs")


if __name__ == "__main__":
    main() 