"""
GRAPHITE Inference Time Estimator

Simple tool for estimating inference times for GRAPHITE visualization pipeline.
Fixed configuration: 5040×5040 images, V100 GPU, FP32 precision.

Two pipelines:
1. Pipeline 1: GradCAM visualization (training_step_1 + visualization_step_1)
2. Pipeline 2: GRAPHITE fusion (training_step_1 + training_step_2 + visualization_step_2)
"""

from typing import Dict


class GraphiteInferenceEstimator:
    """Simple inference time estimator for GRAPHITE pipeline"""
    
    def __init__(self):
        """Initialize estimator with fixed V100 GPU and 5040x5040 image specs"""
        # Fixed configuration
        self.image_size = (5040, 5040)
        self.num_patches = 484  # 22x22 patches of 224x224
        self.gpu_tflops = 15.7  # V100 FP32 performance
        self.efficiency = 0.75  # Realistic GPU utilization
        
        # Model specifications (based on actual GRAPHITE architecture)
        self.mil_flops = 875.7e9  # GFLOPs for MIL model (ResNet18 + components)
        self.hiergat_flops = 18.3e9  # GFLOPs for HierGAT
        
        # Model parameters (millions)
        self.resnet18_params = 11.2  # Million parameters
        self.mil_classifier_params = 1.3  # Million parameters (projectors + classifier)
        self.hiergat_params = 2.8  # Million parameters (graph attention layers)
        
        # CAM method overhead factors (only GradCAM and FullGrad supported)
        self.cam_factors = {
            'gradcam': 1.2,
            'fullgrad': 2.5
        }
    
    def get_pipeline1_specs(self, cam_method: str = 'fullgrad') -> Dict[str, float]:
        """
        Get detailed specifications for Pipeline 1
        
        Args:
            cam_method: CAM method ('gradcam' or 'fullgrad')
            
        Returns:
            Dictionary with parameters, FLOPs, memory, and timing specs
        """
        if cam_method not in self.cam_factors:
            raise ValueError(f"Unsupported CAM method: {cam_method}. Use 'gradcam' or 'fullgrad'")
        
        # Parameters (training_step_1 only)
        total_params = self.resnet18_params + self.mil_classifier_params  # Million parameters
        
        # FLOPs (MIL model + CAM overhead)
        base_flops = self.mil_flops  # Base inference FLOPs
        cam_factor = self.cam_factors[cam_method]
        total_flops = base_flops * cam_factor  # Total with CAM computation
        
        # Memory usage (FP32)
        model_memory = total_params * 4 / 1000  # GB (4 bytes per parameter)
        feature_memory = self.num_patches * 512 * 4 / 1e9  # GB (feature activations)
        gradient_memory = feature_memory * (cam_factor - 1.0)  # Additional memory for gradients
        total_memory = model_memory + feature_memory + gradient_memory
        
        # Timing
        timing_result = self.estimate_pipeline1_time(cam_method)
        
        return {
            'parameters_millions': total_params,
            'flops_gflops': total_flops / 1e9,
            'memory_gb': total_memory,
            'inference_time_ms': timing_result['total_time_ms'],
            'components': {
                'models': 'training_step_1 (ResNet18 + MIL classifier)',
                'resnet18_params_m': self.resnet18_params,
                'mil_classifier_params_m': self.mil_classifier_params,
                'base_flops_gflops': base_flops / 1e9,
                'cam_overhead_factor': cam_factor
            }
        }
    
    def get_pipeline2_specs(self) -> Dict[str, float]:
        """
        Get detailed specifications for Pipeline 2 (GRAPHITE)
        
        Returns:
            Dictionary with parameters, FLOPs, memory, and timing specs
        """
        # Parameters (training_step_1 + training_step_2)
        total_params = self.resnet18_params + self.mil_classifier_params + self.hiergat_params
        
        # FLOPs breakdown
        mil_flops = self.mil_flops  # training_step_1 base
        hiergat_flops = self.hiergat_flops  # training_step_2 base
        fullgrad_flops = mil_flops * (self.cam_factors['fullgrad'] - 1.0)  # Additional FullGrad computation
        fusion_flops = self.num_patches * 0.2e6  # Fusion processing (0.2 MFLOPs per patch)
        total_flops = mil_flops + hiergat_flops + fullgrad_flops + fusion_flops
        
        # Memory usage (FP32)
        model_memory = total_params * 4 / 1000  # GB (4 bytes per parameter)
        feature_memory = self.num_patches * 512 * 4 / 1e9  # GB (MIL features)
        graph_memory = self.num_patches * 128 * 4 / 1e9  # GB (HierGAT features)
        attention_maps_memory = 3 * (self.num_patches * 4) / 1e9  # GB (3 attention maps)
        gradient_memory = feature_memory * 1.5  # Additional memory for FullGrad gradients
        total_memory = model_memory + feature_memory + graph_memory + attention_maps_memory + gradient_memory
        
        # Timing
        timing_result = self.estimate_pipeline2_time()
        
        return {
            'parameters_millions': total_params,
            'flops_gflops': total_flops / 1e9,
            'memory_gb': total_memory,
            'inference_time_ms': timing_result['total_time_ms'],
            'components': {
                'models': 'training_step_1 + training_step_2 (ResNet18 + MIL + HierGAT)',
                'resnet18_params_m': self.resnet18_params,
                'mil_classifier_params_m': self.mil_classifier_params,
                'hiergat_params_m': self.hiergat_params,
                'mil_flops_gflops': mil_flops / 1e9,
                'hiergat_flops_gflops': hiergat_flops / 1e9,
                'fullgrad_flops_gflops': fullgrad_flops / 1e9,
                'fusion_flops_gflops': fusion_flops / 1e9,
                'attention_maps': 3  # MIL + FullGrad + Multi-level
            }
        }
    
    def compare_pipeline_specs(self, cam_method: str = 'fullgrad') -> Dict[str, float]:
        """
        Compare specifications between Pipeline 1 and GRAPHITE
        
        Args:
            cam_method: CAM method for Pipeline 1 comparison
            
        Returns:
            Dictionary with detailed comparison
        """
        p1_specs = self.get_pipeline1_specs(cam_method)
        p2_specs = self.get_pipeline2_specs()
        
        return {
            'pipeline1_vs_graphite': {
                'parameters_ratio': p2_specs['parameters_millions'] / p1_specs['parameters_millions'],
                'flops_ratio': p2_specs['flops_gflops'] / p1_specs['flops_gflops'],
                'memory_ratio': p2_specs['memory_gb'] / p1_specs['memory_gb'],
                'time_ratio': p2_specs['inference_time_ms'] / p1_specs['inference_time_ms']
            },
            'pipeline1': {
                'parameters_m': p1_specs['parameters_millions'],
                'flops_gflops': p1_specs['flops_gflops'],
                'memory_gb': p1_specs['memory_gb'],
                'time_ms': p1_specs['inference_time_ms'],
                'cam_method': cam_method
            },
            'graphite': {
                'parameters_m': p2_specs['parameters_millions'],
                'flops_gflops': p2_specs['flops_gflops'],
                'memory_gb': p2_specs['memory_gb'],
                'time_ms': p2_specs['inference_time_ms'],
                'cam_method': 'fullgrad'
            },
            'efficiency_metrics': {
                'p1_flops_per_ms': p1_specs['flops_gflops'] / p1_specs['inference_time_ms'],
                'graphite_flops_per_ms': p2_specs['flops_gflops'] / p2_specs['inference_time_ms'],
                'p1_params_per_gflop': p1_specs['parameters_millions'] / p1_specs['flops_gflops'],
                'graphite_params_per_gflop': p2_specs['parameters_millions'] / p2_specs['flops_gflops']
            }
        }
    
    def estimate_pipeline1_time(self, cam_method: str = 'fullgrad') -> Dict[str, float]:
        """
        Estimate Pipeline 1: GradCAM visualization time
        
        Components: training_step_1 + visualization_step_1
        
        Args:
            cam_method: CAM method ('gradcam' or 'fullgrad')
            
        Returns:
            Dictionary with timing results
        """
        if cam_method not in self.cam_factors:
            raise ValueError(f"Unsupported CAM method: {cam_method}. Use 'gradcam' or 'fullgrad'")
            
        # Base MIL inference time
        base_time_ms = (self.mil_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
        # Apply CAM overhead
        cam_factor = self.cam_factors[cam_method]
        total_time_ms = base_time_ms * cam_factor
        
        return {
            'total_time_ms': total_time_ms,
            'base_inference_ms': base_time_ms,
            'cam_overhead_ms': total_time_ms - base_time_ms,
            'cam_method': cam_method,
            'description': 'GradCAM visualization (training_step_1 + visualization_step_1)'
        }
    
    def estimate_pipeline2_time(self) -> Dict[str, float]:
        """
        Estimate Pipeline 2: GRAPHITE fusion time
        
        Components:
        1. MIL attention map (training_step_1)
        2. CAM map using FullGrad (training_step_1) 
        3. Multi-level Fusion map (training_step_1 + training_step_2)
        4. Final Fusion (combine multilevel + MIL + FullGrad results)
        5. Post-processing (visualization rendering)
        
        Returns:
            Dictionary with detailed timing breakdown
        """
        # Base model inference time for training_step_1 (MIL)
        mil_inference_time = (self.mil_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
        # training_step_2 (HierGAT) inference time
        hiergat_inference_time = (self.hiergat_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
        # 1. MIL attention map generation (using training_step_1 results)
        mil_attention_map_time = self.num_patches * 0.03  # Extract attention from MIL model
        
        # 2. CAM map using FullGrad (separate computation on training_step_1)
        # This requires a complete FullGrad computation on the MIL model
        fullgrad_cam_base = mil_inference_time  # Base inference needed for gradients
        fullgrad_cam_gradient = mil_inference_time * (self.cam_factors['fullgrad'] - 1.0)  # Additional gradient computation
        fullgrad_cam_map_time = fullgrad_cam_base + fullgrad_cam_gradient
        
        # 3. Multi-level Fusion map (training_step_1 + training_step_2)
        # This uses both MIL and HierGAT results
        multilevel_level_generation = self.num_patches * 0.05  # Extract HierGAT Level 0/1/2 maps
        multilevel_fusion_processing = self.num_patches * 0.08  # Gaussian smoothing + weighted combination
        multilevel_fusion_map_time = multilevel_level_generation + multilevel_fusion_processing
        
        # 4. Final Fusion (combine the three maps: multilevel + MIL + FullGrad)
        final_fusion_time = self.num_patches * 0.1  # Combine 3 attention maps
        
        # 5. Post-processing (visualization rendering)
        post_processing_time = 100.0 + (self.num_patches * 0.2)  # Rendering and visualization
        
        # Total time (note: MIL inference is used for both MIL attention and FullGrad CAM)
        # HierGAT inference is used for multi-level fusion
        total_time = (mil_inference_time +           # training_step_1 base inference
                      hiergat_inference_time +       # training_step_2 base inference  
                      mil_attention_map_time +       # MIL attention map generation
                      fullgrad_cam_gradient +        # FullGrad gradient computation (additional to base)
                      multilevel_fusion_map_time +   # Multi-level fusion processing
                      final_fusion_time +            # Final fusion of 3 maps
                      post_processing_time)          # Post-processing
        
        return {
            'total_time_ms': total_time,
            'mil_inference_ms': mil_inference_time,
            'hiergat_inference_ms': hiergat_inference_time,
            'mil_attention_map_ms': mil_attention_map_time,
            'fullgrad_cam_map_ms': fullgrad_cam_gradient,  # Only the additional gradient computation
            'multilevel_fusion_map_ms': multilevel_fusion_map_time,
            'final_fusion_ms': final_fusion_time,
            'post_processing_ms': post_processing_time,
            'cam_method': 'fullgrad',
            'description': 'GRAPHITE fusion: MIL attention + FullGrad CAM + Multi-level fusion → Final fusion'
        }
    
    def compare_pipelines(self, cam_method: str = 'fullgrad') -> Dict[str, float]:
        """
        Compare Pipeline 1 vs Pipeline 2 (GRAPHITE)
        
        Args:
            cam_method: CAM method for Pipeline 1 ('gradcam' or 'fullgrad')
            
        Returns:
            Dictionary with comparison results
        """
        p1_results = self.estimate_pipeline1_time(cam_method)
        p2_results = self.estimate_pipeline2_time()
        
        complexity_ratio = p2_results['total_time_ms'] / p1_results['total_time_ms']
        
        return {
            'pipeline1_ms': p1_results['total_time_ms'],
            'pipeline2_ms': p2_results['total_time_ms'],
            'complexity_ratio': complexity_ratio,
            'speed_advantage_p1': f"{complexity_ratio:.1f}x faster",
            'pipeline1_cam_method': cam_method,
            'pipeline2_cam_method': 'fullgrad',
            'summary': f"Pipeline 1 ({cam_method}): {p1_results['total_time_ms']:.0f}ms, GRAPHITE: {p2_results['total_time_ms']:.0f}ms"
        }


# Simple factory function
def create_estimator() -> GraphiteInferenceEstimator:
    """Create a GRAPHITE inference estimator"""
    return GraphiteInferenceEstimator()


# Example usage
if __name__ == "__main__":
    estimator = create_estimator()
    
    print("GRAPHITE Inference Time Estimates (5040×5040, V100, FP32)")
    print("=" * 60)
    
    # Compare pipeline specifications
    specs_comparison = estimator.compare_pipeline_specs('fullgrad')
    print(f"\nPipeline Specifications Comparison:")
    print(f"  Pipeline 1: {specs_comparison['pipeline1']['parameters_m']:.1f}M params, {specs_comparison['pipeline1']['flops_gflops']:.0f} GFLOPs, {specs_comparison['pipeline1']['memory_gb']:.2f} GB")
    print(f"  GRAPHITE:   {specs_comparison['graphite']['parameters_m']:.1f}M params, {specs_comparison['graphite']['flops_gflops']:.0f} GFLOPs, {specs_comparison['graphite']['memory_gb']:.2f} GB")
    
    ratios = specs_comparison['pipeline1_vs_graphite']
    print(f"\nComplexity Ratios (GRAPHITE vs Pipeline 1):")
    print(f"  Parameters: {ratios['parameters_ratio']:.1f}x")
    print(f"  FLOPs:      {ratios['flops_ratio']:.1f}x") 
    print(f"  Memory:     {ratios['memory_ratio']:.1f}x")
    print(f"  Time:       {ratios['time_ratio']:.1f}x")
    
    # Compare pipelines with FullGrad for Pipeline 1
    comparison = estimator.compare_pipelines('fullgrad')
    print(f"\nTiming Comparison:")
    print(f"  Pipeline 1 (FullGrad):    {comparison['pipeline1_ms']:.0f} ms")
    print(f"  GRAPHITE (FullGrad):      {comparison['pipeline2_ms']:.0f} ms")
    print(f"  Complexity ratio:         {comparison['complexity_ratio']:.1f}x") 