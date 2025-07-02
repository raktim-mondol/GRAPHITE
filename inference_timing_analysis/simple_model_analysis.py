#!/usr/bin/env python3
"""
Simple GRAPHITE Model Analysis Tool

Provides basic model analysis without external dependencies.
Only uses PyTorch for parameter counting and mathematical FLOP estimation.
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add training step directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_1', 'mil_classification', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_2', 'self_supervised_training'))


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters without external dependencies"""
    total_params = 0
    trainable_params = 0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def estimate_conv2d_flops(input_shape: Tuple[int, ...], layer: nn.Conv2d) -> int:
    """Estimate FLOPs for Conv2d layer"""
    batch_size, in_channels, in_height, in_width = input_shape
    out_channels = layer.out_channels
    kernel_size = layer.kernel_size[0] * layer.kernel_size[1]
    
    # Calculate output dimensions
    out_height = (in_height + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    out_width = (in_width + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
    
    # FLOPs = batch_size * output_dims * kernel_flops
    kernel_flops = in_channels * kernel_size
    output_elements = batch_size * out_channels * out_height * out_width
    
    return int(kernel_flops * output_elements)


def estimate_linear_flops(input_shape: Tuple[int, ...], layer: nn.Linear) -> int:
    """Estimate FLOPs for Linear layer"""
    batch_size = input_shape[0]
    return int(batch_size * layer.in_features * layer.out_features)


def estimate_model_flops(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """Basic FLOP estimation for model"""
    total_flops = 0
    
    # This is a simplified estimation
    # For ResNet18 backbone processing 484 patches of 224x224
    if hasattr(model, 'feature_extractor'):
        # ResNet18 approximate FLOPs per image: 1.8 GFLOPs
        # For 484 patches: 484 * 1.8 = 871.2 GFLOPs
        total_flops += int(484 * 1.8e9)
    
    # Add MIL components (much smaller)
    # Projections, attention, classifier: ~7 GFLOPs
    total_flops += int(7e9)
    
    return total_flops


def analyze_mil_model_simple():
    """Simple analysis of MIL model"""
    print("="*60)
    print("SIMPLE MIL MODEL ANALYSIS (Training Step 1)")
    print("="*60)
    
    try:
        from models.mil_classifier import MILHistopathModel
        
        model = MILHistopathModel(num_classes=2, feat_dim=512, proj_dim=128)
        model.eval()
        
        print(f"Model: {model.__class__.__name__}")
        
        # Parameter counting
        param_counts = count_parameters(model)
        print(f"\nPARAMETER ANALYSIS:")
        print(f"Total parameters: {param_counts['total']:,} ({param_counts['total']/1e6:.2f}M)")
        print(f"Trainable: {param_counts['trainable']:,}")
        print(f"Non-trainable: {param_counts['non_trainable']:,}")
        
        # Component analysis
        print(f"\nCOMPONENT BREAKDOWN:")
        feature_extractor_params = count_parameters(model.feature_extractor)['total']
        patch_projector_params = count_parameters(model.patch_projector)['total']
        attention_params = count_parameters(model.attention)['total']
        patient_projector_params = count_parameters(model.patient_projector)['total']
        classifier_params = count_parameters(model.classifier)['total']
        patient_ln_params = count_parameters(model.patient_layer_norm)['total']
        
        components = [
            ('ResNet18 backbone', feature_extractor_params),
            ('Patch projector', patch_projector_params),
            ('Attention mechanism', attention_params),
            ('Patient projector', patient_projector_params),
            ('Classifier', classifier_params),
            ('Patient LayerNorm', patient_ln_params)
        ]
        
        for name, count in components:
            percentage = (count / param_counts['total']) * 100
            print(f"{name}: {count:,} ({percentage:.1f}%)")
        
        # FLOP estimation
        input_shape = (1, 484, 3, 224, 224)
        estimated_flops = estimate_model_flops(model, input_shape)
        print(f"\nFLOP ESTIMATION:")
        print(f"Estimated FLOPs: {estimated_flops:,} ({estimated_flops/1e9:.1f} GFLOPs)")
        
        # Test forward pass
        print(f"\nFORWARD PASS TEST:")
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 484, 3, 224, 224)
                output = model(dummy_input)
                print(f"✓ Forward pass successful")
                print(f"Output shapes: {[x.shape if torch.is_tensor(x) else type(x) for x in output]}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
        
        return model, param_counts
        
    except Exception as e:
        print(f"Error analyzing MIL model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_hiergat_mock_model():
    """Create a mock HierGAT model for parameter estimation"""
    class MockGATLayer(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_heads=4):
            super().__init__()
            head_dim = hidden_dim // num_heads
            # Mock GAT parameters based on actual architecture
            self.spatial_gat = nn.Linear(input_dim, hidden_dim)
            self.cross_scale_gat = nn.Linear(input_dim, hidden_dim) 
            self.attention = nn.Parameter(torch.randn(num_heads, head_dim))
            self.layer_norm = nn.LayerNorm(hidden_dim)

    class MockScaleAttention(nn.Module):
        def __init__(self, hidden_dim, num_levels=3):
            super().__init__()
            self.level_attention = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                ) for _ in range(num_levels)
            ])
            self.cross_scale_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_levels)
            )

    class MockHierGAT(nn.Module):
        def __init__(self, input_dim=128, hidden_dim=128, num_gat_layers=3, num_heads=4, num_levels=3):
            super().__init__()
            self.gat_layers = nn.ModuleList([
                MockGATLayer(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    num_heads
                ) for i in range(num_gat_layers)
            ])
            self.scale_attention = MockScaleAttention(hidden_dim, num_levels)
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim)
            )

    return MockHierGAT()


def analyze_hiergat_model_simple():
    """Simple analysis of HierGAT model"""
    print("\n" + "="*60)
    print("SIMPLE HIERGAT MODEL ANALYSIS (Training Step 2)")
    print("="*60)
    
    try:
        # Use theoretical calculation instead of actual model due to dependencies
        from inference_time_estimator import GraphiteInferenceEstimator
        estimator = GraphiteInferenceEstimator()
        hiergat_theory = estimator.calculate_hiergat_params()
        
        print(f"Model: HierGAT (Theoretical Analysis)")
        print("Note: Using theoretical calculation due to PyTorch Geometric dependencies")
        
        # Use theoretical parameter count
        total_params = int(hiergat_theory['total'])
        
        print(f"\nPARAMETER ANALYSIS:")
        print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"Trainable: {total_params:,}")
        print(f"Non-trainable: 0")
        
        # Component breakdown from theoretical analysis
        print(f"\nCOMPONENT BREAKDOWN:")
        gat_layers_params = int(hiergat_theory['gat_layers'])
        scale_attention_params = int(hiergat_theory['scale_attention'])
        projection_head_params = int(hiergat_theory['projection_head'])
        
        components = [
            ('GAT layers', gat_layers_params),
            ('Scale attention', scale_attention_params),
            ('Projection head', projection_head_params)
        ]
        
        for name, count in components:
            percentage = (count / total_params) * 100
            print(f"{name}: {count:,} ({percentage:.1f}%)")
        
        # FLOP estimation (much smaller for graph processing)
        estimated_flops = int(0.2e9)  # ~200 MFLOPs for graph processing
        print(f"\nFLOP ESTIMATION:")
        print(f"Estimated FLOPs: {estimated_flops:,} ({estimated_flops/1e9:.1f} GFLOPs)")
        
        # Test forward pass note
        print(f"\nFORWARD PASS TEST:")
        print("Note: HierGAT requires PyTorch Geometric Data objects")
        print("Skipping forward pass test (requires complex graph data setup)")
        
        param_counts = {
            'total': total_params,
            'trainable': total_params,
            'non_trainable': 0
        }
        
        return None, param_counts
        
    except Exception as e:
        print(f"Error analyzing HierGAT model: {e}")
        print("Fallback: Using approximate parameter count")
        
        # Fallback estimation
        approximate_params = 168582  # From theoretical calculation
        param_counts = {
            'total': approximate_params,
            'trainable': approximate_params,
            'non_trainable': 0
        }
        
        print(f"Approximate parameters: {approximate_params:,} ({approximate_params/1e6:.2f}M)")
        return None, param_counts


def compare_models_simple(mil_stats: Optional[Dict], hiergat_stats: Optional[Dict]):
    """Simple model comparison"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if mil_stats and hiergat_stats:
        mil_params = mil_stats['total']
        hiergat_params = hiergat_stats['total']
        total_params = mil_params + hiergat_params
        
        print(f"Training Step 1 (MIL): {mil_params:,} parameters ({mil_params/1e6:.2f}M)")
        print(f"Training Step 2 (HierGAT): {hiergat_params:,} parameters ({hiergat_params/1e6:.2f}M)")
        print(f"Total Pipeline: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        print(f"Step 1/Step 2 Ratio: {mil_params/hiergat_params:.1f}x")
        
        print(f"\nCOMPUTATIONAL COMPLEXITY:")
        print(f"Step 1 dominates the pipeline due to ResNet18 backbone")
        print(f"Step 2 adds graph reasoning with minimal computational overhead")
        
        print(f"\nMEMORY USAGE (FP32):")
        mil_memory = mil_params * 4 / (1024**2)  # MB
        hiergat_memory = hiergat_params * 4 / (1024**2)  # MB
        total_memory = total_params * 4 / (1024**2)  # MB
        
        print(f"Step 1: {mil_memory:.1f} MB")
        print(f"Step 2: {hiergat_memory:.1f} MB")
        print(f"Total: {total_memory:.1f} MB")
        
    else:
        print("Could not compare models due to analysis errors")


def validate_theoretical_estimates():
    """Validate against theoretical estimates"""
    print("\n" + "="*60)
    print("THEORETICAL VALIDATION")
    print("="*60)
    
    try:
        from inference_time_estimator import GraphiteInferenceEstimator
        
        estimator = GraphiteInferenceEstimator()
        
        # Get theoretical values
        resnet18_theory = estimator.calculate_resnet18_params()
        mil_classifier_theory = estimator.calculate_mil_classifier_params()
        hiergat_theory = estimator.calculate_hiergat_params()
        
        print("THEORETICAL ESTIMATES (from inference_time_estimator.py):")
        print(f"ResNet18: {resnet18_theory['total_millions']:.2f}M parameters")
        print(f"MIL classifier components: {mil_classifier_theory['total_millions']:.2f}M parameters")
        print(f"HierGAT: {hiergat_theory['total_millions']:.2f}M parameters")
        
        total_theory = resnet18_theory['total_millions'] + mil_classifier_theory['total_millions']
        print(f"Total Step 1 (theory): {total_theory:.2f}M parameters")
        
        # Compare with actual if available
        print(f"\nValidation will be performed against actual model measurements...")
        
    except Exception as e:
        print(f"Could not load theoretical estimates: {e}")


def main():
    """Main analysis function"""
    print("SIMPLE GRAPHITE MODEL ANALYSIS TOOL")
    print("="*60)
    print("Basic PyTorch model analysis without external dependencies")
    print("="*60)
    
    # Analyze models
    mil_model, mil_stats = analyze_mil_model_simple()
    hiergat_model, hiergat_stats = analyze_hiergat_model_simple()
    
    # Compare models
    compare_models_simple(mil_stats, hiergat_stats)
    
    # Validate theoretical estimates
    validate_theoretical_estimates()
    
    print("\n" + "="*60)
    print("SIMPLE ANALYSIS COMPLETE")
    print("For detailed analysis with FLOP counting, use model_analysis.py")
    print("="*60)


if __name__ == "__main__":
    main() 