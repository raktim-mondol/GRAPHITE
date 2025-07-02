#!/usr/bin/env python3
"""
GRAPHITE Model Analysis Tool

Comprehensive analysis of PyTorch models from training_step_1 and training_step_2.
Provides detailed parameter counts, model summaries, and FLOP calculations.

Features:
- PyTorch model summaries using torchinfo
- FLOP calculations using fvcore and ptflops
- Detailed parameter breakdowns
- Memory usage estimates
- Comparison between models
"""

import sys
import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Add training step directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_1', 'mil_classification', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_2', 'self_supervised_training'))

try:
    from torchinfo import summary
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    print("torchinfo not available. Install with: pip install torchinfo")

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("fvcore not available. Install with: pip install fvcore")

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    PTFLOPS_AVAILABLE = False
    print("ptflops not available. Install with: pip install ptflops")


def get_model_parameter_count(model: nn.Module) -> Dict[str, int]:
    """
    Get detailed parameter count for a PyTorch model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    param_details = {}
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        else:
            non_trainable_params += param_count
            
        param_details[name] = {
            'count': param_count,
            'shape': list(param.shape),
            'trainable': param.requires_grad
        }
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'param_details': param_details
    }


def analyze_mil_model():
    """Analyze the MIL model from training_step_1"""
    print("="*60)
    print("ANALYZING MIL MODEL (Training Step 1)")
    print("="*60)
    
    try:
        from models.mil_classifier import MILHistopathModel
        
        # Create model with default parameters
        model = MILHistopathModel(num_classes=2, feat_dim=512, proj_dim=128)
        model.eval()
        
        print(f"Model class: {model.__class__.__name__}")
        print(f"Model architecture: {model}")
        print("\n" + "-"*50)
        
        # Parameter analysis
        param_stats = get_model_parameter_count(model)
        print(f"PARAMETER ANALYSIS:")
        print(f"Total parameters: {param_stats['total_params']:,} ({param_stats['total_params']/1e6:.2f}M)")
        print(f"Trainable parameters: {param_stats['trainable_params']:,}")
        print(f"Non-trainable parameters: {param_stats['non_trainable_params']:,}")
        
        # Component breakdown
        print(f"\nCOMPONENT BREAKDOWN:")
        components = {
            'feature_extractor': 0,
            'patch_projector': 0,
            'attention': 0,
            'patient_projector': 0,
            'classifier': 0,
            'patient_layer_norm': 0
        }
        
        for name, details in param_stats['param_details'].items():
            if 'feature_extractor' in name:
                components['feature_extractor'] += details['count']
            elif 'patch_projector' in name:
                components['patch_projector'] += details['count']
            elif 'attention' in name:
                components['attention'] += details['count']
            elif 'patient_projector' in name:
                components['patient_projector'] += details['count']
            elif 'classifier' in name:
                components['classifier'] += details['count']
            elif 'patient_layer_norm' in name:
                components['patient_layer_norm'] += details['count']
        
        for component, count in components.items():
            percentage = (count / param_stats['total_params']) * 100
            print(f"{component}: {count:,} ({percentage:.1f}%)")
        
        # Torchinfo summary
        if TORCHINFO_AVAILABLE:
            print(f"\nTORCHINFO SUMMARY:")
            try:
                # Create dummy input: (batch_size=1, num_patches=484, channels=3, height=224, width=224)
                dummy_input = torch.randn(1, 484, 3, 224, 224)
                summary_result = summary(model, input_data=dummy_input, verbose=0)
                print(summary_result)
            except Exception as e:
                print(f"Error in torchinfo summary: {e}")
        
        # FLOP analysis with fvcore
        if FVCORE_AVAILABLE:
            print(f"\nFVCORE FLOP ANALYSIS:")
            try:
                dummy_input = torch.randn(1, 484, 3, 224, 224)
                flops = flop_count(model, (dummy_input,), supported_ops=None)
                total_flops = sum(flops[0].values()) if flops[0] else 0
                print(f"Total FLOPs: {total_flops:,} ({total_flops/1e9:.2f} GFLOPs)")
            except Exception as e:
                print(f"Error in fvcore FLOP analysis: {e}")
        
        # FLOP analysis with ptflops
        if PTFLOPS_AVAILABLE:
            print(f"\nPTFLOPS ANALYSIS:")
            try:
                # ptflops expects input shape without batch dimension
                macs, params = get_model_complexity_info(
                    model, 
                    (484, 3, 224, 224),  # (num_patches, channels, height, width)
                    print_per_layer_stat=False,
                    verbose=False
                )
                print(f"MACs: {macs}")
                print(f"Parameters: {params}")
            except Exception as e:
                print(f"Error in ptflops analysis: {e}")
        
        return model, param_stats
        
    except Exception as e:
        print(f"Error analyzing MIL model: {e}")
        return None, None


def analyze_hiergat_model():
    """Analyze the HierGAT model from training_step_2"""
    print("\n" + "="*60)
    print("ANALYZING HIERGAT MODEL (Training Step 2)")
    print("="*60)
    
    try:
        from models.hiergat import HierGATSSL
        
        # Create model with default parameters
        model = HierGATSSL(
            input_dim=128,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            num_levels=3,
            dropout=0.1
        )
        model.eval()
        
        print(f"Model class: {model.__class__.__name__}")
        print(f"Model architecture: {model}")
        print("\n" + "-"*50)
        
        # Parameter analysis
        param_stats = get_model_parameter_count(model)
        print(f"PARAMETER ANALYSIS:")
        print(f"Total parameters: {param_stats['total_params']:,} ({param_stats['total_params']/1e6:.2f}M)")
        print(f"Trainable parameters: {param_stats['trainable_params']:,}")
        print(f"Non-trainable parameters: {param_stats['non_trainable_params']:,}")
        
        # Component breakdown
        print(f"\nCOMPONENT BREAKDOWN:")
        components = {
            'gat_layers': 0,
            'scale_attention': 0,
            'projection_head': 0
        }
        
        for name, details in param_stats['param_details'].items():
            if 'gat_layers' in name:
                components['gat_layers'] += details['count']
            elif 'scale_attention' in name:
                components['scale_attention'] += details['count']
            elif 'projection_head' in name:
                components['projection_head'] += details['count']
        
        for component, count in components.items():
            percentage = (count / param_stats['total_params']) * 100
            print(f"{component}: {count:,} ({percentage:.1f}%)")
        
        # Torchinfo summary
        if TORCHINFO_AVAILABLE:
            print(f"\nTORCHINFO SUMMARY:")
            print("Note: HierGAT uses PyTorch Geometric data format")
            print("Complex to analyze with standard tools due to graph structure")
        
        return model, param_stats
        
    except Exception as e:
        print(f"Error analyzing HierGAT model: {e}")
        return None, None


def compare_models(mil_stats: Optional[Dict], hiergat_stats: Optional[Dict]):
    """Compare the two models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    if mil_stats and hiergat_stats:
        mil_params = mil_stats['total_params']
        hiergat_params = hiergat_stats['total_params']
        total_params = mil_params + hiergat_params
        
        print(f"MIL Model (Step 1): {mil_params:,} parameters ({mil_params/1e6:.2f}M)")
        print(f"HierGAT Model (Step 2): {hiergat_params:,} parameters ({hiergat_params/1e6:.2f}M)")
        print(f"Total Pipeline: {total_params:,} parameters ({total_params/1e6:.2f}M)")
        print(f"Ratio (Step 1 / Step 2): {mil_params/hiergat_params:.1f}x")
        
        print(f"\nMEMORY ESTIMATES (FP32):")
        mil_memory = mil_params * 4 / (1024**2)  # 4 bytes per parameter, convert to MB
        hiergat_memory = hiergat_params * 4 / (1024**2)
        total_memory = total_params * 4 / (1024**2)
        
        print(f"MIL Model: {mil_memory:.1f} MB")
        print(f"HierGAT Model: {hiergat_memory:.1f} MB")
        print(f"Total Pipeline: {total_memory:.1f} MB")
        
    else:
        print("Could not compare models due to analysis errors")


def detailed_layer_analysis():
    """Provide detailed layer-by-layer analysis"""
    print("\n" + "="*60)
    print("DETAILED LAYER ANALYSIS")
    print("="*60)
    
    try:
        from models.mil_classifier import MILHistopathModel
        model = MILHistopathModel()
        
        print("MIL MODEL LAYERS:")
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 0:
                    print(f"{name}: {module} - {param_count:,} parameters")
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")


def theoretical_vs_actual_comparison():
    """Compare theoretical estimates with actual model parameters"""
    print("\n" + "="*60)
    print("THEORETICAL vs ACTUAL COMPARISON")
    print("="*60)
    
    # Load theoretical estimates from inference_time_estimator
    try:
        sys.path.append(os.path.dirname(__file__))
        from inference_time_estimator import GraphiteInferenceEstimator
        
        estimator = GraphiteInferenceEstimator()
        
        # Get theoretical estimates
        resnet18_theory = estimator.calculate_resnet18_params()
        mil_classifier_theory = estimator.calculate_mil_classifier_params()
        hiergat_theory = estimator.calculate_hiergat_params()
        
        print("THEORETICAL ESTIMATES:")
        print(f"ResNet18: {resnet18_theory['total_millions']:.2f}M parameters")
        print(f"MIL Classifier: {mil_classifier_theory['total_millions']:.2f}M parameters")
        print(f"HierGAT: {hiergat_theory['total_millions']:.2f}M parameters")
        
        # Get actual measurements
        try:
            from models.mil_classifier import MILHistopathModel
            mil_model = MILHistopathModel()
            mil_actual = get_model_parameter_count(mil_model)
            
            print(f"\nACTUAL MEASUREMENTS:")
            print(f"MIL Model (total): {mil_actual['total_params']/1e6:.2f}M parameters")
            
            # Calculate differences
            mil_theory_total = resnet18_theory['total_millions'] + mil_classifier_theory['total_millions']
            mil_actual_total = mil_actual['total_params'] / 1e6
            mil_diff = abs(mil_theory_total - mil_actual_total) / mil_actual_total * 100
            
            print(f"\nCOMPARISON:")
            print(f"MIL Model - Theory: {mil_theory_total:.2f}M, Actual: {mil_actual_total:.2f}M")
            print(f"Difference: {mil_diff:.1f}%")
            
        except Exception as e:
            print(f"Error in actual measurements: {e}")
            
    except Exception as e:
        print(f"Error in theoretical comparison: {e}")


def main():
    """Main analysis function"""
    print("GRAPHITE MODEL ANALYSIS TOOL")
    print("="*60)
    print("Analyzing PyTorch models from training steps 1 and 2")
    print("Providing parameter counts, model summaries, and FLOP calculations")
    print("="*60)
    
    # Check available libraries
    print("AVAILABLE ANALYSIS LIBRARIES:")
    print(f"- torchinfo: {'✓' if TORCHINFO_AVAILABLE else '✗'}")
    print(f"- fvcore: {'✓' if FVCORE_AVAILABLE else '✗'}")
    print(f"- ptflops: {'✓' if PTFLOPS_AVAILABLE else '✗'}")
    print()
    
    # Analyze models
    mil_model, mil_stats = analyze_mil_model()
    hiergat_model, hiergat_stats = analyze_hiergat_model()
    
    # Compare models
    compare_models(mil_stats, hiergat_stats)
    
    # Detailed analysis
    detailed_layer_analysis()
    
    # Theoretical comparison
    theoretical_vs_actual_comparison()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main() 