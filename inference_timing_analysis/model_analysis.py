#!/usr/bin/env python3
"""
GRAPHITE Model Analysis Tool

Comprehensive analysis of PyTorch models from training_step_1 and training_step_2.
Provides detailed parameter counts, model summaries, and FLOP calculations.

Features:
- PyTorch model summaries using torchinfo
- FLOP calculations using fvcore and ptflops (with memory safety)
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
import gc
import psutil
warnings.filterwarnings('ignore')

# Add training step directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_1', 'mil_classification', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_2', 'self_supervised_training'))

# Enhanced library detection with error handling
def check_library_availability():
    """Check availability of analysis libraries with detailed error reporting"""
    libraries = {}
    
    # Check torchinfo
    try:
        from torchinfo import summary
        libraries['torchinfo'] = True
    except ImportError as e:
        libraries['torchinfo'] = False
        print(f"torchinfo not available: {e}")
        print("Install with: pip install torchinfo")
    
    # Check fvcore with enhanced detection
    try:
        import fvcore
        from fvcore.nn import flop_count
        libraries['fvcore'] = True
        print(f"fvcore version: {fvcore.__version__} found at: {fvcore.__file__}")
    except ImportError as e:
        libraries['fvcore'] = False
        print(f"fvcore not available: {e}")
        print("Install with: pip install fvcore")
    except Exception as e:
        libraries['fvcore'] = False
        print(f"fvcore import error: {e}")
    
    # Check ptflops with safer approach
    try:
        import ptflops
        from ptflops import get_model_complexity_info
        libraries['ptflops'] = True
        print(f"ptflops found at: {ptflops.__file__}")
    except ImportError as e:
        libraries['ptflops'] = False
        print(f"ptflops not available: {e}")
        print("Install with: pip install ptflops")
    
    return libraries

# Initialize library availability
LIBS = check_library_availability()


def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def safe_ptflops_analysis(model, input_shape, timeout_seconds=30):
    """Safely run ptflops analysis with memory monitoring and timeout protection"""
    if not LIBS['ptflops']:
        return None
        
    try:
        from ptflops import get_model_complexity_info
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("ptflops analysis timed out")
        
        # Monitor initial memory
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Set timeout (only works on Unix-like systems)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
        
        try:
            # Use smaller input for memory efficiency
            small_input_shape = (1, 3, 224, 224) if len(input_shape) > 3 else input_shape
            print(f"Using reduced input shape for ptflops: {small_input_shape}")
            
            macs, params = get_model_complexity_info(
                model, 
                small_input_shape,
                print_per_layer_stat=False,
                verbose=False,
                as_strings=True
            )
            
            # Cancel alarm if set
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            return {'macs': macs, 'params': params}
            
        except (TimeoutError, RuntimeError, MemoryError) as e:
            print(f"ptflops analysis failed: {e}")
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            return None
            
    except Exception as e:
        print(f"Error in safe ptflops analysis: {e}")
        return None
    finally:
        # Clean up memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


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
        
        # Torchinfo summary (memory safe)
        if LIBS['torchinfo']:
            print(f"\nTORCHINFO SUMMARY:")
            try:
                # Use smaller input for memory efficiency
                dummy_input = torch.randn(1, 16, 3, 224, 224)  # Reduced from 484 patches
                print(f"Using reduced input size (16 patches instead of 484) for analysis")
                
                from torchinfo import summary
                summary_result = summary(model, input_data=dummy_input, verbose=0)
                print(summary_result)
            except Exception as e:
                print(f"Error in torchinfo summary: {e}")
        
        # Enhanced FLOP analysis with fvcore
        if LIBS['fvcore']:
            print(f"\nFVCORE FLOP ANALYSIS:")
            try:
                from fvcore.nn import flop_count
                
                # Use smaller input to avoid memory issues
                dummy_input = torch.randn(1, 16, 3, 224, 224)
                print(f"Using reduced input (16 patches) for FLOP calculation")
                
                flops = flop_count(model, (dummy_input,), supported_ops=None)
                if flops and len(flops) > 0 and flops[0]:
                    total_flops = sum(flops[0].values())
                    # Scale up to full 484 patches
                    scaled_flops = total_flops * (484 / 16)
                    print(f"FLOPs for 16 patches: {total_flops:,} ({total_flops/1e9:.2f} GFLOPs)")
                    print(f"Estimated FLOPs for 484 patches: {scaled_flops:,} ({scaled_flops/1e9:.2f} GFLOPs)")
                else:
                    print("Could not calculate FLOPs with fvcore")
            except Exception as e:
                print(f"Error in fvcore FLOP analysis: {e}")
        
        # Safe FLOP analysis with ptflops
        if LIBS['ptflops']:
            print(f"\nPTFLOPS ANALYSIS (MEMORY SAFE):")
            ptflops_result = safe_ptflops_analysis(
                model, 
                (16, 3, 224, 224),  # Reduced input size
                timeout_seconds=30
            )
            if ptflops_result:
                print(f"MACs (16 patches): {ptflops_result['macs']}")
                print(f"Parameters: {ptflops_result['params']}")
                print(f"Note: Results scaled for 16 patches instead of 484 to avoid memory issues")
            else:
                print("ptflops analysis skipped due to memory/timeout concerns")
        
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
        if LIBS['torchinfo']:
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
    print(f"- torchinfo: {'✓' if LIBS['torchinfo'] else '✗'}")
    print(f"- fvcore: {'✓' if LIBS['fvcore'] else '✗'}")
    print(f"- ptflops: {'✓' if LIBS['ptflops'] else '✗'}")
    print(f"- psutil: ✓ (for memory monitoring)")
    print()
    
    print(f"Initial system memory usage: {get_memory_usage():.1f} MB")
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
    
    print(f"\nFinal system memory usage: {get_memory_usage():.1f} MB")
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("\nNOTES:")
    print("- ptflops analysis uses reduced input sizes to prevent memory issues")
    print("- Results are scaled/estimated for full 484-patch inputs where applicable")
    print("- Memory monitoring helps track resource usage during analysis")


if __name__ == "__main__":
    main() 