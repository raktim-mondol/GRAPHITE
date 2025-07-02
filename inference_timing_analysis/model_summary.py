#!/usr/bin/env python3
"""
GRAPHITE Model Summary

Quick summary of model parameters and computational requirements.
"""

import sys
import os
import torch
import warnings
warnings.filterwarnings('ignore')

# Add training step directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_1', 'mil_classification', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training_step_2', 'self_supervised_training'))


def count_params(model):
    """Quick parameter count"""
    return sum(p.numel() for p in model.parameters())


def analyze_step1():
    """Analyze Step 1 (MIL) model"""
    try:
        from models.mil_classifier import MILHistopathModel
        model = MILHistopathModel()
        
        total_params = count_params(model)
        
        # Component breakdown
        resnet_params = count_params(model.feature_extractor)
        mil_components = total_params - resnet_params
        
        return {
            'total': total_params,
            'resnet18': resnet_params,
            'mil_components': mil_components,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def analyze_step2():
    """Analyze Step 2 (HierGAT) model"""
    try:
        from models.hiergat import HierGATSSL
        model = HierGATSSL(
            input_dim=128,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            num_levels=3,
            dropout=0.1
        )
        
        total_params = count_params(model)
        
        return {
            'total': total_params,
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_theoretical_estimates():
    """Get theoretical estimates from inference_time_estimator"""
    try:
        from inference_time_estimator import GraphiteInferenceEstimator
        estimator = GraphiteInferenceEstimator()
        
        resnet18 = estimator.calculate_resnet18_params()
        mil_classifier = estimator.calculate_mil_classifier_params()
        hiergat = estimator.calculate_hiergat_params()
        
        return {
            'resnet18_theory': resnet18['total_millions'],
            'mil_classifier_theory': mil_classifier['total_millions'],
            'hiergat_theory': hiergat['total_millions'],
            'success': True
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def main():
    """Main summary function"""
    print("GRAPHITE PIPELINE MODEL SUMMARY")
    print("="*50)
    
    # Analyze models
    step1_results = analyze_step1()
    step2_results = analyze_step2()
    theory_results = get_theoretical_estimates()
    
    print("\nTRAINING STEP 1 (MIL Classification):")
    if step1_results['success']:
        total_m = step1_results['total'] / 1e6
        resnet_m = step1_results['resnet18'] / 1e6
        mil_m = step1_results['mil_components'] / 1e6
        
        print(f"  Total Parameters: {step1_results['total']:,} ({total_m:.2f}M)")
        print(f"  ├─ ResNet18 backbone: {step1_results['resnet18']:,} ({resnet_m:.2f}M)")
        print(f"  └─ MIL components: {step1_results['mil_components']:,} ({mil_m:.2f}M)")
        print(f"  Estimated FLOPs: ~878 GFLOPs")
    else:
        print(f"  ✗ Analysis failed: {step1_results['error']}")
    
    print("\nTRAINING STEP 2 (HierGAT):")
    if step2_results['success']:
        total_m = step2_results['total'] / 1e6
        print(f"  Total Parameters: {step2_results['total']:,} ({total_m:.2f}M)")
        print(f"  Estimated FLOPs: ~0.2 GFLOPs")
    else:
        print(f"  ✗ Analysis failed: {step2_results['error']}")
    
    print("\nPIPELINE SUMMARY:")
    if step1_results['success'] and step2_results['success']:
        total_params = step1_results['total'] + step2_results['total']
        total_m = total_params / 1e6
        ratio = step1_results['total'] / step2_results['total']
        
        print(f"  Combined Parameters: {total_params:,} ({total_m:.2f}M)")
        print(f"  Step 1/Step 2 Ratio: {ratio:.1f}x")
        print(f"  Total Estimated FLOPs: ~878.2 GFLOPs")
        print(f"  Memory (FP32): ~{total_params * 4 / (1024**2):.1f} MB")
    
    print("\nTHEORETICAL VALIDATION:")
    if theory_results['success']:
        step1_theory = theory_results['resnet18_theory'] + theory_results['mil_classifier_theory']
        step2_theory = theory_results['hiergat_theory']
        
        print(f"  Step 1 Theory: {step1_theory:.2f}M parameters")
        print(f"  Step 2 Theory: {step2_theory:.2f}M parameters")
        
        if step1_results['success'] and step2_results['success']:
            step1_actual = step1_results['total'] / 1e6
            step2_actual = step2_results['total'] / 1e6
            
            step1_diff = abs(step1_theory - step1_actual) / step1_actual * 100
            step2_diff = abs(step2_theory - step2_actual) / step2_actual * 100
            
            print(f"  Step 1 Difference: {step1_diff:.1f}%")
            print(f"  Step 2 Difference: {step2_diff:.1f}%")
    else:
        print(f"  ✗ Theory validation failed: {theory_results['error']}")
    
    print("\nARCHITECTURE OVERVIEW:")
    print("  • Step 1: ResNet18 + MIL classifier")
    print("    - Processes 484 patches (22×22 grid)")
    print("    - Attention-based aggregation")
    print("  • Step 2: Hierarchical Graph Attention")
    print("    - 3-level hierarchy with cross-scale attention")
    print("    - Multi-head GAT layers")
    print("  • Pipeline dominance: Step 1 (>95% of parameters)")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    main() 