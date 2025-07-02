#!/usr/bin/env python3
"""
HierGAT Model Analysis
Analyze the actual HierGAT model from training_step_2 to get exact parameter counts and FLOP calculations
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import Data

# Add training_step_2 to path
training_step_2_path = Path(__file__).parent.parent / "training_step_2" / "self_supervised_training"
sys.path.insert(0, str(training_step_2_path))

try:
    from models.hiergat import HierGATSSL
    from models.attention import ScaleWiseAttention, HierarchicalGAT
    print("✓ Successfully imported HierGAT models")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed")
    sys.exit(1)

def count_parameters(model, detailed=True):
    """Count parameters in a model with detailed breakdown"""
    total_params = 0
    param_details = {}
    
    if detailed:
        print(f"\n{'='*60}")
        print(f"Parameter Analysis for {model.__class__.__name__}")
        print(f"{'='*60}")
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            if module_params > 0:
                param_details[name] = module_params
                total_params += module_params
                if detailed:
                    print(f"{name:40} {module_params:>12,} parameters")
    
    if detailed:
        print(f"{'='*60}")
        print(f"{'Total Parameters':40} {total_params:>12,}")
        print(f"{'='*60}")
    
    return total_params, param_details

def analyze_gat_conv_params(input_dim, output_dim, heads):
    """Calculate exact parameters for GATConv layer"""
    # GATConv parameters:
    # - Linear transformation: input_dim * output_dim * heads
    # - Attention weights: 2 * output_dim * heads (for source and target concatenation)
    # - Bias terms: output_dim * heads
    
    linear_params = input_dim * output_dim * heads
    attention_params = 2 * output_dim * heads  # a_src and a_dst
    bias_params = output_dim * heads
    
    total = linear_params + attention_params + bias_params
    
    return {
        'linear_transformation': linear_params,
        'attention_weights': attention_params,
        'bias': bias_params,
        'total': total
    }

def analyze_hiergat_architecture():
    """Analyze the HierGAT model architecture in detail"""
    
    print("\n" + "="*80)
    print("HIERGAT MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    
    # Standard configuration from the code
    input_dim = 128
    hidden_dim = 128
    num_gat_layers = 3
    num_heads = 4
    num_levels = 3
    dropout = 0.1
    
    print(f"Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  GAT layers: {num_gat_layers}")
    print(f"  Attention heads: {num_heads}")
    print(f"  Hierarchy levels: {num_levels}")
    print(f"  Dropout: {dropout}")
    
    # Create the model
    model = HierGATSSL(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_gat_layers=num_gat_layers,
        num_heads=num_heads,
        num_levels=num_levels,
        dropout=dropout
    )
    
    # Count total parameters
    total_params, param_details = count_parameters(model, detailed=True)
    
    # Detailed analysis of each component
    print(f"\n{'='*60}")
    print("COMPONENT-WISE PARAMETER BREAKDOWN")
    print(f"{'='*60}")
    
    # 1. GAT Layers Analysis
    print(f"\n1. GAT LAYERS ({num_gat_layers} layers):")
    print(f"   Each layer has spatial + cross-scale GAT convolutions")
    
    head_dim = hidden_dim // num_heads
    print(f"   Head dimension: {head_dim}")
    
    gat_params_total = 0
    for i in range(num_gat_layers):
        layer_input_dim = input_dim if i == 0 else hidden_dim
        
        # Spatial GAT
        spatial_gat = analyze_gat_conv_params(layer_input_dim, head_dim, num_heads)
        # Cross-scale GAT  
        cross_scale_gat = analyze_gat_conv_params(layer_input_dim, head_dim, num_heads)
        
        # LayerNorm and Dropout
        layer_norm_params = 2 * hidden_dim  # weight + bias
        
        layer_total = spatial_gat['total'] + cross_scale_gat['total'] + layer_norm_params
        gat_params_total += layer_total
        
        print(f"   Layer {i+1}:")
        print(f"     Input dim: {layer_input_dim}")
        print(f"     Spatial GAT: {spatial_gat['total']:,} parameters")
        print(f"       - Linear transformation: {spatial_gat['linear_transformation']:,}")
        print(f"       - Attention weights: {spatial_gat['attention_weights']:,}")
        print(f"       - Bias: {spatial_gat['bias']:,}")
        print(f"     Cross-scale GAT: {cross_scale_gat['total']:,} parameters")
        print(f"       - Linear transformation: {cross_scale_gat['linear_transformation']:,}")
        print(f"       - Attention weights: {cross_scale_gat['attention_weights']:,}")
        print(f"       - Bias: {cross_scale_gat['bias']:,}")
        print(f"     LayerNorm: {layer_norm_params:,} parameters")
        print(f"     Layer total: {layer_total:,} parameters")
    
    print(f"   Total GAT parameters: {gat_params_total:,}")
    
    # 2. Scale-wise Attention Analysis
    print(f"\n2. SCALE-WISE ATTENTION:")
    
    # Level-specific attention (3 levels)
    level_attention_params = 0
    for level in range(num_levels):
        # Each level: Linear(hidden_dim, hidden_dim//2) + LayerNorm + Linear(hidden_dim//2, 1)
        linear1_params = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)  # weights + bias
        layer_norm_params = 2 * (hidden_dim // 2)  # weight + bias
        linear2_params = (hidden_dim // 2) * 1 + 1  # weights + bias
        
        level_params = linear1_params + layer_norm_params + linear2_params
        level_attention_params += level_params
        
        print(f"   Level {level} attention: {level_params:,} parameters")
        print(f"     Linear1 ({hidden_dim} -> {hidden_dim//2}): {linear1_params:,}")
        print(f"     LayerNorm: {layer_norm_params:,}")
        print(f"     Linear2 ({hidden_dim//2} -> 1): {linear2_params:,}")
    
    # Cross-scale attention
    cross_scale_linear1 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
    cross_scale_norm = 2 * (hidden_dim // 2)
    cross_scale_linear2 = (hidden_dim // 2) * num_levels + num_levels
    cross_scale_total = cross_scale_linear1 + cross_scale_norm + cross_scale_linear2
    
    print(f"   Cross-scale attention: {cross_scale_total:,} parameters")
    print(f"     Linear1 ({hidden_dim} -> {hidden_dim//2}): {cross_scale_linear1:,}")
    print(f"     LayerNorm: {cross_scale_norm:,}")
    print(f"     Linear2 ({hidden_dim//2} -> {num_levels}): {cross_scale_linear2:,}")
    
    scale_attention_total = level_attention_params + cross_scale_total
    print(f"   Total Scale-wise attention: {scale_attention_total:,} parameters")
    
    # 3. Projection Head Analysis
    print(f"\n3. PROJECTION HEAD:")
    proj_linear1 = hidden_dim * hidden_dim + hidden_dim  # weights + bias
    proj_norm = 2 * hidden_dim  # weight + bias
    proj_linear2 = hidden_dim * hidden_dim + hidden_dim  # weights + bias
    projection_total = proj_linear1 + proj_norm + proj_linear2
    
    print(f"   Linear1 ({hidden_dim} -> {hidden_dim}): {proj_linear1:,} parameters")
    print(f"   LayerNorm: {proj_norm:,} parameters")
    print(f"   Linear2 ({hidden_dim} -> {hidden_dim}): {proj_linear2:,} parameters")
    print(f"   Total projection head: {projection_total:,} parameters")
    
    # Summary
    total_calculated = gat_params_total + scale_attention_total + projection_total
    
    print(f"\n{'='*60}")
    print("PARAMETER SUMMARY")
    print(f"{'='*60}")
    print(f"GAT Layers:           {gat_params_total:>12,} ({gat_params_total/total_params*100:.1f}%)")
    print(f"Scale-wise Attention: {scale_attention_total:>12,} ({scale_attention_total/total_params*100:.1f}%)")
    print(f"Projection Head:      {projection_total:>12,} ({projection_total/total_params*100:.1f}%)")
    print(f"{'='*60}")
    print(f"Total (calculated):   {total_calculated:>12,}")
    print(f"Total (actual):       {total_params:>12,}")
    print(f"Difference:           {abs(total_params - total_calculated):>12,}")
    print(f"{'='*60}")
    
    return model, total_params, {
        'gat_layers': gat_params_total,
        'scale_attention': scale_attention_total,
        'projection_head': projection_total
    }

def estimate_hiergat_flops():
    """Estimate FLOPs for HierGAT model during inference"""
    
    print(f"\n{'='*60}")
    print("HIERGAT FLOPS ESTIMATION")
    print(f"{'='*60}")
    
    # Configuration
    input_dim = 128
    hidden_dim = 128
    num_gat_layers = 3
    num_heads = 4
    num_levels = 3
    
    # Typical graph sizes (based on 484 patches distributed across 3 levels)
    # Level 0: ~220 patches, Level 1: ~160 patches, Level 2: ~104 patches
    level_nodes = [220, 160, 104]
    total_nodes = sum(level_nodes)
    
    # Estimate edges (spatial + cross-scale)
    # Spatial edges: each node connects to ~4 neighbors at same level
    spatial_edges_per_level = [nodes * 4 for nodes in level_nodes]
    total_spatial_edges = sum(spatial_edges_per_level)
    
    # Cross-scale edges: hierarchical connections between levels
    cross_scale_edges = level_nodes[0] * 0.3 + level_nodes[1] * 0.3  # Rough estimate
    total_edges = total_spatial_edges + cross_scale_edges
    
    print(f"Graph structure estimate:")
    print(f"  Level nodes: {level_nodes}")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Spatial edges: {total_spatial_edges:.0f}")
    print(f"  Cross-scale edges: {cross_scale_edges:.0f}")
    print(f"  Total edges: {total_edges:.0f}")
    
    total_flops = 0
    
    # 1. GAT Layer FLOPs
    print(f"\n1. GAT LAYERS FLOPS:")
    
    head_dim = hidden_dim // num_heads
    gat_flops_total = 0
    
    for i in range(num_gat_layers):
        layer_input_dim = input_dim if i == 0 else hidden_dim
        
        # Spatial GAT FLOPs
        # Linear transformation: nodes * input_dim * (head_dim * num_heads) 
        spatial_linear_flops = total_nodes * layer_input_dim * hidden_dim
        # Attention computation: edges * head_dim * num_heads
        spatial_attention_flops = total_spatial_edges * hidden_dim
        spatial_total = spatial_linear_flops + spatial_attention_flops
        
        # Cross-scale GAT FLOPs (similar structure)
        cross_scale_linear_flops = total_nodes * layer_input_dim * hidden_dim
        cross_scale_attention_flops = cross_scale_edges * hidden_dim
        cross_scale_total = cross_scale_linear_flops + cross_scale_attention_flops
        
        # LayerNorm FLOPs
        layer_norm_flops = total_nodes * hidden_dim * 2  # mean and std computation
        
        layer_flops = spatial_total + cross_scale_total + layer_norm_flops
        gat_flops_total += layer_flops
        
        print(f"   Layer {i+1}: {layer_flops:,.0f} FLOPs")
        print(f"     Spatial GAT: {spatial_total:,.0f} FLOPs")
        print(f"     Cross-scale GAT: {cross_scale_total:,.0f} FLOPs")
        print(f"     LayerNorm: {layer_norm_flops:,.0f} FLOPs")
    
    print(f"   Total GAT FLOPs: {gat_flops_total:,.0f}")
    total_flops += gat_flops_total
    
    # 2. Scale-wise Attention FLOPs
    print(f"\n2. SCALE-WISE ATTENTION FLOPS:")
    
    scale_attention_flops = 0
    
    # Level-specific attention for each level
    for level in range(num_levels):
        level_node_count = level_nodes[level]
        # Linear1: nodes * hidden_dim * (hidden_dim // 2)
        linear1_flops = level_node_count * hidden_dim * (hidden_dim // 2)
        # Linear2: nodes * (hidden_dim // 2) * 1
        linear2_flops = level_node_count * (hidden_dim // 2) * 1
        # Softmax and aggregation
        softmax_flops = level_node_count * hidden_dim
        
        level_flops = linear1_flops + linear2_flops + softmax_flops
        scale_attention_flops += level_flops
        
        print(f"   Level {level}: {level_flops:,.0f} FLOPs")
    
    # Cross-scale attention
    # Process 3 level outputs: 3 * hidden_dim * (hidden_dim // 2) + 3 * (hidden_dim // 2) * 3
    cross_scale_flops = num_levels * hidden_dim * (hidden_dim // 2) + num_levels * (hidden_dim // 2) * num_levels
    cross_scale_flops += num_levels * hidden_dim  # Final aggregation
    
    scale_attention_flops += cross_scale_flops
    print(f"   Cross-scale: {cross_scale_flops:,.0f} FLOPs")
    print(f"   Total scale attention: {scale_attention_flops:,.0f} FLOPs")
    total_flops += scale_attention_flops
    
    # 3. Projection Head FLOPs
    print(f"\n3. PROJECTION HEAD FLOPS:")
    
    # Operations on final graph embedding (single vector of size hidden_dim)
    proj_linear1_flops = hidden_dim * hidden_dim
    proj_linear2_flops = hidden_dim * hidden_dim
    projection_flops = proj_linear1_flops + proj_linear2_flops
    
    print(f"   Linear1: {proj_linear1_flops:,.0f} FLOPs")
    print(f"   Linear2: {proj_linear2_flops:,.0f} FLOPs")
    print(f"   Total projection: {projection_flops:,.0f} FLOPs")
    total_flops += projection_flops
    
    # Summary
    print(f"\n{'='*60}")
    print("FLOPS SUMMARY")
    print(f"{'='*60}")
    print(f"GAT Layers:           {gat_flops_total:>15,.0f} FLOPs ({gat_flops_total/total_flops*100:.1f}%)")
    print(f"Scale-wise Attention: {scale_attention_flops:>15,.0f} FLOPs ({scale_attention_flops/total_flops*100:.1f}%)")
    print(f"Projection Head:      {projection_flops:>15,.0f} FLOPs ({projection_flops/total_flops*100:.1f}%)")
    print(f"{'='*60}")
    print(f"Total FLOPs:          {total_flops:>15,.0f} ({total_flops/1e9:.3f} GFLOPs)")
    print(f"{'='*60}")
    
    return total_flops, {
        'gat_layers': gat_flops_total,
        'scale_attention': scale_attention_flops,
        'projection_head': projection_flops
    }

def test_model_forward():
    """Test the model with a sample graph to verify it works"""
    
    print(f"\n{'='*60}")
    print("MODEL FORWARD PASS TEST")
    print(f"{'='*60}")
    
    try:
        # Create model
        model = HierGATSSL(
            input_dim=128,
            hidden_dim=128,
            num_gat_layers=3,
            num_heads=4,
            num_levels=3,
            dropout=0.1
        )
        model.eval()
        
        # Create sample graph data
        num_nodes = 484  # Total patches
        node_features = torch.randn(num_nodes, 128)
        
        # Create sample edges (spatial + cross-scale)
        num_edges = 2000
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        edge_type = torch.randint(0, 2, (num_edges,))  # 0: spatial, 1: cross-scale
        
        # Create level indices
        level_0_indices = torch.arange(0, 220)
        level_1_indices = torch.arange(220, 380) 
        level_2_indices = torch.arange(380, 484)
        
        # Create PyG Data object
        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            pos=torch.randn(num_nodes, 2)
        )
        
        # Add level indices
        data.level_0_indices = level_0_indices
        data.level_1_indices = level_1_indices
        data.level_2_indices = level_2_indices
        
        print(f"Sample graph:")
        print(f"  Nodes: {num_nodes}")
        print(f"  Edges: {num_edges}")
        print(f"  Level 0 nodes: {len(level_0_indices)}")
        print(f"  Level 1 nodes: {len(level_1_indices)}")
        print(f"  Level 2 nodes: {len(level_2_indices)}")
        
        # Forward pass
        with torch.no_grad():
            outputs = model(data)
        
        print(f"\nForward pass successful!")
        print(f"  Node embeddings shape: {outputs['node_embeddings'].shape}")
        print(f"  Graph embedding shape: {outputs['graph_embedding'].shape}")
        print(f"  Projection shape: {outputs['projection'].shape}")
        print(f"  Attention weights keys: {list(outputs['attention_weights'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main analysis function"""
    
    print("HierGAT Model Analysis")
    print("Analyzing the actual HierGAT model from training_step_2")
    
    # Test model loading and forward pass
    if not test_model_forward():
        print("Model testing failed. Cannot proceed with analysis.")
        return
    
    # Analyze architecture and parameters
    model, total_params, param_breakdown = analyze_hiergat_architecture()
    
    # Estimate FLOPs
    total_flops, flop_breakdown = estimate_hiergat_flops()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL HIERGAT ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Total FLOPs: {total_flops:,.0f} ({total_flops/1e9:.3f} GFLOPs)")
    print(f"\nParameter Distribution:")
    for component, params in param_breakdown.items():
        print(f"  {component}: {params:,} ({params/total_params*100:.1f}%)")
    print(f"\nFLOP Distribution:")
    for component, flops in flop_breakdown.items():
        print(f"  {component}: {flops:,.0f} ({flops/total_flops*100:.1f}%)")
    
    # Compare with MIL model
    mil_params = 12_034_882  # From previous analysis
    print(f"\nComparison with MIL model:")
    print(f"  MIL Parameters: {mil_params:,}")
    print(f"  HierGAT Parameters: {total_params:,}")
    print(f"  Ratio: {mil_params/total_params:.1f}x (MIL is {mil_params/total_params:.1f}x larger)")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main() 