#!/usr/bin/env python3
"""
Detailed Parameter and FLOP Calculation Breakdown for GRAPHITE

This script shows every single calculation step by step so you can verify
that the parameter and FLOP calculations are correct based on the actual
architecture.
"""

from inference_time_estimator import create_estimator


def print_resnet18_detailed():
    """Print detailed ResNet18 parameter and FLOP breakdown"""
    print("ResNet18 Detailed Architecture Analysis")
    print("=" * 50)
    
    # ResNet18 parameter calculation (detailed)
    print("\n1. ResNet18 Parameter Calculation:")
    print("-" * 30)
    
    # Conv1: 3→64, kernel 7x7, stride=2, padding=3
    conv1_weights = 3 * 64 * 7 * 7
    conv1_bn = 64 * 2  # BatchNorm γ, β
    conv1_total = conv1_weights + conv1_bn
    print(f"Conv1: 3→64, 7x7 kernel + BN")
    print(f"  Weights: 3×64×7×7 = {conv1_weights:,}")
    print(f"  BatchNorm: 64×2 = {conv1_bn:,}")
    print(f"  Conv1 Total: {conv1_total:,}")
    
    # Layer 1: 2 blocks, 64 channels, no downsample
    print(f"\nLayer1 (2 blocks, 64 channels):")
    block1_conv1 = 64 * 64 * 3 * 3  # 64→64, 3x3
    block1_conv2 = 64 * 64 * 3 * 3  # 64→64, 3x3
    block1_bn = 64 * 4  # 2 BN layers
    block1_total = block1_conv1 + block1_conv2 + block1_bn
    layer1_total = 2 * block1_total  # 2 identical blocks
    print(f"  Per block: (64×64×3×3) + (64×64×3×3) + (64×4) = {block1_total:,}")
    print(f"  Layer1 Total (2 blocks): {layer1_total:,}")
    
    # Layer 2: 2 blocks, 128 channels, first block downsamples
    print(f"\nLayer2 (2 blocks, 128 channels, downsample):")
    # First block with downsample
    block2_1_conv1 = 64 * 128 * 3 * 3  # 64→128, 3x3, stride=2
    block2_1_conv2 = 128 * 128 * 3 * 3  # 128→128, 3x3
    block2_1_shortcut = 64 * 128 * 1 * 1  # shortcut 64→128, 1x1, stride=2
    block2_1_bn = 128 * 6  # 3 BN layers (2 main + 1 shortcut)
    block2_1_total = block2_1_conv1 + block2_1_conv2 + block2_1_shortcut + block2_1_bn
    
    # Second block (no downsample)
    block2_2_conv1 = 128 * 128 * 3 * 3  # 128→128, 3x3
    block2_2_conv2 = 128 * 128 * 3 * 3  # 128→128, 3x3
    block2_2_bn = 128 * 4  # 2 BN layers
    block2_2_total = block2_2_conv1 + block2_2_conv2 + block2_2_bn
    
    layer2_total = block2_1_total + block2_2_total
    print(f"  Block1 (downsample): {block2_1_total:,}")
    print(f"    Conv1: 64×128×3×3 = {block2_1_conv1:,}")
    print(f"    Conv2: 128×128×3×3 = {block2_1_conv2:,}")
    print(f"    Shortcut: 64×128×1×1 = {block2_1_shortcut:,}")
    print(f"    BN: 128×6 = {block2_1_bn:,}")
    print(f"  Block2: {block2_2_total:,}")
    print(f"  Layer2 Total: {layer2_total:,}")
    
    # Layer 3: 2 blocks, 256 channels, first block downsamples
    print(f"\nLayer3 (2 blocks, 256 channels, downsample):")
    block3_1_conv1 = 128 * 256 * 3 * 3
    block3_1_conv2 = 256 * 256 * 3 * 3
    block3_1_shortcut = 128 * 256 * 1 * 1
    block3_1_bn = 256 * 6
    block3_1_total = block3_1_conv1 + block3_1_conv2 + block3_1_shortcut + block3_1_bn
    
    block3_2_conv1 = 256 * 256 * 3 * 3
    block3_2_conv2 = 256 * 256 * 3 * 3
    block3_2_bn = 256 * 4
    block3_2_total = block3_2_conv1 + block3_2_conv2 + block3_2_bn
    
    layer3_total = block3_1_total + block3_2_total
    print(f"  Block1 (downsample): {block3_1_total:,}")
    print(f"  Block2: {block3_2_total:,}")
    print(f"  Layer3 Total: {layer3_total:,}")
    
    # Layer 4: 2 blocks, 512 channels, first block downsamples
    print(f"\nLayer4 (2 blocks, 512 channels, downsample):")
    block4_1_conv1 = 256 * 512 * 3 * 3
    block4_1_conv2 = 512 * 512 * 3 * 3
    block4_1_shortcut = 256 * 512 * 1 * 1
    block4_1_bn = 512 * 6
    block4_1_total = block4_1_conv1 + block4_1_conv2 + block4_1_shortcut + block4_1_bn
    
    block4_2_conv1 = 512 * 512 * 3 * 3
    block4_2_conv2 = 512 * 512 * 3 * 3
    block4_2_bn = 512 * 4
    block4_2_total = block4_2_conv1 + block4_2_conv2 + block4_2_bn
    
    layer4_total = block4_1_total + block4_2_total
    print(f"  Block1 (downsample): {block4_1_total:,}")
    print(f"  Block2: {block4_2_total:,}")
    print(f"  Layer4 Total: {layer4_total:,}")
    
    # Total ResNet18 parameters
    resnet18_total = conv1_total + layer1_total + layer2_total + layer3_total + layer4_total
    print(f"\nResNet18 Total Parameters: {resnet18_total:,} ({resnet18_total/1e6:.1f}M)")
    
    # ResNet18 FLOP calculation (per 224x224 patch)
    print(f"\n2. ResNet18 FLOP Calculation (per 224×224 patch):")
    print("-" * 30)
    
    # Conv1: 3→64, 7x7, stride=2 → output 64×112×112
    conv1_flops = 3 * 64 * 7 * 7 * 112 * 112
    print(f"Conv1: 3×64×7×7×112×112 = {conv1_flops:,} FLOPs")
    
    # Layer1: 64×56×56 (after maxpool)
    layer1_flops = 2 * (64*64*3*3*56*56 + 64*64*3*3*56*56)
    print(f"Layer1: 2×(64×64×3×3×56×56 + 64×64×3×3×56×56) = {layer1_flops:,} FLOPs")
    
    # Layer2: 128×28×28
    layer2_block1_flops = 64*128*3*3*28*28 + 128*128*3*3*28*28 + 64*128*1*1*28*28
    layer2_block2_flops = 128*128*3*3*28*28 + 128*128*3*3*28*28
    layer2_flops = layer2_block1_flops + layer2_block2_flops
    print(f"Layer2: {layer2_flops:,} FLOPs")
    
    # Layer3: 256×14×14
    layer3_block1_flops = 128*256*3*3*14*14 + 256*256*3*3*14*14 + 128*256*1*1*14*14
    layer3_block2_flops = 256*256*3*3*14*14 + 256*256*3*3*14*14
    layer3_flops = layer3_block1_flops + layer3_block2_flops
    print(f"Layer3: {layer3_flops:,} FLOPs")
    
    # Layer4: 512×7×7
    layer4_block1_flops = 256*512*3*3*7*7 + 512*512*3*3*7*7 + 256*512*1*1*7*7
    layer4_block2_flops = 512*512*3*3*7*7 + 512*512*3*3*7*7
    layer4_flops = layer4_block1_flops + layer4_block2_flops
    print(f"Layer4: {layer4_flops:,} FLOPs")
    
    # Total per patch
    resnet18_flops_per_patch = conv1_flops + layer1_flops + layer2_flops + layer3_flops + layer4_flops
    resnet18_flops_all_patches = resnet18_flops_per_patch * 484  # 484 patches
    print(f"\nResNet18 per patch: {resnet18_flops_per_patch:,} FLOPs")
    print(f"ResNet18 all patches (484): {resnet18_flops_all_patches:,} FLOPs ({resnet18_flops_all_patches/1e9:.1f} GFLOPs)")
    
    return resnet18_total, resnet18_flops_all_patches


def print_mil_classifier_detailed():
    """Print detailed MIL classifier parameter and FLOP breakdown"""
    print("\n\nMIL Classifier Detailed Architecture Analysis")
    print("=" * 50)
    
    feat_dim = 512
    proj_dim = 128
    num_patches = 484
    
    print(f"\nArchitecture: feat_dim={feat_dim}, proj_dim={proj_dim}, patches={num_patches}")
    
    print(f"\n1. MIL Classifier Parameter Calculation:")
    print("-" * 30)
    
    # 1. Patch projector: 512→512, LN, ReLU, 512→128, LN
    patch_proj_linear1 = feat_dim * feat_dim + feat_dim  # Linear + bias
    patch_proj_ln1 = feat_dim * 2  # LayerNorm γ, β
    patch_proj_linear2 = feat_dim * proj_dim + proj_dim  # Linear + bias
    patch_proj_ln2 = proj_dim * 2  # LayerNorm γ, β
    patch_projector_total = patch_proj_linear1 + patch_proj_ln1 + patch_proj_linear2 + patch_proj_ln2
    
    print(f"Patch Projector:")
    print(f"  Linear1: 512×512 + 512 = {patch_proj_linear1:,}")
    print(f"  LayerNorm1: 512×2 = {patch_proj_ln1:,}")
    print(f"  Linear2: 512×128 + 128 = {patch_proj_linear2:,}")
    print(f"  LayerNorm2: 128×2 = {patch_proj_ln2:,}")
    print(f"  Total: {patch_projector_total:,}")
    
    # 2. Attention: 512→128, LN, Tanh, 128→1
    attn_linear1 = feat_dim * 128 + 128
    attn_ln = 128 * 2
    attn_linear2 = 128 * 1 + 1
    attention_total = attn_linear1 + attn_ln + attn_linear2
    
    print(f"\nAttention Mechanism:")
    print(f"  Linear1: 512×128 + 128 = {attn_linear1:,}")
    print(f"  LayerNorm: 128×2 = {attn_ln:,}")
    print(f"  Linear2: 128×1 + 1 = {attn_linear2:,}")
    print(f"  Total: {attention_total:,}")
    
    # 3. Patient projector (same as patch projector)
    patient_projector_total = patch_projector_total
    print(f"\nPatient Projector: {patient_projector_total:,} (same as patch projector)")
    
    # 4. Classifier: 512→256, LN, ReLU, Dropout, 256→1
    classifier_linear1 = feat_dim * (feat_dim // 2) + (feat_dim // 2)  # 512×256 + bias
    classifier_ln = (feat_dim // 2) * 2  # LayerNorm γ, β
    classifier_linear2 = (feat_dim // 2) * 1 + 1  # 256×1 + bias
    classifier_total = classifier_linear1 + classifier_ln + classifier_linear2
    
    print(f"\nClassifier:")
    print(f"  Linear1: 512×256 + 256 = {classifier_linear1:,}")
    print(f"  LayerNorm: 256×2 = {classifier_ln:,}")
    print(f"  Linear2: 256×1 + 1 = {classifier_linear2:,}")
    print(f"  Total: {classifier_total:,}")
    
    # 5. Patient LayerNorm
    patient_ln = feat_dim * 2
    print(f"\nPatient LayerNorm: 512×2 = {patient_ln:,}")
    
    # Total MIL classifier parameters
    mil_total = patch_projector_total + attention_total + patient_projector_total + classifier_total + patient_ln
    print(f"\nMIL Classifier Total: {mil_total:,} ({mil_total/1e6:.1f}M)")
    
    print(f"\n2. MIL Classifier FLOP Calculation:")
    print("-" * 30)
    
    # 1. Patch projector (applied to all patches)
    patch_proj_flops1 = num_patches * feat_dim * feat_dim
    patch_proj_flops2 = num_patches * feat_dim * proj_dim
    patch_projector_flops = patch_proj_flops1 + patch_proj_flops2
    
    print(f"Patch Projector (applied to {num_patches} patches):")
    print(f"  Linear1: {num_patches}×512×512 = {patch_proj_flops1:,}")
    print(f"  Linear2: {num_patches}×512×128 = {patch_proj_flops2:,}")
    print(f"  Total: {patch_projector_flops:,}")
    
    # 2. Attention (applied to all patches)
    attn_flops1 = num_patches * feat_dim * 128
    attn_flops2 = num_patches * 128 * 1
    attn_softmax = num_patches  # Softmax computation
    attention_flops = attn_flops1 + attn_flops2 + attn_softmax
    
    print(f"\nAttention (applied to {num_patches} patches):")
    print(f"  Linear1: {num_patches}×512×128 = {attn_flops1:,}")
    print(f"  Linear2: {num_patches}×128×1 = {attn_flops2:,}")
    print(f"  Softmax: {attn_softmax:,}")
    print(f"  Total: {attention_flops:,}")
    
    # 3. Weighted aggregation
    weighted_agg = num_patches * feat_dim
    print(f"\nWeighted Aggregation: {num_patches}×512 = {weighted_agg:,}")
    
    # 4. Patient projector (applied once)
    patient_proj_flops1 = feat_dim * feat_dim
    patient_proj_flops2 = feat_dim * proj_dim
    patient_projector_flops = patient_proj_flops1 + patient_proj_flops2
    
    print(f"\nPatient Projector (applied once):")
    print(f"  Linear1: 512×512 = {patient_proj_flops1:,}")
    print(f"  Linear2: 512×128 = {patient_proj_flops2:,}")
    print(f"  Total: {patient_projector_flops:,}")
    
    # 5. Classifier (applied once)
    classifier_flops1 = feat_dim * (feat_dim // 2)
    classifier_flops2 = (feat_dim // 2) * 1
    classifier_flops = classifier_flops1 + classifier_flops2
    
    print(f"\nClassifier (applied once):")
    print(f"  Linear1: 512×256 = {classifier_flops1:,}")
    print(f"  Linear2: 256×1 = {classifier_flops2:,}")
    print(f"  Total: {classifier_flops:,}")
    
    # Total MIL classifier FLOPs
    mil_flops_total = patch_projector_flops + attention_flops + weighted_agg + patient_projector_flops + classifier_flops
    print(f"\nMIL Classifier Total FLOPs: {mil_flops_total:,} ({mil_flops_total/1e9:.1f} GFLOPs)")
    
    return mil_total, mil_flops_total


def print_hiergat_detailed():
    """Print detailed HierGAT parameter and FLOP breakdown"""
    print("\n\nHierGAT Detailed Architecture Analysis")
    print("=" * 50)
    
    input_dim = 128
    hidden_dim = 128
    num_heads = 4
    num_gat_layers = 3
    num_levels = 3
    head_dim = hidden_dim // num_heads  # 32
    num_nodes = 484
    num_edges = num_nodes * 4  # Approximate edges
    
    print(f"\nArchitecture: input_dim={input_dim}, hidden_dim={hidden_dim}, heads={num_heads}, layers={num_gat_layers}")
    
    print(f"\n1. HierGAT Parameter Calculation:")
    print("-" * 30)
    
    # HierarchicalGAT layers
    print(f"HierarchicalGAT Layers (3 layers):")
    
    # Layer 0: input_dim → hidden_dim
    # Each layer has spatial and cross-scale GATConv
    # GATConv parameters: (input * head_dim + head_dim) * num_heads + input * num_heads
    gat0_spatial = (input_dim * head_dim + head_dim) * num_heads + (input_dim * num_heads)
    gat0_cross = (input_dim * head_dim + head_dim) * num_heads + (input_dim * num_heads)
    gat0_ln = hidden_dim * 2  # LayerNorm
    gat0_total = gat0_spatial + gat0_cross + gat0_ln
    
    print(f"  Layer 0 (128→128):")
    print(f"    Spatial GATConv: ({input_dim}×{head_dim}+{head_dim})×{num_heads} + {input_dim}×{num_heads} = {gat0_spatial:,}")
    print(f"    Cross-scale GATConv: {gat0_cross:,}")
    print(f"    LayerNorm: {hidden_dim}×2 = {gat0_ln:,}")
    print(f"    Layer 0 Total: {gat0_total:,}")
    
    # Layers 1-2: hidden_dim → hidden_dim
    gat_hidden_spatial = (hidden_dim * head_dim + head_dim) * num_heads + (hidden_dim * num_heads)
    gat_hidden_cross = (hidden_dim * head_dim + head_dim) * num_heads + (hidden_dim * num_heads)
    gat_hidden_ln = hidden_dim * 2
    gat_hidden_total = gat_hidden_spatial + gat_hidden_cross + gat_hidden_ln
    
    print(f"  Layers 1-2 (128→128 each):")
    print(f"    Per layer: {gat_hidden_total:,}")
    print(f"    Layers 1-2 Total: {gat_hidden_total * 2:,}")
    
    total_gat_layers = gat0_total + (gat_hidden_total * 2)
    print(f"  All GAT Layers Total: {total_gat_layers:,}")
    
    # ScaleWiseAttention
    print(f"\nScaleWiseAttention:")
    # Level-specific attention: num_levels × (128→64, LN, ReLU, 64→1)
    level_attn_linear1 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)  # 128×64 + bias
    level_attn_ln = (hidden_dim // 2) * 2  # LayerNorm
    level_attn_linear2 = (hidden_dim // 2) * 1 + 1  # 64×1 + bias
    level_attn_per_level = level_attn_linear1 + level_attn_ln + level_attn_linear2
    level_attention_total = num_levels * level_attn_per_level
    
    print(f"  Level-specific attention (3 levels):")
    print(f"    Per level: 128×64+64 + 64×2 + 64×1+1 = {level_attn_per_level:,}")
    print(f"    All levels: {level_attention_total:,}")
    
    # Cross-scale attention: 128→64, LN, ReLU, 64→3
    cross_scale_linear1 = hidden_dim * (hidden_dim // 2) + (hidden_dim // 2)
    cross_scale_ln = (hidden_dim // 2) * 2
    cross_scale_linear2 = (hidden_dim // 2) * num_levels + num_levels
    cross_scale_total = cross_scale_linear1 + cross_scale_ln + cross_scale_linear2
    
    print(f"  Cross-scale attention:")
    print(f"    128×64+64 + 64×2 + 64×3+3 = {cross_scale_total:,}")
    
    scale_attention_total = level_attention_total + cross_scale_total
    print(f"  ScaleWiseAttention Total: {scale_attention_total:,}")
    
    # Projection head: 128→128, LN, ReLU, 128→128
    proj_linear1 = hidden_dim * hidden_dim + hidden_dim
    proj_ln = hidden_dim * 2
    proj_linear2 = hidden_dim * hidden_dim + hidden_dim
    projection_head_total = proj_linear1 + proj_ln + proj_linear2
    
    print(f"\nProjection Head:")
    print(f"  Linear1: 128×128+128 = {proj_linear1:,}")
    print(f"  LayerNorm: 128×2 = {proj_ln:,}")
    print(f"  Linear2: 128×128+128 = {proj_linear2:,}")
    print(f"  Total: {projection_head_total:,}")
    
    # Total HierGAT parameters
    hiergat_total = total_gat_layers + scale_attention_total + projection_head_total
    print(f"\nHierGAT Total Parameters: {hiergat_total:,} ({hiergat_total/1e6:.1f}M)")
    
    print(f"\n2. HierGAT FLOP Calculation:")
    print("-" * 30)
    print(f"Graph: {num_nodes} nodes, ~{num_edges} edges")
    
    # GAT layers FLOPs
    # Message passing + attention computation per edge
    gat0_message = num_edges * input_dim * head_dim * num_heads
    gat0_attention = num_edges * input_dim
    gat0_flops = (gat0_message + gat0_attention) * 2  # spatial + cross-scale
    
    gat_hidden_message = num_edges * hidden_dim * head_dim * num_heads
    gat_hidden_attention = num_edges * hidden_dim
    gat_hidden_flops = (gat_hidden_message + gat_hidden_attention) * 2
    
    total_gat_flops = gat0_flops + (gat_hidden_flops * 2)
    
    print(f"GAT Layers:")
    print(f"  Layer 0: {gat0_flops:,} FLOPs")
    print(f"  Layers 1-2: {gat_hidden_flops * 2:,} FLOPs")
    print(f"  Total: {total_gat_flops:,} FLOPs")
    
    # Scale-wise attention FLOPs
    nodes_per_level = num_nodes // 3  # ~161 nodes per level
    level_attn_flops = 3 * (nodes_per_level * hidden_dim * (hidden_dim//2) + 
                           nodes_per_level * (hidden_dim//2) * 1)
    
    cross_scale_flops = 3 * hidden_dim * (hidden_dim//2) + 3 * (hidden_dim//2) * 3
    
    scale_attention_flops = level_attn_flops + cross_scale_flops
    
    print(f"\nScale-wise Attention:")
    print(f"  Level attention: {level_attn_flops:,} FLOPs")
    print(f"  Cross-scale: {cross_scale_flops:,} FLOPs")
    print(f"  Total: {scale_attention_flops:,} FLOPs")
    
    # Projection head FLOPs
    projection_flops = hidden_dim * hidden_dim + hidden_dim * hidden_dim
    print(f"\nProjection Head: {projection_flops:,} FLOPs")
    
    # Total HierGAT FLOPs
    hiergat_flops_total = total_gat_flops + scale_attention_flops + projection_flops
    print(f"\nHierGAT Total FLOPs: {hiergat_flops_total:,} ({hiergat_flops_total/1e9:.1f} GFLOPs)")
    
    return hiergat_total, hiergat_flops_total


def print_pipeline_comparison():
    """Print complete pipeline comparison"""
    print("\n\n" + "="*60)
    print("COMPLETE PIPELINE ANALYSIS")
    print("="*60)
    
    # Get detailed calculations
    resnet18_params, resnet18_flops = print_resnet18_detailed()
    mil_params, mil_flops = print_mil_classifier_detailed()
    hiergat_params, hiergat_flops = print_hiergat_detailed()
    
    print(f"\n\nPIPELINE SUMMARY")
    print("="*30)
    
    # Pipeline 1: training_step_1
    p1_params = resnet18_params + mil_params
    p1_base_flops = resnet18_flops + mil_flops
    
    print(f"\nPipeline 1 (training_step_1):")
    print(f"  ResNet18: {resnet18_params:,} params, {resnet18_flops/1e9:.1f} GFLOPs")
    print(f"  MIL Classifier: {mil_params:,} params, {mil_flops/1e9:.1f} GFLOPs")
    print(f"  Total: {p1_params:,} params ({p1_params/1e6:.1f}M), {p1_base_flops/1e9:.1f} GFLOPs")
    
    # Pipeline 1 with CAM methods
    gradcam_factor = 1.2
    fullgrad_factor = 2.5
    
    print(f"\nPipeline 1 with CAM methods:")
    print(f"  GradCAM (1.2x): {(p1_base_flops * gradcam_factor)/1e9:.1f} GFLOPs")
    print(f"  FullGrad (2.5x): {(p1_base_flops * fullgrad_factor)/1e9:.1f} GFLOPs")
    
    # Pipeline 2: GRAPHITE
    p2_params = resnet18_params + mil_params + hiergat_params
    p2_base_flops = resnet18_flops + mil_flops + hiergat_flops
    p2_fullgrad_additional = p1_base_flops * (fullgrad_factor - 1.0)  # Additional FullGrad
    fusion_flops = 484 * 0.2e6  # Fusion processing
    p2_total_flops = p2_base_flops + p2_fullgrad_additional + fusion_flops
    
    print(f"\nPipeline 2 (GRAPHITE):")
    print(f"  ResNet18: {resnet18_params:,} params, {resnet18_flops/1e9:.1f} GFLOPs")
    print(f"  MIL Classifier: {mil_params:,} params, {mil_flops/1e9:.1f} GFLOPs")
    print(f"  HierGAT: {hiergat_params:,} params, {hiergat_flops/1e9:.1f} GFLOPs")
    print(f"  FullGrad Additional: {p2_fullgrad_additional/1e9:.1f} GFLOPs")
    print(f"  Fusion: {fusion_flops/1e9:.1f} GFLOPs")
    print(f"  Total: {p2_params:,} params ({p2_params/1e6:.1f}M), {p2_total_flops/1e9:.1f} GFLOPs")
    
    print(f"\nComparison (GRAPHITE vs Pipeline 1 FullGrad):")
    p1_fullgrad_flops = p1_base_flops * fullgrad_factor
    print(f"  Parameter ratio: {p2_params/p1_params:.1f}x")
    print(f"  FLOP ratio: {p2_total_flops/p1_fullgrad_flops:.1f}x")
    print(f"  Complexity justified by:")
    print(f"    - Additional HierGAT model ({hiergat_params/1e3:.0f}K params, {hiergat_flops/1e9:.1f} GFLOPs)")
    print(f"    - Separate FullGrad computation ({p2_fullgrad_additional/1e9:.1f} GFLOPs)")
    print(f"    - Multi-level fusion processing ({fusion_flops/1e9:.1f} GFLOPs)")


if __name__ == "__main__":
    print_pipeline_comparison() 