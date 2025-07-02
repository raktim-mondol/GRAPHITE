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
    
    def calculate_resnet18_params(self) -> Dict[str, float]:
        """
        Calculate ResNet18 parameters in detail
        
        Returns:
            Dictionary with detailed parameter breakdown
        """
        # ResNet18 architecture breakdown
        # Conv1: 3x64, 7x7 kernel + BatchNorm
        conv1_params = (3 * 64 * 7 * 7) + (64 * 2)  # weights + BN (γ, β)
        
        # Layer 1: 2 blocks, 64 channels
        # Block 1: 64->64, 3x3 + 64->64, 3x3 + BN*2
        # Block 2: 64->64, 3x3 + 64->64, 3x3 + BN*2
        layer1_params = 2 * ((64*64*3*3) + (64*64*3*3) + (64*4))  # 2 blocks
        
        # Layer 2: 2 blocks, 128 channels
        # Block 1: 64->128, 3x3 + 128->128, 3x3 + shortcut 64->128, 1x1 + BN*3
        # Block 2: 128->128, 3x3 + 128->128, 3x3 + BN*2
        layer2_params = ((64*128*3*3) + (128*128*3*3) + (64*128*1*1) + (128*6)) + \
                       ((128*128*3*3) + (128*128*3*3) + (128*4))
        
        # Layer 3: 2 blocks, 256 channels
        # Block 1: 128->256, 3x3 + 256->256, 3x3 + shortcut 128->256, 1x1 + BN*3
        # Block 2: 256->256, 3x3 + 256->256, 3x3 + BN*2
        layer3_params = ((128*256*3*3) + (256*256*3*3) + (128*256*1*1) + (256*6)) + \
                       ((256*256*3*3) + (256*256*3*3) + (256*4))
        
        # Layer 4: 2 blocks, 512 channels
        # Block 1: 256->512, 3x3 + 512->512, 3x3 + shortcut 256->512, 1x1 + BN*3
        # Block 2: 512->512, 3x3 + 512->512, 3x3 + BN*2
        layer4_params = ((256*512*3*3) + (512*512*3*3) + (256*512*1*1) + (512*6)) + \
                       ((512*512*3*3) + (512*512*3*3) + (512*4))
        
        # Average pooling (no parameters) + FC layer (removed in our case)
        
        total_params = conv1_params + layer1_params + layer2_params + layer3_params + layer4_params
        
        return {
            'conv1': conv1_params,
            'layer1': layer1_params,
            'layer2': layer2_params,
            'layer3': layer3_params,
            'layer4': layer4_params,
            'total': total_params,
            'total_millions': total_params / 1e6
        }
    
    def calculate_mil_classifier_params(self) -> Dict[str, float]:
        """
        Calculate MIL classifier parameters (everything except ResNet18)
        
        Architecture:
        1. Patch projector: 512->512, LN, ReLU, 512->128, LN
        2. Attention: 512->128, LN, Tanh, 128->1
        3. Patient projector: 512->512, LN, ReLU, 512->128, LN
        4. Classifier: 512->256, LN, ReLU, Dropout, 256->1
        5. Patient LayerNorm: 512
        
        Returns:
            Dictionary with detailed parameter breakdown
        """
        feat_dim = 512
        proj_dim = 128
        
        # 1. Patch projector
        patch_proj_linear1 = feat_dim * feat_dim + feat_dim  # 512*512 + bias
        patch_proj_ln1 = feat_dim * 2  # LayerNorm (γ, β)
        patch_proj_linear2 = feat_dim * proj_dim + proj_dim  # 512*128 + bias  
        patch_proj_ln2 = proj_dim * 2  # LayerNorm (γ, β)
        patch_projector_params = patch_proj_linear1 + patch_proj_ln1 + patch_proj_linear2 + patch_proj_ln2
        
        # 2. Attention mechanism
        attn_linear1 = feat_dim * 128 + 128  # 512*128 + bias
        attn_ln = 128 * 2  # LayerNorm (γ, β)
        attn_linear2 = 128 * 1 + 1  # 128*1 + bias
        attention_params = attn_linear1 + attn_ln + attn_linear2
        
        # 3. Patient projector (same as patch projector)
        patient_projector_params = patch_projector_params  # Same architecture
        
        # 4. Classifier
        classifier_linear1 = feat_dim * (feat_dim // 2) + (feat_dim // 2)  # 512*256 + bias
        classifier_ln = (feat_dim // 2) * 2  # LayerNorm (γ, β)
        classifier_linear2 = (feat_dim // 2) * 1 + 1  # 256*1 + bias
        classifier_params = classifier_linear1 + classifier_ln + classifier_linear2
        
        # 5. Patient LayerNorm
        patient_ln_params = feat_dim * 2  # LayerNorm (γ, β)
        
        total_params = (patch_projector_params + attention_params + 
                       patient_projector_params + classifier_params + patient_ln_params)
        
        return {
            'patch_projector': patch_projector_params,
            'attention': attention_params,
            'patient_projector': patient_projector_params,
            'classifier': classifier_params,
            'patient_layernorm': patient_ln_params,
            'total': total_params,
            'total_millions': total_params / 1e6
        }
    
    def calculate_hiergat_params(self) -> Dict[str, float]:
        """
        Calculate HierGAT model parameters
        
        Architecture:
        - 3 HierarchicalGAT layers (input_dim=128, hidden_dim=128, num_heads=4)
        - ScaleWiseAttention (hidden_dim=128, num_levels=3)
        - Projection head: 128->128, LN, ReLU, 128->128
        
        Returns:
            Dictionary with detailed parameter breakdown
        """
        input_dim = 128
        hidden_dim = 128
        num_heads = 4
        num_gat_layers = 3
        num_levels = 3
        head_dim = hidden_dim // num_heads  # 32
        
        # 1. HierarchicalGAT layers
        # Each layer has 2 GATConv: spatial and cross-scale
        # GATConv params: (input_dim * head_dim + head_dim) * num_heads for weights + attention weights
        
        # Layer 0: input_dim -> hidden_dim
        gat_layer0_spatial = (input_dim * head_dim + head_dim) * num_heads + (input_dim * num_heads)  # GATConv spatial
        gat_layer0_cross = (input_dim * head_dim + head_dim) * num_heads + (input_dim * num_heads)  # GATConv cross-scale
        gat_layer0_ln = hidden_dim * 2  # LayerNorm (γ, β)
        gat_layer0_total = gat_layer0_spatial + gat_layer0_cross + gat_layer0_ln
        
        # Layers 1-2: hidden_dim -> hidden_dim
        gat_layer_hidden_spatial = (hidden_dim * head_dim + head_dim) * num_heads + (hidden_dim * num_heads)
        gat_layer_hidden_cross = (hidden_dim * head_dim + head_dim) * num_heads + (hidden_dim * num_heads)
        gat_layer_hidden_ln = hidden_dim * 2
        gat_layer_hidden_total = gat_layer_hidden_spatial + gat_layer_hidden_cross + gat_layer_hidden_ln
        
        total_gat_layers = gat_layer0_total + (num_gat_layers - 1) * gat_layer_hidden_total
        
        # 2. ScaleWiseAttention
        # Level-specific attention: num_levels * (128->64, LN, ReLU, 64->1)
        level_attn_per_level = (hidden_dim * (hidden_dim//2) + (hidden_dim//2)) + ((hidden_dim//2) * 2) + ((hidden_dim//2) * 1 + 1)
        level_attention_params = num_levels * level_attn_per_level
        
        # Cross-scale attention: 128->64, LN, ReLU, 64->3
        cross_scale_attn = (hidden_dim * (hidden_dim//2) + (hidden_dim//2)) + ((hidden_dim//2) * 2) + ((hidden_dim//2) * num_levels + num_levels)
        
        scale_attention_params = level_attention_params + cross_scale_attn
        
        # 3. Projection head: 128->128, LN, ReLU, 128->128
        proj_linear1 = hidden_dim * hidden_dim + hidden_dim
        proj_ln = hidden_dim * 2
        proj_linear2 = hidden_dim * hidden_dim + hidden_dim
        projection_head_params = proj_linear1 + proj_ln + proj_linear2
        
        total_params = total_gat_layers + scale_attention_params + projection_head_params
        
        return {
            'gat_layers': total_gat_layers,
            'scale_attention': scale_attention_params,
            'projection_head': projection_head_params,
            'total': total_params,
            'total_millions': total_params / 1e6,
            'breakdown': {
                'gat_layer_0': gat_layer0_total,
                'gat_layers_1_2': gat_layer_hidden_total * 2,
                'level_attention': level_attention_params,
                'cross_scale_attention': cross_scale_attn
            }
        }
    
    def calculate_resnet18_flops(self) -> Dict[str, float]:
        """
        Calculate ResNet18 FLOPs for 224x224 input (per patch)
        
        Returns:
            Dictionary with detailed FLOP breakdown
        """
        # Input: 3x224x224
        
        # Conv1: 3->64, 7x7, stride=2, padding=3 -> 64x112x112
        conv1_flops = 3 * 64 * 7 * 7 * 112 * 112
        
        # MaxPool: 3x3, stride=2 -> 64x56x56 (no FLOPs)
        
        # Layer 1: 2 blocks at 64x56x56
        # Block 1: 64->64 (3x3) + 64->64 (3x3)
        # Block 2: 64->64 (3x3) + 64->64 (3x3)
        layer1_flops = 2 * (64*64*3*3*56*56 + 64*64*3*3*56*56)
        
        # Layer 2: 2 blocks, first downsamples to 128x28x28
        # Block 1: 64->128 (3x3, stride=2) + 128->128 (3x3) + shortcut 64->128 (1x1, stride=2)
        layer2_block1 = 64*128*3*3*28*28 + 128*128*3*3*28*28 + 64*128*1*1*28*28
        # Block 2: 128->128 (3x3) + 128->128 (3x3)
        layer2_block2 = 128*128*3*3*28*28 + 128*128*3*3*28*28
        layer2_flops = layer2_block1 + layer2_block2
        
        # Layer 3: 2 blocks, first downsamples to 256x14x14
        # Block 1: 128->256 (3x3, stride=2) + 256->256 (3x3) + shortcut 128->256 (1x1, stride=2)
        layer3_block1 = 128*256*3*3*14*14 + 256*256*3*3*14*14 + 128*256*1*1*14*14
        # Block 2: 256->256 (3x3) + 256->256 (3x3)
        layer3_block2 = 256*256*3*3*14*14 + 256*256*3*3*14*14
        layer3_flops = layer3_block1 + layer3_block2
        
        # Layer 4: 2 blocks, first downsamples to 512x7x7
        # Block 1: 256->512 (3x3, stride=2) + 512->512 (3x3) + shortcut 256->512 (1x1, stride=2)
        layer4_block1 = 256*512*3*3*7*7 + 512*512*3*3*7*7 + 256*512*1*1*7*7
        # Block 2: 512->512 (3x3) + 512->512 (3x3)
        layer4_block2 = 512*512*3*3*7*7 + 512*512*3*3*7*7
        layer4_flops = layer4_block1 + layer4_block2
        
        # AdaptiveAvgPool: 512x7x7 -> 512x1x1 (no FLOPs)
        # FC layer removed in our case
        
        total_flops = conv1_flops + layer1_flops + layer2_flops + layer3_flops + layer4_flops
        
        return {
            'conv1': conv1_flops,
            'layer1': layer1_flops,
            'layer2': layer2_flops,
            'layer3': layer3_flops,
            'layer4': layer4_flops,
            'total_per_patch': total_flops,
            'total_all_patches': total_flops * self.num_patches,
            'total_gflops': (total_flops * self.num_patches) / 1e9
        }
    
    def calculate_mil_classifier_flops(self) -> Dict[str, float]:
        """
        Calculate MIL classifier FLOPs (per forward pass)
        
        Returns:
            Dictionary with detailed FLOP breakdown
        """
        feat_dim = 512
        proj_dim = 128
        num_patches = self.num_patches
        
        # 1. Patch projector (applied to all patches)
        patch_proj_linear1 = num_patches * feat_dim * feat_dim  # 484 * 512 * 512
        patch_proj_linear2 = num_patches * feat_dim * proj_dim  # 484 * 512 * 128
        patch_projector_flops = patch_proj_linear1 + patch_proj_linear2
        
        # 2. Attention mechanism (applied to all patches)
        attn_linear1 = num_patches * feat_dim * 128  # 484 * 512 * 128
        attn_linear2 = num_patches * 128 * 1  # 484 * 128 * 1
        attn_softmax = num_patches  # Softmax over patches
        attention_flops = attn_linear1 + attn_linear2 + attn_softmax
        
        # 3. Weighted aggregation (patient features)
        weighted_aggregation = num_patches * feat_dim  # 484 * 512
        
        # 4. Patient projector (applied once)
        patient_proj_linear1 = feat_dim * feat_dim  # 512 * 512
        patient_proj_linear2 = feat_dim * proj_dim  # 512 * 128
        patient_projector_flops = patient_proj_linear1 + patient_proj_linear2
        
        # 5. Classifier (applied once)
        classifier_linear1 = feat_dim * (feat_dim // 2)  # 512 * 256
        classifier_linear2 = (feat_dim // 2) * 1  # 256 * 1
        classifier_flops = classifier_linear1 + classifier_linear2
        
        total_flops = (patch_projector_flops + attention_flops + weighted_aggregation + 
                      patient_projector_flops + classifier_flops)
        
        return {
            'patch_projector': patch_projector_flops,
            'attention': attention_flops,
            'weighted_aggregation': weighted_aggregation,
            'patient_projector': patient_projector_flops,
            'classifier': classifier_flops,
            'total': total_flops,
            'total_gflops': total_flops / 1e9
        }
    
    def calculate_hiergat_flops(self) -> Dict[str, float]:
        """
        Calculate HierGAT FLOPs (per forward pass)
        Assumes graph with 484 nodes and ~2000 edges
        
        Returns:
            Dictionary with detailed FLOP breakdown
        """
        num_nodes = self.num_patches  # 484
        num_edges = num_nodes * 4  # Approximate edges (spatial + cross-scale)
        input_dim = 128
        hidden_dim = 128
        num_heads = 4
        head_dim = hidden_dim // num_heads  # 32
        
        # 1. GAT layers (3 layers)
        # Each GAT layer: message passing + attention computation
        # Message passing: num_edges * input_dim * head_dim * num_heads
        # Attention: num_edges * input_dim (for attention score computation)
        
        # Layer 0: input_dim -> hidden_dim
        gat_layer0_message = num_edges * input_dim * head_dim * num_heads
        gat_layer0_attention = num_edges * input_dim
        gat_layer0_total = (gat_layer0_message + gat_layer0_attention) * 2  # spatial + cross-scale
        
        # Layers 1-2: hidden_dim -> hidden_dim
        gat_layer_hidden_message = num_edges * hidden_dim * head_dim * num_heads
        gat_layer_hidden_attention = num_edges * hidden_dim
        gat_layer_hidden_total = (gat_layer_hidden_message + gat_layer_hidden_attention) * 2
        
        total_gat_flops = gat_layer0_total + 2 * gat_layer_hidden_total
        
        # 2. Scale-wise attention
        # Level-specific attention: num_levels * (nodes_per_level * hidden_dim * (hidden_dim//2) + ...)
        nodes_per_level = num_nodes // 3  # ~161 nodes per level
        level_attn_flops = 3 * (nodes_per_level * hidden_dim * (hidden_dim//2) + 
                                nodes_per_level * (hidden_dim//2) * 1)
        
        # Cross-scale attention: 3 levels * hidden_dim * (hidden_dim//2) + 3 * (hidden_dim//2) * 3
        cross_scale_flops = 3 * hidden_dim * (hidden_dim//2) + 3 * (hidden_dim//2) * 3
        
        scale_attention_flops = level_attn_flops + cross_scale_flops
        
        # 3. Projection head: hidden_dim * hidden_dim + hidden_dim * hidden_dim
        projection_flops = hidden_dim * hidden_dim + hidden_dim * hidden_dim
        
        total_flops = total_gat_flops + scale_attention_flops + projection_flops
        
        return {
            'gat_layers': total_gat_flops,
            'scale_attention': scale_attention_flops,
            'projection_head': projection_flops,
            'total': total_flops,
            'total_gflops': total_flops / 1e9
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
        
        # Detailed parameter calculation
        resnet18_params = self.calculate_resnet18_params()
        mil_classifier_params = self.calculate_mil_classifier_params()
        total_params = resnet18_params['total'] + mil_classifier_params['total']
        
        # Detailed FLOP calculation
        resnet18_flops = self.calculate_resnet18_flops()
        mil_classifier_flops = self.calculate_mil_classifier_flops()
        base_flops = resnet18_flops['total_all_patches'] + mil_classifier_flops['total']
        
        cam_factor = self.cam_factors[cam_method]
        total_flops = base_flops * cam_factor  # Total with CAM computation
        
        # Memory usage (FP32)
        model_memory = total_params * 4 / 1e9  # GB (4 bytes per parameter)
        feature_memory = self.num_patches * 512 * 4 / 1e9  # GB (feature activations)
        gradient_memory = feature_memory * (cam_factor - 1.0)  # Additional memory for gradients
        total_memory = model_memory + feature_memory + gradient_memory
        
        # Timing
        timing_result = self.estimate_pipeline1_time(cam_method)
        
        return {
            'parameters_millions': total_params / 1e6,
            'flops_gflops': total_flops / 1e9,
            'memory_gb': total_memory,
            'inference_time_ms': timing_result['total_time_ms'],
            'detailed_params': {
                'resnet18': resnet18_params,
                'mil_classifier': mil_classifier_params,
                'total_params': total_params
            },
            'detailed_flops': {
                'resnet18': resnet18_flops,
                'mil_classifier': mil_classifier_flops,
                'base_flops': base_flops,
                'cam_factor': cam_factor,
                'total_flops': total_flops
            },
            'components': {
                'models': 'training_step_1 (ResNet18 + MIL classifier)',
                'cam_overhead_factor': cam_factor
            }
        }
    
    def get_pipeline2_specs(self) -> Dict[str, float]:
        """
        Get detailed specifications for Pipeline 2 (GRAPHITE)
        
        Returns:
            Dictionary with parameters, FLOPs, memory, and timing specs
        """
        # Detailed parameter calculation
        resnet18_params = self.calculate_resnet18_params()
        mil_classifier_params = self.calculate_mil_classifier_params()
        hiergat_params = self.calculate_hiergat_params()
        total_params = resnet18_params['total'] + mil_classifier_params['total'] + hiergat_params['total']
        
        # Detailed FLOP calculation
        resnet18_flops = self.calculate_resnet18_flops()
        mil_classifier_flops = self.calculate_mil_classifier_flops()
        hiergat_flops = self.calculate_hiergat_flops()
        
        mil_base_flops = resnet18_flops['total_all_patches'] + mil_classifier_flops['total']
        hiergat_base_flops = hiergat_flops['total']
        fullgrad_additional_flops = mil_base_flops * (self.cam_factors['fullgrad'] - 1.0)
        fusion_flops = self.num_patches * 0.2e6  # Fusion processing
        
        total_flops = mil_base_flops + hiergat_base_flops + fullgrad_additional_flops + fusion_flops
        
        # Memory usage (FP32)
        model_memory = total_params * 4 / 1e9  # GB (4 bytes per parameter)
        feature_memory = self.num_patches * 512 * 4 / 1e9  # GB (MIL features)
        graph_memory = self.num_patches * 128 * 4 / 1e9  # GB (HierGAT features)
        attention_maps_memory = 3 * (self.num_patches * 4) / 1e9  # GB (3 attention maps)
        gradient_memory = feature_memory * 1.5  # Additional memory for FullGrad gradients
        total_memory = model_memory + feature_memory + graph_memory + attention_maps_memory + gradient_memory
        
        # Timing
        timing_result = self.estimate_pipeline2_time()
        
        return {
            'parameters_millions': total_params / 1e6,
            'flops_gflops': total_flops / 1e9,
            'memory_gb': total_memory,
            'inference_time_ms': timing_result['total_time_ms'],
            'detailed_params': {
                'resnet18': resnet18_params,
                'mil_classifier': mil_classifier_params,
                'hiergat': hiergat_params,
                'total_params': total_params
            },
            'detailed_flops': {
                'resnet18': resnet18_flops,
                'mil_classifier': mil_classifier_flops,
                'hiergat': hiergat_flops,
                'mil_base_flops': mil_base_flops,
                'fullgrad_additional_flops': fullgrad_additional_flops,
                'fusion_flops': fusion_flops,
                'total_flops': total_flops
            },
            'components': {
                'models': 'training_step_1 + training_step_2 (ResNet18 + MIL + HierGAT)',
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
            },
            'detailed_breakdown': {
                'pipeline1': p1_specs,
                'graphite': p2_specs
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
        
        # Calculate base inference time using detailed FLOPs
        resnet18_flops = self.calculate_resnet18_flops()
        mil_classifier_flops = self.calculate_mil_classifier_flops()
        base_flops = resnet18_flops['total_all_patches'] + mil_classifier_flops['total']
        
        base_time_ms = (base_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
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
        # Calculate base inference times using detailed FLOPs
        resnet18_flops = self.calculate_resnet18_flops()
        mil_classifier_flops = self.calculate_mil_classifier_flops()
        hiergat_flops = self.calculate_hiergat_flops()
        
        mil_base_flops = resnet18_flops['total_all_patches'] + mil_classifier_flops['total']
        hiergat_base_flops = hiergat_flops['total']
        
        # Base model inference times
        mil_inference_time = (mil_base_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        hiergat_inference_time = (hiergat_base_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
        # 1. MIL attention map generation (using training_step_1 results)
        mil_attention_map_time = self.num_patches * 0.03  # Extract attention from MIL model
        
        # 2. CAM map using FullGrad (separate computation on training_step_1)
        fullgrad_cam_gradient_flops = mil_base_flops * (self.cam_factors['fullgrad'] - 1.0)
        fullgrad_cam_gradient_time = (fullgrad_cam_gradient_flops / (self.gpu_tflops * 1e12 * self.efficiency)) * 1000
        
        # 3. Multi-level Fusion map (training_step_1 + training_step_2)
        multilevel_level_generation = self.num_patches * 0.05  # Extract HierGAT Level 0/1/2 maps
        multilevel_fusion_processing = self.num_patches * 0.08  # Gaussian smoothing + weighted combination
        multilevel_fusion_map_time = multilevel_level_generation + multilevel_fusion_processing
        
        # 4. Final Fusion (combine the three maps: multilevel + MIL + FullGrad)
        final_fusion_time = self.num_patches * 0.1  # Combine 3 attention maps
        
        # 5. Post-processing (visualization rendering)
        post_processing_time = 100.0 + (self.num_patches * 0.2)  # Rendering and visualization
        
        # Total time
        total_time = (mil_inference_time +           # training_step_1 base inference
                      hiergat_inference_time +       # training_step_2 base inference  
                      mil_attention_map_time +       # MIL attention map generation
                      fullgrad_cam_gradient_time +   # FullGrad gradient computation
                      multilevel_fusion_map_time +   # Multi-level fusion processing
                      final_fusion_time +            # Final fusion of 3 maps
                      post_processing_time)          # Post-processing
        
        return {
            'total_time_ms': total_time,
            'mil_inference_ms': mil_inference_time,
            'hiergat_inference_ms': hiergat_inference_time,
            'mil_attention_map_ms': mil_attention_map_time,
            'fullgrad_cam_map_ms': fullgrad_cam_gradient_time,
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
    
    # Get detailed specifications
    p1_specs = estimator.get_pipeline1_specs('fullgrad')
    p2_specs = estimator.get_pipeline2_specs()
    
    print(f"\nDetailed Parameter Analysis:")
    print(f"Pipeline 1 (FullGrad): {p1_specs['parameters_millions']:.1f}M parameters")
    print(f"  ResNet18: {p1_specs['detailed_params']['resnet18']['total_millions']:.1f}M")
    print(f"  MIL Classifier: {p1_specs['detailed_params']['mil_classifier']['total_millions']:.1f}M")
    
    print(f"\nGRAPHITE: {p2_specs['parameters_millions']:.1f}M parameters")
    print(f"  ResNet18: {p2_specs['detailed_params']['resnet18']['total_millions']:.1f}M")
    print(f"  MIL Classifier: {p2_specs['detailed_params']['mil_classifier']['total_millions']:.1f}M")
    print(f"  HierGAT: {p2_specs['detailed_params']['hiergat']['total_millions']:.1f}M")
    
    print(f"\nDetailed FLOP Analysis:")
    print(f"Pipeline 1 (FullGrad): {p1_specs['flops_gflops']:.0f} GFLOPs")
    print(f"  ResNet18: {p1_specs['detailed_flops']['resnet18']['total_gflops']:.0f} GFLOPs")
    print(f"  MIL Classifier: {p1_specs['detailed_flops']['mil_classifier']['total_gflops']:.0f} GFLOPs")
    print(f"  CAM Factor: {p1_specs['detailed_flops']['cam_factor']:.1f}x")
    
    print(f"\nGRAPHITE: {p2_specs['flops_gflops']:.0f} GFLOPs")
    print(f"  ResNet18: {p2_specs['detailed_flops']['resnet18']['total_gflops']:.0f} GFLOPs")
    print(f"  MIL Classifier: {p2_specs['detailed_flops']['mil_classifier']['total_gflops']:.0f} GFLOPs")
    print(f"  HierGAT: {p2_specs['detailed_flops']['hiergat']['total_gflops']:.0f} GFLOPs")
    print(f"  FullGrad Additional: {p2_specs['detailed_flops']['fullgrad_additional_flops']/1e9:.0f} GFLOPs")
    
    # Compare pipelines
    comparison = estimator.compare_pipelines('fullgrad')
    print(f"\nTiming Comparison:")
    print(f"  Pipeline 1 (FullGrad):    {comparison['pipeline1_ms']:.0f} ms")
    print(f"  GRAPHITE (FullGrad):      {comparison['pipeline2_ms']:.0f} ms")
    print(f"  Complexity ratio:         {comparison['complexity_ratio']:.1f}x") 