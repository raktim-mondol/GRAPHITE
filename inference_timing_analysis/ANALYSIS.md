# GRAPHITE Inference Time Analysis

## Overview
This analysis provides inference time estimates for the GRAPHITE histopathology visualization pipeline. Fixed configuration: **5040×5040 images, V100 GPU, FP32 precision**.

## Detailed Architecture Analysis

### ResNet18 Parameter Breakdown (11.2M parameters)
| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Conv1 + BN | 9,536 | 3×64×7×7 + 64×2 |
| Layer1 (2 blocks) | 147,968 | 2×(64×64×3×3 + 64×64×3×3 + 64×4) |
| Layer2 (2 blocks) | 525,568 | Downsample block + regular block |
| Layer3 (2 blocks) | 2,099,712 | Downsample block + regular block |
| Layer4 (2 blocks) | 8,393,728 | Downsample block + regular block |
| **Total** | **11,176,512** | **11.2M parameters** |

### MIL Classifier Parameter Breakdown (0.9M parameters)
| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Patch Projector | 329,600 | 512×512+512 + 512×2 + 512×128+128 + 128×2 |
| Attention | 66,049 | 512×128+128 + 128×2 + 128×1+1 |
| Patient Projector | 329,600 | Same as patch projector |
| Classifier | 132,097 | 512×256+256 + 256×2 + 256×1+1 |
| Patient LayerNorm | 1,024 | 512×2 |
| **Total** | **858,370** | **0.9M parameters** |

### HierGAT Parameter Breakdown (0.2M parameters)
| Component | Parameters | Calculation |
|-----------|------------|-------------|
| GAT Layers (3 layers) | 102,912 | 2×GATConv + LayerNorm per layer |
| ScaleWiseAttention | 33,926 | 3 level attention + cross-scale |
| Projection Head | 33,280 | 128×128+128 + 128×2 + 128×128+128 |
| **Total** | **170,118** | **0.2M parameters** |

## Detailed FLOP Analysis

### ResNet18 FLOP Breakdown (877.8 GFLOPs for 484 patches)
| Layer | FLOPs per patch | Total FLOPs (484 patches) |
|-------|-----------------|---------------------------|
| Conv1 | 118,013,952 | 57.1 GFLOPs |
| Layer1 | 462,422,016 | 223.7 GFLOPs |
| Layer2 | 411,041,792 | 199.0 GFLOPs |
| Layer3 | 411,041,792 | 199.0 GFLOPs |
| Layer4 | 411,041,792 | 199.0 GFLOPs |
| **Total** | **1,813,561,344** | **877.8 GFLOPs** |

### MIL Classifier FLOP Breakdown (0.2 GFLOPs)
| Component | FLOPs | Calculation |
|-----------|-------|-------------|
| Patch Projector | 158,597,120 | 484×(512×512 + 512×128) |
| Attention | 31,781,860 | 484×(512×128 + 128×1) + softmax |
| Weighted Aggregation | 247,808 | 484×512 |
| Patient Projector | 327,680 | 512×512 + 512×128 |
| Classifier | 131,328 | 512×256 + 256×1 |
| **Total** | **191,085,796** | **0.2 GFLOPs** |

### HierGAT FLOP Breakdown (0.2 GFLOPs)
| Component | FLOPs | Description |
|-----------|-------|-------------|
| GAT Layers | 191,803,392 | Message passing + attention (3 layers) |
| Scale Attention | 4,012,800 | Level-specific + cross-scale attention |
| Projection Head | 32,768 | 128×128 + 128×128 |
| **Total** | **195,848,960** | **0.2 GFLOPs** |

## Pipeline Specifications

### Pipeline 1: GradCAM Visualization
- **Components**: training_step_1 + visualization_step_1
- **Models**: ResNet18 (11.2M) + MIL classifier (0.9M) 
- **Total Parameters**: 12.0M
- **Base FLOPs**: 878.0 GFLOPs
- **With CAM Methods**:
  - GradCAM (1.2x): 1,053.5 GFLOPs, 89ms
  - FullGrad (2.5x): 2,194.9 GFLOPs, 186ms

### Pipeline 2: GRAPHITE Fusion
- **Components**: training_step_1 + training_step_2 + visualization_step_2
- **Models**: ResNet18 (11.2M) + MIL classifier (0.9M) + HierGAT (0.2M)
- **Total Parameters**: 12.2M
- **Base FLOPs**: 878.2 GFLOPs (MIL + HierGAT)
- **Additional FLOPs**:
  - FullGrad computation: 1,316.9 GFLOPs
  - Fusion processing: 0.1 GFLOPs
- **Total FLOPs**: 2,195.2 GFLOPs
- **Inference Time**: 510ms (fixed FullGrad)

## Complexity Analysis

### Parameter Comparison (GRAPHITE vs Pipeline 1 FullGrad)
- **Total Parameters**: 12.2M vs 12.0M = **1.02x more**
- **Additional HierGAT**: 170K parameters (1.4% increase)
- **Parameter efficiency**: Very similar parameter count

### FLOP Comparison (GRAPHITE vs Pipeline 1 FullGrad)
- **Total FLOPs**: 2,195.2 vs 2,194.9 GFLOPs = **1.00x similar**
- **FLOP distribution**:
  - ResNet18: 877.8 GFLOPs (40.0%) - same for both
  - MIL Classifier: 0.2 GFLOPs (0.0%) - same for both
  - FullGrad: 1,316.9 GFLOPs (60.0%) - same for both
  - HierGAT: 0.2 GFLOPs (0.0%) - GRAPHITE only
  - Fusion: 0.1 GFLOPs (0.0%) - GRAPHITE only

### Performance Gap Analysis
**Why GRAPHITE is 2.7x slower despite similar parameters/FLOPs:**

1. **Sequential Processing**: GRAPHITE requires 5 sequential stages vs Pipeline 1's single stage
2. **Memory Access Patterns**: Multiple attention map generations require different memory access patterns
3. **GPU Utilization**: Sequential stages reduce GPU parallelization efficiency
4. **Intermediate Storage**: Need to store 3 separate attention maps simultaneously
5. **Post-processing Overhead**: Final fusion and visualization rendering (38.6% of total time)

## Detailed Timing Breakdown

### GRAPHITE Component Analysis (510ms total)
| Component | Time (ms) | FLOPs (GFLOPs) | Efficiency (GFLOPs/ms) |
|-----------|-----------|----------------|------------------------|
| MIL Inference | 74 | 878.0 | 11.9 |
| HierGAT Inference | 2 | 0.2 | 0.1 |
| MIL Attention Map | 15 | - | - |
| FullGrad CAM Map | 112 | 1,316.9 | 11.8 |
| Multi-level Fusion | 63 | 0.1 | 0.002 |
| Final Fusion | 48 | - | - |
| Post-processing | 197 | - | - |

### Efficiency Analysis
- **High-efficiency components**: MIL/ResNet18 inference (11.8-11.9 GFLOPs/ms)
- **Low-efficiency components**: Fusion and post-processing (memory-bound operations)
- **Overall Pipeline 1 efficiency**: 11.8 GFLOPs/ms
- **Overall GRAPHITE efficiency**: 4.3 GFLOPs/ms

## Architecture Justification

### Computational Complexity
The detailed analysis shows that GRAPHITE's 2.7x slower performance is **NOT** due to higher computational requirements:
- Similar parameter count (1.02x difference)
- Identical FLOP count (1.00x difference)
- Same core computational kernels (ResNet18 + MIL)

### Performance Bottlenecks
The performance difference comes from:
1. **Pipeline Architecture**: Sequential vs parallel processing
2. **Memory Efficiency**: Multiple intermediate results storage
3. **Fusion Overhead**: Non-parallelizable attention map combination
4. **Visualization Processing**: Extensive post-processing pipeline

### Trade-off Analysis
- **Pipeline 1**: Optimized for speed, single attention source
- **GRAPHITE**: Comprehensive analysis, multiple complementary attention maps
- **Computational cost**: Minimal additional compute for significant analytical enhancement

## Recommendations

### Use Pipeline 1 When:
- Real-time processing required (<200ms)
- Single attention visualization sufficient
- Resource-constrained environments
- Quick diagnostic feedback needed

### Use GRAPHITE When:
- Research-grade analysis required
- Multiple attention perspectives valuable
- Computational resources available
- Quality and comprehensiveness prioritized over speed

## Technical Summary

| Metric | Pipeline 1 (FullGrad) | GRAPHITE | Ratio |
|--------|------------------------|----------|-------|
| **Parameters** | 12.0M | 12.2M | 1.02x |
| **FLOPs** | 2,194.9 GFLOPs | 2,195.2 GFLOPs | 1.00x |
| **Memory** | 1.39 GB | 2.08 GB | 1.5x |
| **Time** | 186 ms | 510 ms | 2.7x |
| **Efficiency** | 11.8 GFLOPs/ms | 4.3 GFLOPs/ms | 2.7x |

**Key Insight**: The 2.7x performance difference is primarily due to architectural and memory efficiency factors, not computational complexity. GRAPHITE provides comprehensive multi-modal attention analysis at the cost of sequential processing overhead.

```python
from inference_time_estimator import create_estimator

estimator = create_estimator()

# Compare pipelines
comparison = estimator.compare_pipelines('fullgrad')
print(f"Pipeline 1: {comparison['pipeline1_ms']:.0f}ms")
print(f"Pipeline 2: {comparison['pipeline2_ms']:.0f}ms")

# Get detailed GRAPHITE breakdown  
graphite_details = estimator.estimate_pipeline2_time()
print(f"MIL inference: {graphite_details['mil_inference_ms']:.0f}ms")
print(f"FullGrad CAM map: {graphite_details['fullgrad_cam_map_ms']:.0f}ms")
print(f"Multi-level fusion map: {graphite_details['multilevel_fusion_map_ms']:.0f}ms")
```

## Recommendations

### Choose Pipeline 1 When:
- Real-time processing required (<100ms)
- Simple attention visualization sufficient
- Resource constraints critical
- High throughput needed

### Choose GRAPHITE When:
- Comprehensive analysis required
- Research applications  
- Quality over speed priority
- Multi-level insights needed
- Need 3 complementary attention maps (MIL + FullGrad + Multi-level)
- Want independent map generation for better fusion quality

---

**Configuration**: 5040×5040 pixels, V100 GPU, FP32 precision  
**Memory Usage**: <1GB GPU memory for both pipelines  
**Validation**: Based on actual GRAPHITE model architectures 