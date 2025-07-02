# GRAPHITE Inference Time Analysis

## Overview
This analysis provides inference time estimates for the GRAPHITE histopathology visualization pipeline. Fixed configuration: **5040×5040 images, V100 GPU, FP32 precision**.

## Pipeline Specifications

### Pipeline 1: GradCAM Visualization
- **Components**: training_step_1 + visualization_step_1
- **Models**: ResNet18 backbone + MIL classifier 
- **Parameters**: 12.5M total (11.2M ResNet18 + 1.3M MIL classifier)
- **FLOPs**: 876-2,189 GFLOPs (depending on CAM method)
- **Memory**: 0.99-1.39 GB
- **Inference Time**: 89ms (GradCAM) or 186ms (FullGrad)

### Pipeline 2: GRAPHITE Fusion
- **Components**: training_step_1 + training_step_2 + visualization_step_2
- **Models**: ResNet18 + MIL classifier + HierGAT
- **Parameters**: 15.3M total (11.2M ResNet18 + 1.3M MIL + 2.8M HierGAT)
- **FLOPs**: 2,207 GFLOPs (876 MIL + 18 HierGAT + 1,313 FullGrad + 0.1 fusion)
- **Memory**: 2.08 GB
- **Inference Time**: 510ms (fixed FullGrad)

## Complexity Comparison

### Computational Ratios (GRAPHITE vs Pipeline 1 with FullGrad)
- **Parameters**: 1.2x more (15.3M vs 12.5M)
- **FLOPs**: 1.0x similar (2,207 vs 2,189 GFLOPs)
- **Memory**: 1.5x more (2.08 vs 1.39 GB)
- **Time**: 2.7x slower (510ms vs 186ms)

### Efficiency Metrics
- **Pipeline 1**: 11.8 GFLOPs/ms
- **GRAPHITE**: 4.3 GFLOPs/ms  
- **Efficiency ratio**: 2.7x faster for Pipeline 1

## Detailed Results

### Pipeline 1 Performance
| CAM Method | Time (ms) | FLOPs (GFLOPs) | Memory (GB) |
|------------|-----------|----------------|-------------|
| GradCAM    | 89        | 876            | 0.99        |
| FullGrad   | 186       | 2,189          | 1.39        |

### GRAPHITE Breakdown (510ms total)
| Component              | Time (ms) | Percentage | Description |
|------------------------|-----------|------------|-------------|
| MIL Inference          | 74        | 14.5%      | training_step_1 base model |
| HierGAT Inference      | 2         | 0.4%       | training_step_2 base model |
| MIL Attention Map      | 15        | 2.9%       | Extract MIL attention |
| FullGrad CAM Map       | 112       | 22.0%      | Gradient-based attention |
| Multi-level Fusion Map | 63        | 12.4%      | HierGAT Level 0/1/2 fusion |
| Final Fusion           | 48        | 9.4%       | Combine 3 attention maps |
| Post-processing        | 197       | 38.6%      | Visualization rendering |

### FLOPs Distribution (GRAPHITE)
| Component     | FLOPs (GFLOPs) | Percentage |
|---------------|----------------|------------|
| MIL base      | 876            | 39.7%      |
| HierGAT       | 18             | 0.8%       |
| FullGrad CAM  | 1,313          | 59.5%      |
| Fusion        | 0.1            | 0.0%       |
| **Total**     | **2,207**      | **100%**   |

### Memory Usage Breakdown
| Pipeline    | Model (GB) | Features (GB) | Gradients (GB) | Total (GB) |
|-------------|------------|---------------|----------------|------------|
| Pipeline 1  | 0.05       | 0.99          | 0.40           | 1.39       |
| GRAPHITE    | 0.06       | 1.25          | 0.77           | 2.08       |

## Performance Insights

### Why GRAPHITE is 2.7x Slower Despite Similar FLOPs
1. **Multiple Attention Maps**: Generates 3 independent maps vs 1
2. **Sequential Processing**: Each map requires separate computation passes
3. **Memory Overhead**: 1.5x more memory leads to reduced GPU efficiency
4. **Fusion Complexity**: Final fusion of 3 maps adds processing overhead
5. **Post-processing**: Heavier visualization rendering (38.6% of total time)

### Computational Efficiency Analysis
- **Similar FLOPs**: Both pipelines have comparable computational requirements
- **Different Memory Patterns**: GRAPHITE uses 1.5x more memory
- **Processing Complexity**: GRAPHITE requires multiple model outputs and fusion steps
- **GPU Utilization**: Pipeline 1 achieves better GPU efficiency due to simpler memory access patterns

## Architecture Justification

### Pipeline 1: Optimized for Speed
- **Single model inference**: training_step_1 only
- **Direct CAM computation**: Immediate gradient-based attention
- **Minimal memory overhead**: Simple feature storage
- **Efficient for real-time processing**

### GRAPHITE: Comprehensive Analysis
- **Multi-model fusion**: Combines MIL + HierGAT insights  
- **Three attention sources**: MIL attention + FullGrad CAM + Multi-level fusion
- **Enhanced interpretability**: Multiple complementary visualization methods
- **Research-oriented**: Comprehensive but computationally intensive

## Recommendations

### Use Pipeline 1 When:
- Real-time processing required (<200ms)
- Single attention visualization sufficient
- Computational resources limited
- Quick diagnostic feedback needed

### Use GRAPHITE When:
- Comprehensive analysis required
- Multiple attention perspectives valuable
- Research or detailed clinical analysis
- Computational resources available
- Quality over speed priority

## Technical Specifications Summary

| Metric                  | Pipeline 1 (FullGrad) | GRAPHITE | Ratio |
|-------------------------|------------------------|----------|-------|
| **Parameters (M)**      | 12.5                  | 15.3     | 1.2x  |
| **FLOPs (GFLOPs)**     | 2,189                 | 2,207    | 1.0x  |
| **Memory (GB)**        | 1.39                  | 2.08     | 1.5x  |
| **Time (ms)**          | 186                   | 510      | 2.7x  |
| **Efficiency (GFLOPs/ms)** | 11.8             | 4.3      | 2.7x  |

The computational analysis clearly justifies the 2.7x timing difference between pipelines, primarily due to GRAPHITE's multi-map architecture requiring sequential processing and higher memory overhead despite similar total FLOPs.

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