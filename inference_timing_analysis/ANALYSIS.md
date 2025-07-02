# GRAPHITE Inference Time Analysis

Simple analysis for GRAPHITE visualization pipeline performance.

**Fixed Configuration**: 5040×5040 images (484 patches), V100 GPU, FP32 precision

## Pipeline Overview

### Pipeline 1: GradCAM Visualization
- **Components**: `training_step_1` + `visualization_step_1`
- **Purpose**: Basic gradient-based attention visualization
- **Process**: MIL model → CAM generation → visualization

### Pipeline 2: GRAPHITE Fusion  
- **Components**: `training_step_1` + `training_step_2` + `visualization_step_2`
- **Purpose**: Enhanced multi-level fusion visualization
- **Architecture**: 5-stage process:
  1. MIL attention map (training_step_1)
  2. CAM map using FullGrad (training_step_1) 
  3. Multi-level Fusion map (training_step_1 + training_step_2)
  4. Final Fusion (combine multilevel + MIL + FullGrad results)
  5. Post-processing (visualization rendering)

## Performance Results

### Pipeline Comparison

| Pipeline | Time | CAM Method | Description |
|----------|------|------------|-------------|
| **Pipeline 1** | **186 ms** | FullGrad | GradCAM visualization |
| **GRAPHITE** | **510 ms** | FullGrad (separate) | Complete GRAPHITE fusion |
| **Complexity Ratio** | **2.7x** | - | GRAPHITE is 2.7x more complex |

### CAM Method Comparison (Pipeline 1)

| CAM Method | Time | Overhead | Use Case |
|------------|------|----------|----------|
| GradCAM | 89 ms | 1.2x | Fast processing, real-time |
| FullGrad | 186 ms | 2.5x | Better quality, moderate speed |

### GRAPHITE Breakdown (5-component architecture)

| Component | Time | Percentage | Description |
|-----------|------|------------|-------------|
| MIL Inference | 74 ms | 14.5% | training_step_1 base inference |
| HierGAT Inference | 2 ms | 0.4% | training_step_2 base inference |
| MIL Attention Map | 15 ms | 2.9% | Extract attention from MIL model |
| FullGrad CAM Map | 112 ms | 22.0% | Separate FullGrad computation |
| Multi-level Fusion Map | 63 ms | 12.4% | HierGAT level processing + combination |
| Final Fusion | 48 ms | 9.4% | Combine 3 attention maps |
| Post-processing | 197 ms | 38.6% | Visualization rendering |
| **Total** | **510 ms** | **100%** | Complete GRAPHITE pipeline |

## Key Insights

### Performance Distribution
- **Pipeline 1**: 40% model inference + 60% CAM processing
- **GRAPHITE**: 15% model inference + 37% map generation + 9% final fusion + 39% post-processing

### Bottlenecks
1. **Post-processing (39%)**: Visualization rendering - largest optimization opportunity
2. **FullGrad CAM Map (22%)**: Separate gradient computation for fusion
3. **Multi-level Fusion Map (12%)**: HierGAT level processing + combination
4. **Final fusion (9%)**: Three attention map integration
5. **Model inference (15%)**: MIL + HierGAT base computation

### Architecture Advantages
- **Pipeline 1**: 2.7x faster, simple, real-time capable
- **GRAPHITE**: Comprehensive analysis, multi-level insights, research-grade

## Usage

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