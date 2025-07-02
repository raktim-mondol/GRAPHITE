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
- **CAM Method**: Fixed to FullGrad for optimal fusion quality
- **Process**: Two-step fusion:
  1. Multi-level fusion: HierGAT levels → weighted combination
  2. Final fusion: multilevel + MIL + FullGrad → final heatmap

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

### GRAPHITE Breakdown (with separate FullGrad CAM)

| Component | Time | Percentage | Description |
|-----------|------|------------|-------------|
| Core Inference | 76 ms | 14.9% | MIL (74ms) + HierGAT (2ms) |
| Multi-level Fusion | 63 ms | 12.4% | Level extraction + combination |
| FullGrad CAM | 112 ms | 22.0% | Separate FullGrad computation for fusion |
| Final Fusion | 63 ms | 12.4% | Combine multilevel + MIL + FullGrad results |
| Post-processing | 197 ms | 38.6% | Visualization rendering |
| **Total** | **510 ms** | **100%** | Complete GRAPHITE pipeline |

## Key Insights

### Performance Distribution
- **Pipeline 1**: 40% model inference + 60% CAM processing
- **GRAPHITE**: 15% model inference + 22% FullGrad CAM + 25% fusion + 39% post-processing

### Bottlenecks
1. **Post-processing (39%)**: Visualization rendering - largest optimization opportunity
2. **FullGrad CAM (22%)**: Separate gradient computation for fusion
3. **Multi-level fusion (12%)**: HierGAT level processing
4. **Final fusion (12%)**: Three-component integration
5. **Model inference (15%)**: Efficient core computation

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
print(f"Core inference: {graphite_details['core_inference_ms']:.0f}ms")
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
- Best fusion quality required (uses FullGrad)

---

**Configuration**: 5040×5040 pixels, V100 GPU, FP32 precision  
**Memory Usage**: <1GB GPU memory for both pipelines  
**Validation**: Based on actual GRAPHITE model architectures 