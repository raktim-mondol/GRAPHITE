# GRAPHITE Inference Timing Analysis

This folder contains comprehensive inference time estimation tools and analysis for the GRAPHITE histopathology visualization pipeline.

## 📁 Contents

### 🔧 **Core Tool**
- **`inference_time_estimator.py`** - Streamlined inference time estimation engine
  - Supports V100 GPU specifications
  - FP32 and FP16 precision modes
  - Two main estimation functions:
    - `estimate_cam_visualization_time()` - For GradCAM only pipeline
    - `estimate_fusion_visualization_time()` - For full GRAPHITE pipeline

### 📊 **Analysis Documents**
- **`PIPELINE_COMPARISON_TABLE.md`** - Detailed parameter/FLOPS/timing comparison
  - Side-by-side analysis of both pipelines
  - Parameters: 12.5M vs 27.5M (2.2x difference)
  - FLOPS: 875.7G vs 1,784.9G (2.0x difference)  
  - Timing: 1.12s vs 2.51s (2.2x difference)
  - Cost-benefit analysis and use case recommendations

- **`GRAPHITE_PIPELINE_INFERENCE_ANALYSIS.md`** - Comprehensive 12KB analysis
  - Detailed breakdown of both pipelines
  - Realistic performance estimates based on actual model architectures
  - Memory usage analysis and bottleneck identification
  - Production deployment recommendations

- **`INFERENCE_SUMMARY.md`** - Concise 2.5KB summary
  - Key results and performance comparisons
  - Quick reference for deployment decisions

- **`INFERENCE_ANALYSIS_5040x5040_OLD.md`** - Previous enhanced scaling analysis
  - Historical reference with scaling characteristics

## 🎯 **Two GRAPHITE Pipelines Analyzed**

### **Pipeline 1: GradCAM Only**
```
training_step_1 + visualization_step_1/xai_visualization
```
- **Purpose**: Basic gradient-based attention visualization
- **Time**: 0.74-1.12 seconds (FP16-FP32)
- **Memory**: 0.6 GB
- **Use Case**: Real-time screening, rapid analysis

### **Pipeline 2: Full GRAPHITE**
```
training_step_1 + training_step_2 + visualization_step_1 + visualization_step_2
```
- **Purpose**: Enhanced multi-level fusion visualization
- **Time**: 1.61-2.51 seconds (FP16-FP32)
- **Memory**: 1.2 GB
- **Use Case**: Comprehensive research analysis

## 🚀 **Quick Usage**

```python
from inference_time_estimator import create_inference_estimator

# Create estimator for V100 GPU
estimator = create_inference_estimator('V100', 'fp16')

# Estimate GradCAM pipeline (5040x5040 image)
gradcam_timing = estimator.estimate_cam_visualization_time((5040, 5040), 'fullgrad')
print(f"GradCAM: {gradcam_timing['inference_time_ms']:.0f} ms")

# Estimate full GRAPHITE pipeline
fusion_timing = estimator.estimate_fusion_visualization_time((5040, 5040), 'fullgrad', 'confidence')
print(f"Full GRAPHITE: {fusion_timing['total_time_ms']:.0f} ms")
```

## 📈 **Key Performance Results (5040×5040 images, V100 GPU)**

| Pipeline | Precision | Time | Memory | Throughput |
|----------|-----------|------|--------|------------|
| GradCAM Only | FP16 | 0.74 sec | 0.6 GB | 4,865 img/hour |
| GradCAM Only | FP32 | 1.12 sec | 0.6 GB | 3,203 img/hour |
| Full GRAPHITE | FP16 | 1.61 sec | 1.2 GB | 2,239 img/hour |
| Full GRAPHITE | FP32 | 2.51 sec | 1.2 GB | 1,436 img/hour |

## ✅ **Validation**

All estimates are based on:
- Actual GRAPHITE model architectures (ResNet18-TIAToolbox, HierGAT)
- Realistic GPU utilization factors (75-80% efficiency)
- Production overhead considerations
- Empirical FLOPS measurements

## 🔗 **Related**

This analysis supports the main GRAPHITE codebase located in:
- `training_step_1/` - MIL classification model
- `training_step_2/` - HierGAT self-supervised training
- `visualization_step_1/` - XAI visualization
- `visualization_step_2/` - Fusion visualization

---

**Generated**: Comprehensive inference time analysis for GRAPHITE pipelines  
**Target**: 5040×5040 images (25.4 MP, 484 patches)  
**GPU**: NVIDIA V100 (32GB) 