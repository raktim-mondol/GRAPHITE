# GRAPHITE Inference Timing Analysis

Simple inference time estimation for GRAPHITE histopathology visualization pipeline.

**Fixed Configuration**: 5040×5040 images (484 patches), V100 GPU, FP32 precision

## Files

- **`inference_time_estimator.py`** - Core estimation tool
- **`ANALYSIS.md`** - Performance analysis and results  
- **`test.py`** - Simple test script

## Quick Start

```python
from inference_time_estimator import create_estimator

estimator = create_estimator()

# Compare both pipelines
comparison = estimator.compare_pipelines('fullgrad')
print(f"Pipeline 1: {comparison['pipeline1_ms']:.0f}ms")
print(f"Pipeline 2: {comparison['pipeline2_ms']:.0f}ms")
```

## Key Results

| Pipeline | Time | Use Case |
|----------|------|----------|
| **Pipeline 1** (GradCAM) | **86 ms** | Real-time processing |
| **Pipeline 2** (Fusion) | **510 ms** | Research analysis |

Pipeline 2 is **6x more complex** but provides comprehensive multi-level analysis.

## Run Test

```bash
python test.py
```

---

**Note**: This analysis focuses on the most common use case. For detailed analysis see `ANALYSIS.md`. 