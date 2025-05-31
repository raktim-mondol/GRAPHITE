# HierGAT Multi-Level Fusion Visualization

This system provides comprehensive HierGAT-specific visualization with configurable multilevel fusion for hierarchical graph attention analysis of tissue microarray cores.

## Overview

The `main_multi_level_fusion.py` script implements a specialized HierGAT visualization system that features:
- **Individual Level Analysis**: Separate visualization for Level 0, Level 1, and Level 2 attention
- **Configurable Multilevel Fusion**: User-customizable weights for combining different levels
- **Comprehensive Metrics**: Performance evaluation for each level and fusion combinations
- **Core Mask Integration**: Tissue-specific attention refinement
- **Excel Export**: Detailed metrics export for each processed core

## Key Features

### Hierarchical Level Processing
- **Level 0**: Fine-grained attention at base resolution
- **Level 1**: Medium-scale attention patterns (2x scale)
- **Level 2**: Coarse-scale attention patterns (4x scale)
- **Smart Smoothing**: Level-appropriate Gaussian filtering

### Multilevel Fusion Strategies
- **Weighted Combination**: Customizable weights for each level
- **Automatic Normalization**: Weight normalization for consistency
- **Scale-Aware Processing**: Level-specific smoothing parameters
- **Core-Masked Output**: Tissue-specific attention focus

### Performance Evaluation
- **Individual Level Metrics**: Separate evaluation for each HierGAT level
- **Fusion Metrics**: Combined performance assessment
- **Multi-Threshold Analysis**: Comprehensive threshold-based evaluation
- **Excel Reports**: Detailed per-image metrics export

### Visualization Components
- **Original Image**: Source tissue microarray core
- **Core Mask Overlay**: Extracted tissue boundaries
- **Annotated Mask**: Ground truth annotations
- **Individual Level Maps**: Separate attention visualizations
- **Multilevel Fusion**: Combined attention heatmap
- **Statistical Summary**: Comprehensive analysis statistics

## Usage

### Basic Usage (Default Weights: 0.5, 0.3, 0.2)
```bash
python main_multi_level_fusion.py
```

### Custom Weight Configuration
```bash
# Custom fusion weights for Level 0, Level 1, Level 2
python main_multi_level_fusion.py --level_weights 0.4 0.4 0.2

# Equal weighting across all levels
python main_multi_level_fusion.py --level_weights 0.33 0.33 0.34

# Focus on fine-grained features (high Level 0 weight)
python main_multi_level_fusion.py --level_weights 0.7 0.2 0.1
```

### Advanced Configuration
```bash
# Process single image with custom weights
python main_multi_level_fusion.py \
    --single_image /path/to/image.png \
    --level_weights 0.6 0.3 0.1

# Custom model and data paths
python main_multi_level_fusion.py \
    --model_path /path/to/hiergat_model.pt \
    --mil_model_path /path/to/mil_model.pth \
    --dataset_dir /path/to/images \
    --mask_dir /path/to/masks

# Skip metrics calculation for faster processing
python main_multi_level_fusion.py --calculate_metrics False

# Custom metrics thresholds
python main_multi_level_fusion.py --metrics_thresholds 0.2 0.4 0.6 0.8

# Add custom output suffix
python main_multi_level_fusion.py --output_suffix high_res_analysis
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--level_weights` | float[3] | [0.5, 0.3, 0.2] | Weights for Level 0, 1, 2 |
| `--model_path` | str | default_path | HierGAT model file path |
| `--mil_model_path` | str | default_path | MIL model file path |
| `--dataset_dir` | str | default_path | Input images directory |
| `--save_dir` | str | default_path | Output visualization directory |
| `--mask_dir` | str | default_path | Ground truth masks directory |
| `--single_image` | str | None | Process only specified image |
| `--output_suffix` | str | None | Add suffix to output directory |
| `--calculate_metrics` | flag | True | Enable performance metrics |
| `--metrics_thresholds` | list | [0.1-0.9] | Evaluation thresholds |

## Weight Configuration Guidelines

### Recommended Weight Combinations

#### Standard Analysis (Default)
```bash
--level_weights 0.5 0.3 0.2
```
- Balanced emphasis on fine details with coarse context
- Good for general-purpose analysis

#### Fine-Detail Focus
```bash
--level_weights 0.7 0.2 0.1
```
- Emphasizes Level 0 fine-grained features
- Best for detailed texture analysis

#### Balanced Multi-Scale
```bash
--level_weights 0.4 0.4 0.2
```
- Equal emphasis on fine and medium scales
- Good for multi-resolution pattern detection

#### Context-Heavy Analysis
```bash
--level_weights 0.2 0.3 0.5
```
- Emphasizes larger spatial contexts
- Useful for global pattern analysis

### Weight Validation
- Weights are automatically normalized to sum to 1.0
- All weights must be non-negative
- Minimum of 3 weights required (one per level)

## Output Structure

### File Organization
```
hiergat_visualization_result/
├── hiergat_comprehensive_image001.png
├── hiergat_comprehensive_image002.png
├── metrics_image001.xlsx
├── metrics_image002.xlsx
├── processing_summary.json
└── overall_metrics_summary.xlsx
```

### Visualization Layout
The comprehensive visualization includes:
- **Row 1**: Original Image, Core Mask Overlay, Annotated Mask
- **Row 2**: Individual Level Attention Maps (0, 1, 2), Multilevel Fusion
- **Row 3**: Core Mask, Fusion Overlay, High Attention Regions, Statistics Panel

### Metrics Excel Files
Each processed image generates a detailed Excel file with:
- **Level_0 Sheet**: Individual metrics for Level 0 attention
- **Level_1 Sheet**: Individual metrics for Level 1 attention  
- **Level_2 Sheet**: Individual metrics for Level 2 attention
- **Multilevel_Fusion Sheet**: Combined fusion metrics
- **Threshold highlighting**: Best performing thresholds highlighted

## Technical Implementation

### HierGAT Processing Pipeline
1. **Model Loading**: HierGAT and MIL model initialization
2. **Level Extraction**: Individual attention extraction per level
3. **Spatial Mapping**: Patch-to-pixel coordinate mapping
4. **Core Masking**: Tissue boundary application
5. **Smoothing Application**: Level-appropriate Gaussian filtering
6. **Fusion Computation**: Weighted combination with custom weights
7. **Metrics Calculation**: Multi-threshold performance evaluation

### Core Extraction System
- **Automatic Detection**: Threshold-based tissue detection
- **Contour Extraction**: Precise boundary identification
- **Mask Application**: Attention refinement to tissue regions
- **Boundary Visualization**: Clear tissue boundary display

### Performance Metrics
- **IoU (Intersection over Union)**: Spatial overlap accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: False positive control
- **Recall**: False negative control (sensitivity)
- **Accuracy**: Overall correctness measure
- **Specificity**: True negative rate

## Expected Results

### Console Output Example
```
Using device: cuda
Level weights: Level0=0.500, Level1=0.300, Level2=0.200
Calculate metrics: True
Metrics thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

==================================================
Processing: image_001.png
==================================================

📊 Calculating performance metrics for image_001.png...
   📈 Calculating metrics for Level 0
   📈 Calculating metrics for Level 1
   📈 Calculating metrics for Level 2
   📈 Calculating metrics for Multilevel Fusion

✅ Successfully processed image_001.png
   Fusion weights used: [0.500 0.300 0.200]
   Max attention: 0.9234
   Coverage: 23.4%
   📊 Level 0: F1=0.731, IoU=0.577
   📊 Level 1: F1=0.692, IoU=0.529
   📊 Level 2: F1=0.658, IoU=0.491
   📊 Multilevel Fusion: F1=0.748, IoU=0.599

✅ Metrics Excel file saved: metrics_image_001.xlsx
```

### Performance Expectations
- **Processing Speed**: 3-6 minutes per image (GPU)
- **Memory Usage**: 3-5 GB GPU memory per image
- **Typical Performance**: 
  - Level 0: F1 scores 0.65-0.75
  - Level 1: F1 scores 0.60-0.70
  - Level 2: F1 scores 0.55-0.65
  - Multilevel Fusion: F1 scores 0.70-0.80

### Output File Sizes
- **Comprehensive PNG**: 5-15 MB per image
- **Metrics Excel**: 50-200 KB per image
- **Summary JSON**: 10-50 KB total
- **Overall Excel**: 100-500 KB total

## Fusion Weight Impact Analysis

### Level 0 Weight Impact
- **High Weight (>0.6)**: Emphasizes fine texture details
- **Medium Weight (0.3-0.6)**: Balanced detail-context trade-off
- **Low Weight (<0.3)**: Reduces noise from fine details

### Level 1 Weight Impact
- **High Weight (>0.5)**: Focus on medium-scale patterns
- **Medium Weight (0.2-0.5)**: Standard multi-scale analysis
- **Low Weight (<0.2)**: Minimal medium-scale contribution

### Level 2 Weight Impact
- **High Weight (>0.4)**: Emphasizes global context
- **Medium Weight (0.1-0.4)**: Contextual support
- **Low Weight (<0.1)**: Minimal global influence

## Dependencies

### Core Requirements
- PyTorch ≥ 1.9.0
- torch-geometric ≥ 2.0.0
- torchvision ≥ 0.10.0

### Visualization Libraries
- matplotlib ≥ 3.4.0
- opencv-python ≥ 4.5.0
- scipy ≥ 1.7.0
- PIL ≥ 8.0.0

### Analysis and Export
- pandas ≥ 1.3.0
- openpyxl ≥ 3.0.0
- numpy ≥ 1.21.0

### Installation
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Common Issues

#### Weight Configuration Errors
```bash
# Error: Wrong number of weights
ValueError: level_weights must contain exactly 3 values

# Solution: Provide exactly 3 weights
python main_multi_level_fusion.py --level_weights 0.5 0.3 0.2
```

#### Memory Issues
```bash
# Error: CUDA out of memory
RuntimeError: CUDA out of memory

# Solutions:
# 1. Use CPU processing
CUDA_VISIBLE_DEVICES="" python main_multi_level_fusion.py

# 2. Process single images
python main_multi_level_fusion.py --single_image image.png
```

#### Performance Issues
```bash
# Error: Low F1 scores across all levels
# Check: Ground truth mask quality and alignment
# Solution: Verify mask preprocessing and image registration
```

### Debug Mode
```bash
# Enable detailed logging
export PYTHONHASHSEED=25
python main_multi_level_fusion.py --calculate_metrics True
```

## Advanced Usage Examples

### Research Analysis Workflow
```bash
# Step 1: Quick overview with default weights
python main_multi_level_fusion.py --output_suffix default_analysis

# Step 2: Fine-detail analysis
python main_multi_level_fusion.py \
    --level_weights 0.7 0.2 0.1 \
    --output_suffix fine_detail_analysis

# Step 3: Context-heavy analysis  
python main_multi_level_fusion.py \
    --level_weights 0.2 0.3 0.5 \
    --output_suffix context_analysis

# Step 4: Compare results across different weight configurations
```

### Single Image Deep Analysis
```bash
python main_multi_level_fusion.py \
    --single_image /path/to/important_sample.png \
    --level_weights 0.4 0.4 0.2 \
    --metrics_thresholds 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 \
    --output_suffix detailed_single_analysis
```

