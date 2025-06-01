# Final Fusion Visualization System

This system provides comprehensive saliency map visualization combining HierGAT, MIL, and CAM-based attention mechanisms for tissue microarray core analysis.

## Overview

The `main_final_fusion.py` script implements a sophisticated fusion system that combines:
- **HierGAT Multilevel Attention**: Hierarchical graph attention across multiple scales
- **MIL (Multiple Instance Learning) Attention**: Patch-level attention from pre-trained MIL models
- **CAM-based Attention**: Class Activation Maps using various gradient-based methods
- **Final Fusion**: Advanced fusion strategies to combine all attention sources

## Key Features

### Multi-Method CAM Support
- **FullGrad** (default): Complete gradient-based attribution
- **GradCAM**: Standard gradient-weighted class activation
- **HiResCAM**: High-resolution class activation mapping
- **ScoreCAM**: Perturbation-based scoring
- **GradCAM++**: Enhanced gradient weighting
- **AblationCAM**: Systematic feature ablation
- **XGradCAM**: Extended gradient attribution
- **EigenCAM**: Eigenvector-based activation

### Fusion Methods
- **Confidence**: Confidence-weighted fusion (default)
- **Optimal**: Mathematically optimal combination
- **Weighted**: Manual weight assignment
- **Adaptive**: Dynamic weight adaptation
- **Multiscale**: Scale-aware fusion

### Performance Metrics
- **F1 Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union
- **Precision**: True positive rate
- **Recall**: Sensitivity measure
- **Multiple Thresholds**: Comprehensive evaluation across thresholds

## Usage

### Basic Usage
```bash
python main_final_fusion.py
```

### Advanced Configuration
```bash
# Use specific CAM method and fusion strategy
python main_final_fusion.py --cam_method gradcam --fusion_method optimal

# Custom model paths
python main_final_fusion.py \
    --model_path /path/to/gat_model.pt \
    --mil_model_path /path/to/mil_model.pth

# Custom directories
python main_final_fusion.py \
    --dataset_dir /path/to/images \
    --save_dir /path/to/results \
    --mask_dir /path/to/masks

# Skip metrics calculation for faster processing
python main_final_fusion.py --calculate_metrics False

# Custom metric thresholds
python main_final_fusion.py --metrics_thresholds 0.2 0.4 0.6 0.8
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--cam_method` | str | fullgrad | CAM visualization method |
| `--fusion_method` | str | confidence | Final fusion strategy |
| `--model_path` | str | default_path | HierGAT model path |
| `--mil_model_path` | str | default_path | MIL model path |
| `--dataset_dir` | str | default_path | Input images directory |
| `--save_dir` | str | default_path | Output directory |
| `--mask_dir` | str | default_path | Ground truth masks directory |
| `--calculate_metrics` | flag | True | Enable metrics calculation |
| `--metrics_thresholds` | list | [0.1-0.9] | Threshold values for metrics |

## Output Files

### Visualizations
- **Final fusion heatmaps**: Combined attention visualizations
- **Individual attention maps**: Separate HierGAT, MIL, and CAM visualizations
- **Overlay visualizations**: Attention maps overlaid on original images
- **Statistical summaries**: Performance metrics and attention statistics

### Metrics Reports
- **JSON Summary**: `processing_summary_{cam_method}_{fusion_method}.json`
- **Excel Metrics**: `overall_fusion_metrics_summary_{cam_method}_{fusion_method}.xlsx`
- **Individual Reports**: Per-image metrics and statistics

### Data Structure
```
visualization_result/
├── final_fusion_heatmaps/
├── individual_attention_maps/
├── overlay_visualizations/
├── processing_summary_*.json
└── overall_fusion_metrics_summary_*.xlsx
```

## Technical Implementation

### Model Integration
- **Consistent Loading**: Unified model loading approach for reproducibility
- **Device Management**: Automatic GPU/CPU detection and allocation
- **Memory Optimization**: Efficient processing for large image datasets

### Attention Fusion Pipeline
1. **HierGAT Processing**: Multi-level graph attention extraction
2. **MIL Attention**: Patch-level attention from pre-trained models
3. **CAM Generation**: Gradient-based class activation mapping
4. **Spatial Alignment**: Resolution matching and coordinate alignment
5. **Fusion Strategy**: Advanced combination using selected method
6. **Core Masking**: Tissue-specific attention refinement

### Quality Assurance
- **Reproducible Results**: Fixed random seeds for consistent outputs
- **Error Handling**: Comprehensive exception handling and logging
- **Validation**: Automatic validation of fusion results
- **Metrics Verification**: Multi-threshold performance evaluation

## Performance Metrics

The system evaluates performance using multiple metrics across various thresholds:

### Calculated Metrics
- **F1 Score**: Overall performance measure
- **IoU (Intersection over Union)**: Spatial overlap accuracy
- **Precision**: False positive control
- **Recall**: False negative control
- **Accuracy**: Overall correctness
- **Specificity**: True negative rate

### Evaluation Strategy
- **Multi-threshold Analysis**: Performance across 0.1-0.9 thresholds
- **Best Performance Selection**: Automatic identification of optimal thresholds
- **Statistical Summaries**: Mean and standard deviation across samples
- **Comparative Analysis**: Side-by-side method comparison

## Dependencies

### Core Requirements
- PyTorch ≥ 1.9.0
- torchvision ≥ 0.10.0
- torch-geometric ≥ 2.0.0
- grad-cam ≥ 1.4.0

### Analysis Libraries
- numpy ≥ 1.21.0
- scipy ≥ 1.7.0
- matplotlib ≥ 3.4.0
- opencv-python ≥ 4.5.0
- pandas ≥ 1.3.0
- openpyxl ≥ 3.0.0

### Installation
```bash
pip install -r requirements.txt
```

## Expected Results

### Console Output
```
Using device: cuda
Using CAM method: FULLGRAD
Using fusion method: CONFIDENCE
Calculate metrics: True
Metrics thresholds: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Processing image_001.png with FULLGRAD
Successfully processed image_001.png
Final fusion weights: [0.425, 0.338, 0.237]
Final fusion heatmap shape: (512, 512)
Attention range: [0.0000, 1.0000]

📊 Performance Metrics Summary:
   📈 Multilevel Fusion: F1=0.742, IoU=0.591, P=0.836, R=0.665
   📈 MIL Attention: F1=0.698, IoU=0.537, P=0.789, R=0.625
   📈 CAM Based: F1=0.721, IoU=0.564, P=0.812, R=0.645
   📈 Final Fusion Heatmap: F1=0.758, IoU=0.610, P=0.845, R=0.685
```

### Performance Expectations
- **Processing Speed**: 2-5 minutes per image (GPU)
- **Memory Usage**: 2-4 GB GPU memory per image
- **Typical Performance**: F1 scores 0.70-0.80 for well-aligned data
- **Output Size**: ~50-100 MB per image (including all visualizations)

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Loading Errors**: Verify model paths and compatibility
3. **Mask Alignment Issues**: Check image and mask dimensions
4. **Low Performance**: Verify ground truth mask quality

### Debug Mode
Enable verbose logging by setting environment variable:
```bash
export PYTHONHASHSEED=25
python main_final_fusion.py --calculate_metrics True
```

