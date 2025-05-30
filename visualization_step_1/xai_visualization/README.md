# XAI Visualization Tool for Histopathology Images

A comprehensive toolkit for explaining AI model predictions on histopathology images using various explainability methods.

## 🔬 Overview

This tool provides multiple explainability methods for analyzing histopathology images:

### 1. CAM-based Methods
- **GradCAM**: Gradient-weighted Class Activation Mapping
- **HiResCAM**: High Resolution Class Activation Mapping  
- **ScoreCAM**: Score-weighted Class Activation Mapping
- **GradCAM++**: Improved version of GradCAM
- **AblationCAM**: Ablation-based CAM
- **XGradCAM**: Extended GradCAM
- **EigenCAM**: Eigen-based CAM
- **FullGrad**: Full Gradient decomposition

### 2. Model-agnostic Methods
- **SHAP Deep Explainer**: Deep learning-specific SHAP explanations
- **SHAP Gradient Explainer**: Gradient-based SHAP explanations
- **LIME**: Local Interpretable Model-agnostic Explanations

### 3. MIL Attention-based
- **Attention Visualization**: Multiple Instance Learning attention weights

## 🚀 Quick Start

### Setup

**Note:** For initial project setup and dependencies, please refer to the main project README.md in the root directory.

This module requires specific XAI dependencies. Ensure you have the following installed:

```bash
# Additional XAI-specific requirements
pip install grad-cam
pip install shap
pip install lime
pip install captum
```

### Basic Usage

```bash
# GradCAM visualization
python main.py --method gradcam \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results \
    --model_path ./models/best_model.pth

# SHAP visualization
python main.py --method shap_deep \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results

# LIME visualization
python main.py --method lime \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results

# Attention visualization
python main.py --method attention \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results
```

## 📁 Project Structure

```
xai_visualization/
├── main.py                    # Main CLI interface
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── config/
│   └── default.yaml          # Default configuration
├── src/
│   ├── __init__.py
│   │   ├── slide_reader.py   # WSI reading utilities
│   │   └── color_normalizer.py # Color normalization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mil_model.py      # MIL model definition
│   │   └── model_loader.py   # Model loading utilities
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration management
│   │   ├── logger.py         # Logging utilities
│   │   └── metrics.py        # Evaluation metrics
│   └── visualizers/
│       ├── __init__.py
│       ├── base_visualizer.py    # Base class for all visualizers
│       ├── cam_visualizer.py     # CAM-based methods
│       ├── shap_visualizer.py    # SHAP-based methods
│       ├── lime_visualizer.py    # LIME-based methods
│       └── attention_visualizer.py # Attention-based methods
├── examples/
├── docs/
└── tests/
```

## 🛠️ Configuration

The tool uses YAML configuration files. You can customize parameters in `config/default.yaml`:

```yaml
# Model parameters
feat_dim: 512
proj_dim: 128
num_classes: 2

# Image processing
patch_size: 224
stride: 224
target_class: 1

# Method-specific parameters
background_samples: 10  # For SHAP
lime_num_samples: 100   # For LIME
```

## 📊 Available Methods

### CAM-based Methods
All CAM methods support the following parameters:
- `--patch_size`: Size of patches (default: 224)
- `--stride`: Stride between patches (default: 224)
- `--target_class`: Target class for visualization (default: 1)

### SHAP Methods
- `shap_deep`: Uses DeepExplainer for neural networks
- `shap_gradient`: Uses GradientExplainer for gradient-based explanations

### LIME
- Provides local explanations using image segmentation
- Configurable number of samples and features

### Attention
- Visualizes attention weights from MIL models
- Requires models with attention mechanisms

## 📈 Output

The tool generates:
1. **Heatmap visualizations**: PNG files showing explanations overlaid on WSI
2. **Evaluation metrics**: CSV files with quantitative results
3. **Summary statistics**: Overall performance metrics

### Metrics Calculated
- Precision
- Recall  
- F1-score
- IoU (Intersection over Union)
- Confusion matrix elements (TP, FP, FN, TN)

## 🔧 Advanced Usage

### Custom Configuration
```bash
python main.py --method gradcam \
    --config ./config/custom.yaml \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results
```

### Batch Processing
The tool automatically processes all PNG files in the WSI folder and matches them with corresponding masks.

### GPU Support
```bash
python main.py --method gradcam \
    --device cuda \
    --wsi_folder ./data/wsi \
    --mask_folder ./data/masks \
    --output_folder ./results
```

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for complete dependency list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

