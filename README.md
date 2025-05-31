# GRAPHITE: Graph-Based Interpretable Tissue Examination for Enhanced Explainability in Breast Cancer Histopathology

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-complete-brightgreen.svg)](./docs)

## 📄 About

GRAPHITE is a state-of-the-art deep learning framework specifically designed for breast cancer histopathology analysis. Our approach combines graph-based representations with interpretable AI techniques to provide clinically relevant insights for pathologists. The framework leverages hierarchical graph attention networks and multiple instance learning to analyze tissue microenvironments while maintaining full explainability of predictions through advanced visualization techniques.

**🔗 Explore GRAPHITE:**
- **📖 [Understand Our Code](https://deepwiki.com/raktim-mondol/GRAPHITE)** - Interactive code exploration and documentation
- **🔍 [Code Tutorial](https://code2tutorial.com/tutorial/6f28591e-564f-4ea8-9c7b-e6df90011d14/index.md)** - Step-by-step guide to GRAPHITE implementation
- **📊 [Interactive Dashboard](https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/ad21462b3017a3a2ddb4c0b9d796dc41/b4bf40ab-a0e0-446e-858f-1eb8f5aa4c56/index.html)** - Explore our work through an interactive dashboard

**Key Innovation**: GRAPHITE introduces a novel multi-scale graph representation that captures both local cellular interactions and global tissue architecture, enabling more accurate and interpretable breast cancer diagnosis from histopathology images.

A comprehensive deep learning pipeline for histopathology image analysis that combines **Multiple Instance Learning (MIL)**, **hierarchical Graph Attention Networks (HierGAT)**, and **explainable AI (XAI)** techniques for enhanced breast cancer diagnosis and visualization.

## 🌟 Overview

GRAPHITE provides an end-to-end solution for analyzing histopathology images through a carefully designed 4-step pipeline:

1. **MIL Classification** - Train attention-based models on patch-level data
2. **Self-Supervised Learning** - Learn hierarchical representations using Graph Neural Networks  
3. **XAI Visualization** - Generate explainable attention maps and feature visualizations
4. **Multi-Modal Fusion** - Combine different attention mechanisms for enhanced interpretability

## 🏗️ Pipeline Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   Training      │    │  Visualization  │    │  Visualization  │
│   Step 1        │───▶│   Step 2        │───▶│   Step 1        │───▶│   Step 2        │
│                 │    │                 │    │                 │    │                 │
│ MIL             │    │ Self-Supervised │    │ XAI Analysis    │    │ Attention       │
│ Classification  │    │ Learning        │    │ & Feature Maps  │    │ Fusion          │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        ↓                       ↓                       ↓                       ↓
  Patch-level            Hierarchical           Explainable            Multi-modal
  Attention              Graph Features         Visualizations         Fusion Maps
```

## 📋 Table of Contents

- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Data Structure](#-data-structure)
- [Usage](#-usage)
- [Pipeline Steps](#-pipeline-steps)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [Citation](#-citation)

## ✨ Features

### 🔬 **Advanced Deep Learning Models**
- **Attention-based MIL**: State-of-the-art multiple instance learning with attention mechanisms
- **HierGAT**: Hierarchical Graph Attention Networks for spatial relationship modeling
- **ResNet Backbone**: Pre-trained ResNet models optimized for histopathology
- **Self-Supervised Learning**: Learn rich representations without manual annotations

### 🎯 **Explainable AI Integration**
- **Attention Visualization**: Generate high-quality attention heatmaps
- **Feature Analysis**: Comprehensive feature importance analysis
- **Interactive Visualizations**: Web-based interactive exploration tools
- **Multi-scale Analysis**: Patch, region, and slide-level explanations

### 🔄 **Robust Training Pipeline**
- **Reproducible Results**: Fixed random seeds and deterministic training
- **Early Stopping**: Intelligent training termination to prevent overfitting
- **Learning Rate Scheduling**: Adaptive learning rate adjustments
- **Comprehensive Logging**: Detailed training progress and metrics tracking

### 📊 **Comprehensive Evaluation**
- **Multiple Metrics**: Accuracy, F1-score, AUC, and custom metrics
- **Cross-validation**: Robust performance estimation
- **Statistical Analysis**: Confidence intervals and significance testing
- **Visualization Tools**: Training curves, confusion matrices, and ROC curves

## 🔧 System Requirements

### **Hardware Requirements**
- **GPU**: NVIDIA GPU with 8GB+ VRAM (RTX 3070/Tesla V100 or better)
- **RAM**: 16GB+ system memory (32GB recommended for large datasets)
- **Storage**: 50GB+ free space for datasets and models
- **CPU**: Multi-core processor (8+ cores recommended)

### **Software Requirements**
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows with WSL2/Docker
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.3+ (for GPU acceleration)
- **Docker**: 20.0+ (for containerized deployment)

**Note**: All instructions in this guide use Linux/bash syntax. For Windows users, we recommend using WSL2, Git Bash, or Docker for the best experience.

## 🚀 Installation

### Option 1: Quick Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE

# Run automated setup
./quickstart.sh

# Alternative: Manual setup
python training_step_1/run_training.py --install_deps
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv graphite_env
source graphite_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (if needed)
pip install torch-geometric
```

### Option 3: Docker Setup
```bash
# Build and run with Docker Compose
docker-compose up --build

# Or use pre-built image
docker pull your-registry/graphite:latest
docker run --gpus all -v $(pwd):/workspace graphite:latest
```

For detailed installation instructions, see [SETUP.md](SETUP.md).

## ⚡ Quick Start

### 1. **Prepare Your Data**
```bash
# Organize your data according to the expected structure
dataset/
├── training_dataset_step_1/tma_core/
│   ├── 10025/  # Cancer patient patches
│   ├── 20001/  # Normal patient patches
│   └── ...
├── training_dataset_step_2/
│   └── core_image/  # Core images for self-supervised learning (no masks needed)
└── visualization_dataset/
    ├── core_image/  # Images for visualization
    └── mask/        # Ground truth masks for evaluation
```

### 2. **Run the Complete Pipeline**
```bash
# Automated pipeline execution
./quickstart.sh

# Or run individual steps manually
# Step 1: MIL Classification
python training_step_1/run_training.py --epochs 50

# Step 2: Self-Supervised Learning  
python training_step_2/self_supervised_training/train.py --epochs 100

# Step 3: XAI Visualization
python visualization_step_1/xai_visualization/main.py

# Step 4: Attention Fusion
python visualization_step_2/fusion_visualization/main_final_fusion.py
```

### 3. **View Results**
```bash
# Training logs and models
ls training_step_1/mil_classification/output/
ls training_step_2/self_supervised_training/output/

# Generated visualizations
ls visualization_step_1/xai_visualization/output/
ls visualization_step_2/fusion_visualization/output/
```

## 📁 Data Structure

The pipeline expects data organized in three main directories:

```
dataset/
├── training_dataset_step_1/         # MIL Classification Training
│   └── tma_core/
│       ├── 10025/ (cancer)          # Patient folders with image patches
│       ├── 10026/ (cancer)
│       ├── 20001/ (normal)
│       └── ...
├── training_dataset_step_2/         # Self-Supervised Learning  
│   └── core_image/                  # Core images only (no masks needed)
│       ├── image_001.png
│       └── ...
├── visualization_dataset/           # Visualization Analysis
│   ├── core_image/                  # Images for analysis
│   └── mask/                        # Ground truth masks for evaluation
└── label files (cancer.txt, normal.txt)
```

**Important Notes:**
- **Step 1**: Requires patient folders with patches + label files
- **Step 2**: Requires only core images (masks not needed for self-supervised learning)
- **Visualization**: Requires both images and masks for evaluation

For detailed specifications, see [DATA_STRUCTURE.md](DATA_STRUCTURE.md).

## 📖 Usage

### Training Step 1: MIL Classification

Train the Multiple Instance Learning model for patch-level classification:

```bash
cd training_step_1

# Basic training (recommended - uses wrapper script)
python run_training.py \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 0.001 \
    --max_patches 100

# Advanced training with color normalization
python run_training.py \
    --batch_size 16 \
    --epochs 150 \
    --color_norm \
    --balanced_sampler \
    --patience 15

# Quick test mode (2 epochs, fast)
python run_training.py --quick_test

# Direct training (advanced users)
cd mil_classification
python train.py \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --max_patches 100 \
    --use_color_normalization \
    --use_balanced_sampler \
    --early_stopping_patience 15
```

**Key Parameters (wrapper script):**
- `--max_patches`: Maximum patches per patient (default: 100)
- `--epochs`: Number of training epochs (default: 100)
- `--color_norm`: Apply Macenko color normalization
- `--balanced_sampler`: Balance classes in batches
- `--patience`: Early stopping patience (default: 10)
- `--quick_test`: Run quick test (2 epochs, 50 patches)
- `--data_dir`: Root directory containing patient folders
- `--metrics`: Metric to monitor for early stopping (default: 'auc')

### Training Step 2: Self-Supervised Learning

Train the hierarchical Graph Attention Network:

```bash
cd training_step_2/self_supervised_training

# Standard self-supervised training (no masks needed)
python train.py \
    --data_dir "../../dataset/training_dataset_step_2/core_image" \
    --epochs 100 \
    --batch_size 4 \
    --lr 0.001

# Advanced training with custom parameters
python train.py \
    --data_dir "../../dataset/training_dataset_step_2/core_image" \
    --epochs 150 \
    --batch_size 8 \
    --lr 0.0005 \
    --weight_decay 1e-4 \
    --hidden_dim 256 \
    --num_heads 8 \
    --dropout 0.1 \
    --temperature 0.05 \
    --alpha 0.6 \
    --output_dir "./output/my_experiment"

# Training with specific configuration file
python train.py \
    --config config/config.yaml \
    --data_dir "../../dataset/training_dataset_step_2/core_image"
```

**Key Parameters:**
- `--data_dir`: Path to directory containing training images (required)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--hidden_dim`: Hidden dimension for GAT layers (default: 128)
- `--num_heads`: Number of attention heads (default: 4)
- `--num_gat_layers`: Number of GAT layers (default: 3)
- `--dropout`: Dropout rate (default: 0.1)
- `--input_dim`: Input feature dimension (default: 128)
- `--temperature`: Temperature for InfoMax loss (default: 0.07)
- `--alpha`: Weight for InfoMax loss (default: 0.5)
- `--beta`: Weight for Scale-wise loss (default: 0.5)
- `--tau`: Temperature for Scale-wise loss (default: 0.1)
- `--patience`: Early stopping patience (default: 10)
- `--num_workers`: Number of data loading workers (default: 4)
- `--config`: Path to configuration file (default: 'config/config.yaml')
- `--output_dir`: Directory to save models and plots (default: 'output/hiergat_ssl')
- `--resume`: Path to checkpoint to resume training from
- `--seed`: Random seed for reproducibility (default: 78)
- `--device`: Device to use for training ('auto', 'cuda', 'cpu') (default: 'auto')
- `--verbose`: Enable verbose logging

### Visualization Step 1: XAI Analysis

Generate explainable AI visualizations using multiple methods:

```bash
cd visualization_step_1/xai_visualization

# Basic usage example (replace 'gradcam' with any method from the list below)
python main.py \
    --method gradcam \
    --wsi_folder "../../dataset/visualization_dataset/core_image" \
    --mask_folder "../../dataset/visualization_dataset/mask" \
    --output_folder "./output" \
    --model_path "../../training_step_1/mil_classification/output/best_model.pth"

# Advanced usage with custom parameters
python main.py \
    --method shap_deep \
    --wsi_folder "../../dataset/visualization_dataset/core_image" \
    --mask_folder "../../dataset/visualization_dataset/mask" \
    --output_folder "./output" \
    --patch_size 224 \
    --stride 224 \
    --target_class 1 \
    --config "./config/default.yaml" \
    --verbose
```

**Available Visualization Methods:**

*CAM-based Methods:*
- `gradcam`: Gradient-weighted Class Activation Mapping
- `hirescam`: High Resolution Class Activation Mapping
- `scorecam`: Score-weighted Class Activation Mapping
- `gradcampp`: GradCAM++ (improved version)
- `ablationcam`: Ablation-based CAM
- `xgradcam`: Extended GradCAM
- `eigencam`: Eigen-based CAM
- `fullgrad`: Full Gradient decomposition

*Model-agnostic Methods:*
- `shap_deep`: SHAP with Deep Explainer
- `shap_gradient`: SHAP with Gradient Explainer
- `lime`: LIME (Local Interpretable Model-agnostic Explanations)

*MIL-specific Methods:*
- `attention`: MIL attention-based visualization

**Key Parameters:**
- `--method`: Visualization method to use (required)
- `--wsi_folder`: Path to folder containing WSI images (required)
- `--mask_folder`: Path to folder containing ground truth masks (required)
- `--output_folder`: Path to output folder for results (required)
- `--model_path`: Path to trained model (default: './models/best_fine_tuned_model_for_resnet18_cancervsnormal_v4.pth')
- `--config`: Path to configuration file (default: './config/default.yaml')
- `--patch_size`: Size of patches to extract from WSI (default: 224)
- `--stride`: Stride between patches (default: 224)
- `--target_class`: Target class for visualization (1=cancer, 0=normal) (default: 1)
- `--device`: Device to use ('auto', 'cpu', 'cuda') (default: 'auto')
- `--seed`: Random seed for reproducibility (default: 78)
- `--verbose`: Enable verbose logging

### Visualization Step 2: Attention Fusion

Combine and analyze different attention mechanisms using two specialized fusion approaches:

```bash
cd visualization_step_2/fusion_visualization

# Multi-modal attention fusion (combines HierGAT, MIL, and CAM methods)
python main_final_fusion.py \
    --model_path "../../training_step_2/self_supervised_training/output/best_model.pt" \
    --mil_model_path "../../training_step_1/mil_classification/output/best_model.pth" \
    --dataset_dir "../../dataset/training_dataset_step_1/tma_core" \
    --save_dir "./output/visualization_results" \
    --mask_dir "../../dataset/training_dataset_step_2/mask"

# HierGAT multi-level fusion (hierarchical attention analysis)
python main_multi_level_fusion.py \
    --model_path "../../training_step_2/self_supervised_training/output/best_model.pt" \
    --dataset_dir "../../dataset/training_dataset_step_1/tma_core" \
    --level_weights 0.5 0.3 0.2 \
    --save_dir "./output/hiergat_visualization_results"

# Advanced multi-modal fusion with custom CAM method
python main_final_fusion.py \
    --cam_method gradcam \
    --fusion_method optimal \
    --model_path "../../training_step_2/self_supervised_training/output/best_model.pt" \
    --mil_model_path "../../training_step_1/mil_classification/output/best_model.pth" \
    --dataset_dir "../../dataset/training_dataset_step_1/tma_core" \
    --metrics_thresholds 0.2 0.4 0.6 0.8

# Custom HierGAT level analysis with equal weights
python main_multi_level_fusion.py \
    --level_weights 0.33 0.33 0.34 \
    --single_image "/path/to/specific_image.png" \
    --calculate_metrics True
```

**Available Scripts:**

*Multi-Modal Fusion (`main_final_fusion.py`):*
- Combines HierGAT, MIL, and CAM-based attention mechanisms
- Supports 8 different CAM methods: `fullgrad`, `gradcam`, `hirescam`, `scorecam`, `gradcampp`, `ablationcam`, `xgradcam`, `eigencam`
- Offers 5 fusion strategies: `confidence`, `optimal`, `weighted`, `adaptive`, `multiscale`
- Provides comprehensive performance metrics and visualization overlays

*HierGAT Multi-Level Fusion (`main_multi_level_fusion.py`):*
- Focuses on hierarchical attention analysis across 3 levels
- Allows custom weight configuration for Level 0, Level 1, and Level 2
- Provides individual level visualizations and multilevel fusion
- Includes core mask integration and detailed Excel metrics export

**Key Parameters:**

*main_final_fusion.py:*
- `--cam_method`: CAM visualization method (default: 'fullgrad')
- `--fusion_method`: Final fusion strategy (default: 'confidence')
- `--model_path`: HierGAT model path (required)
- `--mil_model_path`: MIL model path (required)
- `--dataset_dir`: Input images directory (required)
- `--save_dir`: Output directory (default: './output/visualization_results')
- `--mask_dir`: Ground truth masks directory (required)
- `--calculate_metrics`: Enable metrics calculation (default: True)
- `--metrics_thresholds`: Threshold values for evaluation (default: 0.1-0.9)

*main_multi_level_fusion.py:*
- `--level_weights`: Weights for Level 0, 1, 2 (default: [0.5, 0.3, 0.2])
- `--model_path`: HierGAT model path (required)
- `--dataset_dir`: Input images directory (required)
- `--save_dir`: Output directory (default: './output/hiergat_visualization_results')
- `--single_image`: Process only specified image (optional)
- `--output_suffix`: Add suffix to output directory (optional)
- `--calculate_metrics`: Enable performance metrics (default: True)
- `--metrics_thresholds`: Evaluation thresholds (default: 0.1-0.9)

## 🔬 Pipeline Steps

### Step 1: Multiple Instance Learning (MIL)
- **Input**: Patient folders with histopathology patches
- **Model**: Attention-based MIL with ResNet backbone
- **Output**: Patient-level cancer classification
- **Key Features**: 
  - Patch-level attention weights
  - Color normalization support
  - Balanced sampling options

### Step 2: Self-Supervised Learning
- **Input**: Core tissue images (no annotations needed)
- **Model**: Hierarchical Graph Attention Network (HierGAT)
- **Output**: Rich feature representations
- **Key Features**: 
  - Graph-based spatial modeling
  - Hierarchical attention mechanisms
  - Unsupervised representation learning

### Step 3: XAI Visualization  
- **Input**: Trained MIL model + visualization dataset
- **Methods**: Attention maps, GradCAM, LIME, integrated gradients
- **Output**: Explainable visualizations and feature importance
- **Key Features**: 
  - Multiple XAI techniques
  - Interactive visualization tools
  - Quantitative explainability metrics

### Step 4: Multi-Modal Fusion
- **Input**: MIL and SSL models + visualization data
- **Process**: Attention fusion and multi-scale analysis
- **Output**: Enhanced interpretability and diagnostic insights
- **Key Features**: 
  - Cross-modal attention fusion
  - Multi-scale feature analysis
  - Comprehensive diagnostic reports

## 📊 Results

### Expected Performance Benchmarks

**MIL Classification (Step 1):**
- Accuracy: 85-92%
- AUC: 0.90-0.95
- F1-Score: 0.82-0.89
- Training Time: 2-4 hours (GPU)

**Self-Supervised Learning (Step 2):**
- Convergence: 50-100 epochs
- Feature Quality: High-dimensional representations
- Training Time: 4-8 hours (GPU)

**Visualization Quality (Steps 3-4):**
- Attention Map Resolution: High-quality heatmaps
- Explainability Metrics: ROAR, KAR scores
- Processing Time: 10-30 minutes per dataset

### Output Files Structure
```
outputs/
├── training_step_1/
│   ├── best_model.pth
│   ├── training_history.png
│   └── metrics.json
├── training_step_2/
│   ├── ssl_model.pth
│   ├── embeddings.npy
│   └── training_logs.txt
├── visualization_step_1/
│   ├── attention_maps/
│   ├── gradcam_visualizations/
│   └── analysis_report.html
└── visualization_step_2/
    ├── fusion_maps/
    ├── multi_scale_analysis/
    └── final_report.pdf
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### **GPU Memory Issues**
```bash
# Reduce batch size
python train.py --batch_size 4

# Use gradient accumulation
python train.py --accumulate_grad_batches 4
```

#### **CUDA Compatibility**
```bash
# Check CUDA version
nvidia-smi

# Install compatible PyTorch
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

#### **Data Loading Errors**
```bash
# Validate data structure
python -c "from src.utils.data_validation import validate_dataset; validate_dataset('dataset/')"

# Check file permissions
chmod -R 755 dataset/
```

#### **Memory Optimization**
- Reduce `max_patches` parameter for MIL training
- Use `--num_workers 0` if experiencing multiprocessing issues
- Enable `--pin_memory False` for systems with limited RAM

#### **Performance Issues**
- Ensure CUDA is properly installed and detected
- Use SSD storage for faster data loading
- Monitor GPU utilization with `nvidia-smi`
- Consider mixed precision training for larger models

### Getting Help

1. **Check Logs**: Review training logs in `output/` directories
2. **Validate Data**: Use provided validation scripts
3. **System Check**: Run `quickstart.sh` option 10 for system diagnostics
4. **Documentation**: Refer to detailed guides in `docs/`
5. **Issues**: Create a GitHub issue with error logs and system info

## 📚 Additional Documentation

- **[SETUP.md](SETUP.md)** - Detailed installation instructions
- **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)** - Comprehensive data organization guide
- **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** - Steps to reproduce results
- **[docs/](docs/)** - API documentation and advanced guides

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TIA Toolbox** for histopathology image processing tools
- **PyTorch Geometric** for graph neural network implementations  
- **Captum** for explainable AI techniques
- **Medical imaging community** for datasets and evaluation benchmarks

## 📞 Contact

For questions and support:
- **Create an issue** on GitHub for bug reports and feature requests
- **Check documentation** in the `docs/` directory for detailed guides
- **Review examples** in the `examples/` directory for usage patterns

---

## 📈 Citation

If you use GRAPHITE in your research, please cite:

```bibtex
@misc{mondol2025graphitegraphbasedinterpretabletissue,
      title={GRAPHITE: Graph-Based Interpretable Tissue Examination for Enhanced Explainability in Breast Cancer Histopathology}, 
      author={Raktim Kumar Mondol and Ewan K. A. Millar and Peter H. Graham and Lois Browne and Arcot Sowmya and Erik Meijering},
      year={2025},
      eprint={2501.04206},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.04206}, 
}
```

---

**⭐ Star this repository if you find it helpful!** 