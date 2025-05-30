# GRAPHITE: Graph-based Histopathology Image Analysis Toolkit for Explainable Cancer Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-complete-brightgreen.svg)](./docs)

A comprehensive deep learning pipeline for histopathology image analysis that combines **Multiple Instance Learning (MIL)**, **hierarchical Graph Attention Networks (HierGAT)**, and **explainable AI (XAI)** techniques for cancer diagnosis and visualization.

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
- **OS**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10/11 with WSL2
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.3+ (for GPU acceleration)
- **Docker**: 20.0+ (for containerized deployment)

## 🚀 Installation

### Option 1: Quick Setup (Recommended)
```bash
# Clone the repository
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd graphite-histopathology

# Run automated setup
./quickstart.sh
```

### Option 2: Manual Installation
```bash
# Create virtual environment
python -m venv graphite_env
source graphite_env/bin/activate  # On Windows: graphite_env\Scripts\activate

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

# Or run individual steps:
# Step 1: MIL Classification
python training_step_1/mil_classification/train.py --num_epochs 50

# Step 2: Self-Supervised Learning  
python training_step_2/self_supervised_training/train.py --epochs 100

# Step 3: XAI Visualization
python visualization_step_1/xai_visualization/generate_visualizations.py

# Step 4: Attention Fusion
python visualization_step_2/fusion_visualization/fusion_analysis.py
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
cd training_step_1/mil_classification

# Basic training
python train.py \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.001 \
    --max_patches 100

# Advanced training with color normalization
python train.py \
    --batch_size 16 \
    --num_epochs 150 \
    --use_color_normalization \
    --use_balanced_sampler \
    --early_stopping_patience 15
```

**Key Parameters:**
- `--max_patches`: Maximum patches per patient (default: 100)
- `--use_color_normalization`: Apply Macenko color normalization
- `--use_balanced_sampler`: Balance classes in batches
- `--early_stopping_patience`: Early stopping patience (default: 10)

### Training Step 2: Self-Supervised Learning

Train the hierarchical Graph Attention Network:

```bash
cd training_step_2/self_supervised_training

# Standard self-supervised training (no masks needed)
python train.py \
    --data_dir "../../dataset/training_dataset_step_2/core_image" \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001

# Advanced training with custom parameters
python train.py \
    --epochs 200 \
    --batch_size 32 \
    --hidden_dim 256 \
    --num_heads 8 \
    --dropout 0.1
```

**Key Parameters:**
- `--hidden_dim`: Hidden layer dimension (default: 128)
- `--num_heads`: Number of attention heads (default: 4)
- `--dropout`: Dropout rate (default: 0.1)
- `--patience`: Early stopping patience (default: 20)

### Visualization Step 1: XAI Analysis

Generate explainable AI visualizations:

```bash
cd visualization_step_1/xai_visualization

# Generate comprehensive visualizations
python generate_visualizations.py \
    --model_path "../../training_step_1/mil_classification/output/best_model.pth" \
    --data_dir "../../dataset/visualization_dataset" \
    --output_dir "./output"

# Custom visualization settings
python generate_visualizations.py \
    --visualization_types "attention,gradcam,lime" \
    --save_individual_patches \
    --overlay_opacity 0.4
```

### Visualization Step 2: Attention Fusion

Combine and analyze different attention mechanisms:

```bash
cd visualization_step_2/fusion_visualization

# Multi-modal attention fusion
python fusion_analysis.py \
    --mil_model "../../training_step_1/mil_classification/output/best_model.pth" \
    --ssl_model "../../training_step_2/self_supervised_training/output/best_model.pth" \
    --data_dir "../../dataset/visualization_dataset"
```

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
@article{graphite2024,
    title={GRAPHITE: Graph-based Histopathology Image Analysis for Explainable Cancer Diagnosis},
    author={[Your Name] and [Co-authors]},
    journal={[Journal Name]},
    year={2024},
    doi={[DOI]}
}
```

---

**⭐ Star this repository if you find it helpful!** 