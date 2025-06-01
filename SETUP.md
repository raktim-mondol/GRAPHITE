# GRAPHITE Setup Guide

This comprehensive guide will help you set up the GRAPHITE histopathology analysis pipeline on your system.

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation Methods](#-installation-methods)
- [Data Setup](#-data-setup)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Verification](#-verification)

## 🔧 System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4-core processor (Intel i5 or AMD equivalent)
- **RAM**: 8GB system memory
- **Storage**: 20GB free space
- **GPU**: Optional but highly recommended for training

#### Recommended Requirements
- **CPU**: 8+ core processor (Intel i7/i9 or AMD Ryzen 7/9)
- **RAM**: 16GB+ system memory (32GB for large datasets)
- **Storage**: 50GB+ free space (SSD recommended)
- **GPU**: NVIDIA Tesla V100 32GB (used/recommended)

### Software Requirements

#### Operating System
- **Linux**: Ubuntu 18.04+ (recommended), CentOS 7+, or other modern distributions
- **Windows**: Windows 10/11 with WSL2 (Windows Subsystem for Linux)

#### Core Software
- **Python**: 3.9.2 or higher (3.9.2 recommended)
- **CUDA**: 11.7 (for GPU acceleration)
- **Git**: For cloning the repository

## 🚀 Installation Methods

### Method 1: Quick Setup (Recommended)

The quickest way to get started is using the automated setup script:

```bash
# Clone the repository
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE

# Make the script executable
chmod +x quickstart.sh

# Run the interactive setup
./quickstart.sh
```

Follow the menu options:
1. Check system requirements
2. Setup environment and install dependencies
3. Setup data directories
4. Generate demo data (optional)
5. Validate installation

### Method 2: Manual Installation with Virtual Environment

#### Step 1: Clone Repository
```bash
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE
```

#### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv graphite_env

# Activate virtual environment
# On Linux:
source graphite_env/bin/activate
# On Windows:
graphite_env\Scripts\activate
```

#### Step 3: Upgrade pip
```bash
pip install --upgrade pip
```

#### Step 4: Install PyTorch
Choose the appropriate command based on your system:

**With CUDA support (recommended for GPU):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

**CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 6: Install PyTorch Geometric
```bash
pip install torch-geometric
```

### Method 3: Conda Installation

#### Step 1: Create Conda Environment
```bash
# Create environment with Python 3.9
conda create -n graphite python=3.9 -y
conda activate graphite
```

#### Step 2: Install PyTorch and PyTorch Geometric
```bash
# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg
```

#### Step 3: Install Other Dependencies
```bash
pip install -r requirements.txt
```

### Method 4: Docker Installation

#### Option A: Using Docker Compose (Recommended)
```bash
# Clone repository
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE

# Build and run with GPU support
docker-compose up --build

# Access Jupyter notebook
# http://localhost:8888
```

#### Option B: Manual Docker Build
```bash
# Build the Docker image
docker build -t graphite:latest .

# Run with GPU support
docker run --gpus all -v $(pwd):/workspace -p 8888:8888 graphite:latest

# Run without GPU
docker run -v $(pwd):/workspace -p 8888:8888 graphite:latest
```

## 📁 Data Setup

### Data Structure Overview

The GRAPHITE pipeline expects data organized in three main directories:

```
dataset/
├── training_dataset_step_1/         # MIL Classification
│   └── tma_core/
│       ├── 10025/ (cancer)
│       ├── 20001/ (normal)
│       └── ...
├── training_dataset_step_2/         # Self-Supervised Learning
│   └── core_image/                  # Core images only (no masks needed)
│       ├── image_001.png
│       └── ...
├── visualization_dataset/           # Visualization
│   ├── core_image/
│   └── mask/                        # Ground truth masks for evaluation
├── cancer.txt                       # Cancer patient labels
└── normal.txt                       # Normal patient labels
```

### Step 1: Create Directory Structure

#### Automated Setup
```bash
./quickstart.sh
# Choose option 3: "Setup data directories"
```

#### Manual Setup
```bash
# Create main directories
mkdir -p dataset/{training_dataset_step_1/tma_core,training_dataset_step_2/core_image,visualization_dataset/{core_image,mask}}

# Create output directories
mkdir -p {training_step_1/mil_classification,training_step_2/self_supervised_training,visualization_step_1/xai_visualization,visualization_step_2/fusion_visualization}/output
```

### Step 2: Prepare Label Files

Create label files for MIL classification:

**dataset/cancer.txt:**
```csv
patient_id
10025
10026
10027
10028
10029
10030
10031
10032
10033
10034
22021
22107
```

**dataset/normal.txt:**
```csv
patient_id
20001
20002
20003
20004
20005
20006
20007
20008
```

### Step 3: Organize Your Data

#### For Training Step 1 (MIL Classification)
Place histopathology patches in patient-specific folders:
```bash
dataset/training_dataset_step_1/tma_core/
├── 10025/          # Cancer patient
│   ├── patch_001.png
│   ├── patch_002.png
│   └── ...
├── 20001/          # Normal patient
│   ├── patch_001.png
│   └── ...
```

#### For Training Step 2 (Self-Supervised Learning)
**Important**: Only core images are needed - masks are NOT required for self-supervised learning:
```bash
dataset/training_dataset_step_2/core_image/
├── image_001.png
├── image_002.png
└── ...
```

#### For Visualization (Steps 3-4)
Place images and corresponding masks for evaluation:
```bash
dataset/visualization_dataset/
├── core_image/
│   ├── vis_001.png
│   └── ...
└── mask/           # Required for evaluation
    ├── vis_mask_001.png
    └── ...
```

### Data Format Requirements

#### Image Specifications
- **Formats**: PNG, JPG, JPEG, TIFF, TIF
- **Color**: RGB (3-channel)
- **Size**: Minimum 224x224 pixels (larger recommended)
- **Quality**: High-resolution, well-focused images

#### Patch Requirements (Step 1)
- **Size**: Typically 224x224 to 512x512 pixels
- **Count**: 50-200 patches per patient
- **Quality**: Clear tissue regions, minimal artifacts

#### Core Image Requirements (Step 2)
- **Size**: 1024x1024 to 2048x2048 pixels
- **Content**: Complete tissue cores
- **Quality**: High-resolution, suitable for self-supervised learning
- **No masks needed**: Self-supervised learning works without ground truth

#### Mask Requirements (Visualization only)
- **Format**: PNG (binary masks preferred)
- **Size**: Same dimensions as corresponding images
- **Values**: Binary (0/255) or multi-class
- **Purpose**: Evaluation of attention quality

## ⚙️ Configuration

### Environment Variables

Set up environment variables for consistent paths:

```bash
# Add to your ~/.bashrc or ~/.zshrc
export GRAPHITE_ROOT="/path/to/GRAPHITE"
export GRAPHITE_DATA="$GRAPHITE_ROOT/dataset"
export CUDA_VISIBLE_DEVICES=0  # Set GPU device
```

### Model Configuration

Default configurations are provided, but you can customize:

#### MIL Configuration
```bash
# training_step_1/mil_classification/train.py parameters
--batch_size 8
--num_epochs 100
--learning_rate 0.001
--max_patches 100
```

#### Self-Supervised Learning Configuration
```bash
# training_step_2/self_supervised_training/train.py parameters
--epochs 100
--batch_size 16
--learning_rate 0.0001
--hidden_dim 128
```

## 🔧 Platform-Specific Setup

### Ubuntu/Debian Linux

#### Install System Dependencies
```bash
sudo apt update
sudo apt install -y python3-dev python3-pip python3-venv git build-essential

# For OpenCV and image processing
sudo apt install -y libopencv-dev python3-opencv libglib2.0-0 libsm6 libxext6 libxrender-dev

# For PyTorch Geometric dependencies
sudo apt install -y libffi-dev libssl-dev
```

#### CUDA Installation (Ubuntu)
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA
sudo apt-get install cuda
```

### Windows (WSL2)

#### Enable WSL2
```powershell
# Run in PowerShell as Administrator
wsl --install
```

#### Install Ubuntu in WSL2
```bash
# After restart, install Ubuntu from Microsoft Store
# Then follow Ubuntu setup instructions above
```

#### GPU Support in WSL2
```bash
# Install NVIDIA CUDA drivers for WSL2
# Download from: https://developer.nvidia.com/cuda/wsl
```

## 🐛 Troubleshooting

### Common Installation Issues

#### PyTorch Geometric Installation Fails

**Problem**: Error installing PyTorch Geometric
```bash
ERROR: Failed building wheel for torch-geometric
```

**Solutions**:

1. **Use conda instead of pip**:
```bash
conda install pyg -c pyg
```

2. **Install dependencies manually**:
```bash
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0+cu117.html
pip install torch-geometric
```

3. **Build from source**:
```bash
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

#### CUDA Version Mismatch

**Problem**: CUDA version conflicts
```bash
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solutions**:

1. **Check CUDA version**:
```bash
nvidia-smi
nvcc --version
```

2. **Install matching PyTorch**:
```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

#### Memory Issues

**Problem**: Out of memory errors during training
```bash
RuntimeError: CUDA out of memory
```

**Solutions**:

1. **Reduce batch size**:
```bash
python train.py --batch_size 4
```

2. **Reduce max patches**:
```bash
python train.py --max_patches 50
```

3. **Use CPU training**:
```bash
export CUDA_VISIBLE_DEVICES=""
```

#### Permission Errors

**Problem**: Permission denied when creating directories
```bash
PermissionError: [Errno 13] Permission denied
```

**Solutions**:

1. **Fix permissions**:
```bash
chmod -R 755 dataset/
```

2. **Change ownership**:
```bash
sudo chown -R $USER:$USER dataset/
```

### Data-Related Issues

#### Missing Data Structure

**Problem**: Data directory not found
```bash
FileNotFoundError: Dataset directory does not exist
```

**Solution**: Create proper directory structure:
```bash
./quickstart.sh
# Choose option 3: "Setup data directories"
```

#### Label File Format Errors

**Problem**: Error reading label files
```bash
pandas.errors.EmptyDataError: No columns to parse from file
```

**Solution**: Ensure label files have correct format:
```csv
patient_id
10025
10026
```

#### Image Loading Errors

**Problem**: Cannot load images
```bash
PIL.UnidentifiedImageError: cannot identify image file
```

**Solutions**:

1. **Check file formats**: Ensure images are PNG, JPG, JPEG, TIFF, or TIF
2. **Verify file integrity**: Check if files are corrupted
3. **Check permissions**: Ensure files are readable

### Performance Issues

#### Slow Data Loading

**Solutions**:
```bash
# Increase number of workers
python train.py --num_workers 4

# Use SSD storage for datasets
# Enable pin_memory for faster GPU transfer
python train.py --pin_memory True
```

#### Training Too Slow

**Solutions**:
```bash
# Enable mixed precision training
python train.py --use_amp

# Reduce image resolution
python train.py --image_size 256

# Use gradient accumulation
python train.py --accumulate_grad_batches 4
```

## ✅ Verification

### Installation Verification

#### Quick Test
```bash
./quickstart.sh
# Choose option 5: "Validate installation"
```

#### Manual Verification
```bash
python3 -c "
import torch
import torchvision
import torch_geometric
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
print('✓ All packages imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Data Structure Verification

#### Automated Check
```bash
# Validate data structure
python -c "
import os
required_dirs = [
    'dataset/training_dataset_step_1/tma_core',
    'dataset/training_dataset_step_2/core_image',  # No mask directory needed
    'dataset/visualization_dataset/core_image',
    'dataset/visualization_dataset/mask'
]
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f'✓ {dir_path}')
    else:
        print(f'✗ {dir_path} (missing)')
"
```

### Demo Run

#### Generate and Test with Demo Data
```bash
# Generate demo data
./quickstart.sh
# Choose option 4: "Generate demo data"

# Run quick training test
cd training_step_1/mil_classification
python train.py --num_epochs 1 --batch_size 2
```

## 📚 Next Steps

After successful setup:

1. **Review Documentation**: Read [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for detailed data requirements
2. **Prepare Your Data**: Organize your histopathology data according to the expected structure
3. **Run Pipeline**: Follow the usage instructions in [README.md](README.md)
4. **Reproducibility**: Check [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact reproduction steps

## 🆘 Getting Help

If you encounter issues not covered in this guide:

1. **Check Logs**: Review error messages and log files
2. **Validate Data**: Ensure data structure matches requirements
3. **Update Dependencies**: Make sure all packages are up to date
4. **Create Issue**: Report bugs on GitHub with system info and error logs
5. **Community**: Join discussions and ask questions

## 📞 Support

For additional support:
- **Documentation**: Check the `docs/` directory
- **Examples**: Review example scripts and notebooks
- **GitHub Issues**: Create detailed bug reports
- **System Info**: Include output from `./quickstart.sh` option 10

---

**Note**: This setup guide assumes you have appropriate permissions and ethical approval for using medical imaging data. 