# GRAPHITE: Graph-based Self-Supervised Learning for Histopathology

A hierarchical Graph Attention Network (GAT) for multi-scale self-supervised learning on histopathology images. This implementation combines InfoMax loss with scale-wise loss to learn hierarchical representations across different magnification levels.

## 🔬 Overview

GRAPHITE implements a novel self-supervised learning approach that:

- **Hierarchical Learning**: Operates on 3 magnification levels (Level 0, 1, 2) representing different scales
- **Graph-based Architecture**: Uses Graph Attention Networks to model spatial relationships between patches
- **Dual Loss Function**: Combines InfoMax loss (local-global mutual information) with Scale-wise loss (cross-scale consistency)
- **Self-Supervised**: No manual annotations required - learns representations from tissue structure

### Key Features

- ✅ **Multi-scale Analysis**: Processes images at 3 hierarchical levels
- ✅ **Attention Mechanisms**: Hierarchical attention for scale-wise feature aggregation  
- ✅ **Reproducible**: Fixed random seeds and deterministic operations
- ✅ **CLI Support**: Easy-to-use command line interface
- ✅ **Training Progress**: Loss tracking and progress visualization
- ✅ **Modular Design**: Clean, well-organized codebase

## 📋 Requirements

### System Requirements
- Python 3.9.2 or higher
- PyTorch 2.0.0
- CUDA 11.7
- NVIDIA Tesla V100 32GB (used/recommended) or equivalent CUDA-capable GPU
- 8GB+ RAM
- 10GB+ storage for model checkpoints

### Dependencies

```bash
torch>=2.0.0
torch-geometric>=2.0.0
torchvision>=0.15.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.3.0
Pillow>=8.0.0
opencv-python>=4.5.0
tqdm>=4.60.0
PyYAML>=5.4.0
timm>=0.4.0
pandas>=1.3.0
```

## 🚀 Setup

**Note:** For initial project setup and dependencies, please refer to the main project README.md in the root directory.

This module requires PyTorch Geometric. If not already installed:

**Method 1 - Conda (Recommended):**
```bash
conda install pyg -c pyg
```

**Method 2 - Pip with pre-built wheels:**
```bash
# Check your PyTorch version first
python -c "import torch; print(torch.__version__)"

# For PyTorch 2.0+ with CUDA 11.8
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For PyTorch 2.0+ with CPU only
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

**Verify Installation:**
```bash
python -c "import torch_geometric; print('PyG installed successfully!')"
```

## 📊 Data Preparation

Your data should be organized as follows:

```
data/
├── train_images/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── test_images/
    ├── test1.png
    ├── test2.png
    └── ...
```

**Supported formats:** PNG, JPG, JPEG, TIFF, TIF

**Recommended image size:** 224x224 pixels or larger

## 🏃‍♂️ Quick Start

### Demo

Run the demo to see GRAPHITE in action:

```bash
python demo.py
```

This will show you:
- Model architecture and parameters
- Training setup
- Data requirements
- Next steps for your own data

### Training

Train a model with default parameters:

```bash
python train.py --data_dir /path/to/your/data
```

Train with custom parameters:

```bash
python train.py \
    --data_dir /path/to/your/data \
    --epochs 150 \
    --batch_size 8 \
    --lr 0.0005 \
    --hidden_dim 256 \
    --output_dir ./output/my_experiment
```

### Using Trained Models

The trained model can be loaded and used for downstream tasks:

```python
import torch
from models.hiergat import HierGATSSL

# Load trained model
checkpoint = torch.load('output/hiergat_ssl/best_model.pt')
model = HierGATSSL(input_dim=128, hidden_dim=128, num_gat_layers=3, num_heads=4, num_levels=3)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for feature extraction, transfer learning, etc.
```

## ⚙️ Configuration

### Command Line Arguments

#### Training (`train.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data_dir` | str | Required | Path to training images |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch_size` | int | 4 | Batch size |
| `--lr` | float | 0.001 | Learning rate |
| `--hidden_dim` | int | 128 | Hidden dimension |
| `--num_heads` | int | 4 | Number of attention heads |
| `--temperature` | float | 0.07 | InfoMax temperature |
| `--alpha` | float | 0.5 | InfoMax loss weight |
| `--beta` | float | 0.5 | Scale-wise loss weight |
| `--resume` | str | None | Resume from checkpoint |

#### Model Loading

To load a trained model programmatically:

```python
checkpoint = torch.load('output/hiergat_ssl/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Configuration File

You can also use a YAML configuration file:

```yaml
# config/config.yaml
data:
  patch_size: 224
  levels: 3

model_params:
  input_dim: 128
  hidden_dim: 256
  num_heads: 4
  num_gat_layers: 3
  dropout: 0.1

training_params:
  batch_size: 8
  learning_rate: 0.0001
  weight_decay: 1e-5
  num_epochs: 100
```

## 🏗️ Architecture

### Model Components

1. **Patch Extraction**: Multi-scale patch extraction at 3 hierarchical levels
2. **Feature Extraction**: Pre-trained CNN backbone for patch feature extraction
3. **Graph Construction**: Spatial and cross-scale edge connections
4. **Hierarchical GAT**: Graph attention networks with scale-wise attention
5. **Loss Function**: Combined InfoMax + Scale-wise loss

### Loss Function

The model uses a hierarchical loss combining:

- **InfoMax Loss**: Maximizes mutual information between local patches and global graph representations
- **Scale-wise Loss**: Enforces consistency across different magnification scales

```
L_total = α × L_infomax + β × L_scale
```

Where:
- `L_infomax`: InfoMax contrastive loss
- `L_scale`: Scale-wise consistency loss  
- `α, β`: Loss weighting parameters

## 📈 Training

### Training Process

1. **Data Loading**: Images are loaded and processed into hierarchical graphs
2. **Graph Construction**: Spatial and cross-scale edges are created
3. **Forward Pass**: Features flow through hierarchical GAT layers
4. **Loss Computation**: Combined InfoMax + Scale-wise loss
5. **Optimization**: Adam optimizer with learning rate scheduling

### Monitoring

Training progress is logged to:
- Console output with progress bars
- Training logs in `output/training.log`
- Loss plots saved after each epoch
- Model checkpoints every 5 epochs

### Early Stopping

Training automatically stops if loss doesn't improve for `patience` epochs (default: 10).

## 💾 Model Outputs

### Training Outputs

The training process generates:

- **Trained Model**: `output/hiergat_ssl/best_model.pt`
- **Training Logs**: `output/hiergat_ssl/training.log`
- **Loss Plots**: `output/hiergat_ssl/training_progress_epoch_X.png`
- **Checkpoints**: `output/hiergat_ssl/checkpoint_epoch_X.pt`

### Model Capabilities

The trained model can generate:

- **Node Embeddings**: Feature representations for graph nodes
- **Graph Embeddings**: Global graph-level representations
- **Attention Weights**: Multi-level attention scores

## 🧪 Experiments

### Example Training Scripts

**Basic training:**
```bash
python train.py --data_dir ./data/train --epochs 50
```

**Advanced training:**
```bash
python train.py \
    --data_dir ./data/train \
    --epochs 200 \
    --batch_size 6 \
    --lr 0.0003 \
    --hidden_dim 256 \
    --num_heads 8 \
    --alpha 0.6 \
    --beta 0.4 \
    --patience 15
```

**Resume training:**
```bash
python train.py \
    --data_dir ./data/train \
    --resume output/hiergat_ssl/checkpoint_epoch_50.pt
```

## 🐛 Troubleshooting

### Installation Issues

**PyTorch Geometric installation fails:**
```bash
# Method 1: Use conda (most reliable)
conda install pyg -c pyg

# Method 2: Try different PyTorch version
pip install torch==2.0.1 torchvision==0.15.2
pip install torch-geometric

# Method 3: Install from source
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Verify installation
python -c "import torch_geometric; print('PyG installed successfully!')"
```

**Import errors:**
```bash
# Run installation test
python test_installation.py

# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Check Python environment
which python
pip list | grep torch
```

### Training Issues

**CUDA out of memory:**
- Reduce `--batch_size` (try 2 or 1)
- Reduce `--hidden_dim` (try 64 or 96)
- Use gradient checkpointing
- Process smaller image patches

**Slow training:**
- Reduce `--num_workers` if using CPU
- Use smaller images (224x224 recommended)
- Enable mixed precision training

**NaN losses:**
- Reduce learning rate (try 0.0001)
- Check input data for corrupted images
- Adjust loss weights (`--alpha`, `--beta`)
- Check for gradient clipping

**Model not converging:**
- Increase learning rate (try 0.001)
- Adjust temperature parameters
- Check data preprocessing
- Verify graph construction

### GPU Memory Requirements

| Batch Size | Hidden Dim | Estimated GPU Memory |
|------------|------------|---------------------|
| 2 | 128 | ~4GB |
| 4 | 128 | ~6GB |
| 8 | 256 | ~12GB |


## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PyTorch Geometric team for graph neural network implementations
- The histopathology research community for datasets and benchmarks
- Contributors and users of this codebase

