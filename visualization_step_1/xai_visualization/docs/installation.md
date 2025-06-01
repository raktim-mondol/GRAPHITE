# Installation Guide

## Prerequisites

- Python 3.9.2 or higher
- PyTorch 2.0.0
- CUDA 11.7
- NVIDIA Tesla V100 32GB (used/recommended) or equivalent CUDA-compatible GPU (optional, for faster processing)

## Installation Methods

### Method 1: From Source (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/raktim-mondol/GRAPHITE.git
cd GRAPHITE
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

### Method 2: Using pip (when published)

```bash
pip install GRAPHITE
```

## Verify Installation

Test the installation by running:

```bash
python main.py --help
```

You should see the help message with all available options.

## GPU Support

For GPU acceleration, ensure you have:

1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed
3. PyTorch with CUDA support:

```bash
# For CUDA 11.7
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch'**
   - Solution: Install PyTorch using the commands above

2. **CUDA out of memory**
   - Solution: Reduce batch size or use CPU mode with `--device cpu`

3. **Missing model file**
   - Solution: Ensure the model path is correct and the file exists

### Getting Help

If you encounter issues:

1. Check the [FAQ](faq.md)
2. Search existing [GitHub issues](https://github.com/raktim-mondol/GRAPHITE/issues)
3. Create a new issue with detailed error information 