Installation Guide
==================

This guide provides comprehensive installation instructions for GRAPHITE on different platforms.

🔧 System Requirements
----------------------

**Hardware Requirements:**
   - **GPU**: NVIDIA Tesla V100 32GB (recommended) or any CUDA-compatible GPU with 8GB+ VRAM
   - **RAM**: 32GB+ system memory recommended (minimum 16GB)
   - **Storage**: 50GB+ free space for datasets and outputs
   - **CPU**: Multi-core processor (8+ cores recommended)

**Software Requirements:**
   - **OS**: Linux (Ubuntu 18.04+), Windows 10/11 with WSL2, or macOS 10.15+
   - **Python**: 3.9.2 or higher
   - **CUDA**: 11.7 or higher (for GPU acceleration)
   - **Docker**: Optional but recommended for containerized deployment

🚀 Quick Installation
---------------------

**Option 1: Automated Setup (Recommended)**

Clone the repository and run the automated setup:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/raktim-mondol/GRAPHITE.git
   cd GRAPHITE
   
   # Run automated setup
   chmod +x quickstart.sh
   ./quickstart.sh

**Option 2: Manual Installation**

.. code-block:: bash

   # Clone repository
   git clone https://github.com/raktim-mondol/GRAPHITE.git
   cd GRAPHITE
   
   # Create virtual environment
   python3 -m venv graphite_env
   source graphite_env/bin/activate  # On Windows: graphite_env\Scripts\activate
   
   # Install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt

🐳 Docker Installation
----------------------

For a containerized setup that ensures reproducibility:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/raktim-mondol/GRAPHITE.git
   cd GRAPHITE
   
   # Build and run with Docker Compose
   docker-compose up --build

📦 Package Dependencies
-----------------------

**Core Dependencies:**

.. code-block:: text

   torch>=2.0.0
   torchvision>=0.15.0
   torch-geometric>=2.3.0
   numpy>=1.21.0
   scikit-learn>=1.0.0
   matplotlib>=3.5.0
   seaborn>=0.11.0
   pandas>=1.3.0
   opencv-python>=4.5.0
   pillow>=8.3.0
   tqdm>=4.62.0
   pyyaml>=6.0
   tensorboard>=2.8.0

**XAI and Visualization:**

.. code-block:: text

   grad-cam>=1.4.0
   shap>=0.40.0
   lime>=0.2.0
   captum>=0.5.0
   plotly>=5.0.0
   bokeh>=2.4.0

**Medical Imaging:**

.. code-block:: text

   openslide-python>=1.1.2
   histomicstk>=1.2.0
   tiatoolbox>=1.4.0

🔧 Platform-Specific Setup
---------------------------

**Ubuntu/Linux:**

.. code-block:: bash

   # Install system dependencies
   sudo apt-get update
   sudo apt-get install -y python3-dev python3-pip git
   sudo apt-get install -y libopencv-dev python3-opencv
   sudo apt-get install -y openslide-tools
   
   # Install CUDA (if using GPU)
   # Follow NVIDIA CUDA installation guide for your system

**Windows (WSL2 Recommended):**

.. code-block:: bash

   # Enable WSL2 and install Ubuntu
   wsl --install -d Ubuntu
   
   # Inside WSL2, follow Linux installation steps
   # Or use native Windows installation:
   
   # Install Python 3.9+ from python.org
   # Install Git for Windows
   # Clone repository and install dependencies as above

**macOS:**

.. code-block:: bash

   # Install Homebrew if not already installed
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   # Install dependencies
   brew install python@3.9 git
   brew install openslide
   
   # Note: GPU acceleration not available on macOS

🧪 Verify Installation
----------------------

Run the verification script to ensure everything is installed correctly:

.. code-block:: bash

   # Test basic installation
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   
   # Run comprehensive tests
   cd GRAPHITE
   python -m pytest tests/ -v

**Expected Output:**

.. code-block:: text

   PyTorch version: 2.0.0+cu117
   CUDA available: True
   ========================= test session starts =========================
   tests/test_installation.py::test_imports PASSED
   tests/test_installation.py::test_cuda PASSED
   tests/test_installation.py::test_data_loading PASSED
   ========================= 3 passed in 2.34s =========================

🚨 Troubleshooting
------------------

**Common Issues:**

**CUDA Not Available:**

.. code-block:: bash

   # Check NVIDIA driver
   nvidia-smi
   
   # Reinstall PyTorch with CUDA support
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117

**OpenSlide Installation Issues:**

.. code-block:: bash

   # Ubuntu/Linux
   sudo apt-get install openslide-tools
   pip install openslide-python
   
   # Windows (use conda)
   conda install -c conda-forge openslide-python

**Memory Issues:**

.. code-block:: bash

   # Reduce batch size in config files
   # Enable gradient checkpointing
   # Use mixed precision training

**Permission Errors:**

.. code-block:: bash

   # Fix file permissions
   chmod -R 755 dataset/
   chmod +x quickstart.sh

🔄 Updates and Maintenance
--------------------------

Keep GRAPHITE updated:

.. code-block:: bash

   # Pull latest changes
   git pull origin main
   
   # Update dependencies
   pip install -r requirements.txt --upgrade
   
   # Re-run tests
   python -m pytest tests/ -v

💡 Performance Optimization
---------------------------

**For Optimal Performance:**

1. **Use SSD storage** for faster data loading
2. **Enable mixed precision** training for memory efficiency
3. **Adjust batch sizes** based on GPU memory
4. **Use multiple GPUs** if available
5. **Optimize data loading** with appropriate num_workers

.. code-block:: python

   # Example configuration for optimal performance
   config = {
       'batch_size': 16,  # Adjust based on GPU memory
       'num_workers': 4,  # Adjust based on CPU cores
       'pin_memory': True,
       'mixed_precision': True,
       'gradient_checkpointing': True
   }

📋 Next Steps
-------------

After successful installation:

1. **Check Data Structure**: Review :doc:`data_structure` for dataset organization
2. **Quick Start**: Follow :doc:`quickstart` for your first run
3. **Tutorials**: Explore :doc:`tutorials/basic_usage` for detailed examples
4. **API Reference**: Browse :doc:`api/training_step_1` for development

For additional help, see :doc:`development/troubleshooting` or create an issue on GitHub.
