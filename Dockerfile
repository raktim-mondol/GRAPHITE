# GRAPHITE: Multi-Step Histopathology Analysis Pipeline
# Docker Configuration for Reproducible Deployment

FROM nvidia/cuda:11.7-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    vim \
    htop \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    openexr \
    libatlas-base-dev \
    python3-numpy \
    libtbb2 \
    libtbb-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support (aligned with requirements.txt)
RUN pip install torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cu117

# Install PyTorch Geometric and related packages
RUN pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional dependencies that may be needed
RUN pip install \
    wandb \
    jupyter \
    ipykernel \
    notebook

# Install conda for better package management (optional)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Copy the entire project
COPY . .

# Create output directories
RUN mkdir -p /workspace/training_step_1/mil_classification/output
RUN mkdir -p /workspace/training_step_2/self_supervised_training/output
RUN mkdir -p /workspace/visualization_step_1/xai_visualization/output
RUN mkdir -p /workspace/visualization_step_2/fusion_visualization/output

# Set permissions
RUN chmod +x training_step_2/self_supervised_training/ss_training.sh || true
RUN chmod +x visualization_step_2/fusion_visualization/run_code.sh || true

# Install Jupyter kernel
RUN python -m ipykernel install --name=graphite --display-name="GRAPHITE"

# Expose ports for Jupyter
EXPOSE 8888

# Create entrypoint script
RUN echo '#!/bin/bash\n\
echo "🔬 GRAPHITE: Multi-Step Histopathology Analysis Pipeline"\n\
echo "========================================================="\n\
echo ""\n\
echo "Available commands:"\n\
echo "  1. Train MIL Classification:"\n\
echo "     cd training_step_1/mil_classification && python train.py --help"\n\
echo ""\n\
echo "  2. Train Self-Supervised Learning:"\n\
echo "     cd training_step_2/self_supervised_training && python train.py --help"\n\
echo ""\n\
echo "  3. Run XAI Visualization:"\n\
echo "     cd visualization_step_1/xai_visualization && python main.py --help"\n\
echo ""\n\
echo "  4. Run Fusion Visualization:"\n\
echo "     cd visualization_step_2/fusion_visualization && python main_final_fusion.py --help"\n\
echo ""\n\
echo "  5. Start Jupyter Notebook:"\n\
echo "     jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root"\n\
echo ""\n\
echo "  6. Run demos:"\n\
echo "     cd training_step_2/self_supervised_training && python demo.py"\n\
echo ""\n\
echo "GPU Status:"\n\
nvidia-smi || echo "No GPU detected"\n\
echo ""\n\
echo "Python Environment:"\n\
python -c "import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA Available: {torch.cuda.is_available()}\")" || echo "PyTorch not available"\n\
python -c "import torch_geometric; print(f\"PyTorch Geometric: {torch_geometric.__version__}\")" || echo "PyTorch Geometric not available"\n\
echo ""\n\
exec "$@"\n' > /entrypoint.sh && chmod +x /entrypoint.sh

# Set default command
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Labels for better organization
LABEL maintainer="GRAPHITE Team"
LABEL description="Multi-Step Histopathology Analysis Pipeline with MIL, SSL, and XAI"
LABEL version="1.0" 