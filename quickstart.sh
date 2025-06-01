#!/bin/bash

# GRAPHITE: Multi-Step Histopathology Analysis Pipeline
# Quickstart Script for Easy Setup and Execution

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[INFO] $1${NC}"
}

# Print banner
print_banner() {
    echo -e "${PURPLE}"
    echo "=================================================================="
    echo "🔬 GRAPHITE: Multi-Step Histopathology Analysis Pipeline"
    echo "=================================================================="
    echo -e "${NC}"
    echo "A comprehensive ML pipeline for histopathology image analysis"
    echo "featuring MIL classification, self-supervised learning, and XAI"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Python
    if ! command -v python &> /dev/null; then
        error "Python is not installed or not in PATH"
    fi
    
    # Check Python version
    python_version=$(python --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    required_version="3.8"
    if [[ $(echo "$python_version < $required_version" | bc -l) -eq 1 ]]; then
        error "Python $required_version or higher is required. Found: $python_version"
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        error "pip is not installed or not in PATH"
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        warn "No NVIDIA GPU detected. CPU-only mode will be used."
    fi
    
    log "Prerequisites check completed ✓"
}

# Setup environment
setup_environment() {
    log "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "graphite_env" ]; then
        info "Creating virtual environment..."
        python -m venv graphite_env
    fi
    
    # Activate virtual environment
    source graphite_env/bin/activate || source graphite_env/Scripts/activate
    
    # Upgrade pip
    pip install --upgrade pip setuptools wheel
    
    log "Environment setup completed ✓"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    # Install PyTorch first
    if command -v nvidia-smi &> /dev/null; then
        info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
    else
        info "Installing PyTorch CPU-only..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install PyTorch Geometric
    info "Installing PyTorch Geometric..."
    if command -v conda &> /dev/null; then
        conda install pyg -c pyg -y
    else
        pip install torch-geometric
        if command -v nvidia-smi &> /dev/null; then
            pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
        else
            pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
        fi
    fi
    
    # Install remaining dependencies
    info "Installing remaining dependencies..."
    pip install -r requirements.txt
    
    log "Dependencies installation completed ✓"
}

# Verify installation
verify_installation() {
    log "Verifying installation..."
    
    python -c "
import torch
import torch_geometric
import numpy as np
import cv2
import matplotlib
import pandas as pd
import sklearn

print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ PyTorch Geometric: {torch_geometric.__version__}')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU Count: {torch.cuda.device_count()}')
    print(f'✓ Current GPU: {torch.cuda.get_device_name()}')
print('✓ All dependencies verified successfully!')
" || error "Installation verification failed"
    
    log "Installation verification completed ✓"
}

# Setup data structure
setup_data_structure() {
    log "Setting up data directory structure..."
    
    # Create directory structure
    mkdir -p dataset/training_dataset_step_1/tma_core
    mkdir -p dataset/training_dataset_step_2/core_image
    mkdir -p dataset/training_dataset_step_2/mask
    mkdir -p dataset/visualization_dataset/core_image
    mkdir -p dataset/visualization_dataset/mask
    mkdir -p logs
    mkdir -p notebooks
    
    # Create output directories
    mkdir -p training_step_1/mil_classification/output
    mkdir -p training_step_2/self_supervised_training/output
    mkdir -p visualization_step_1/xai_visualization/output
    mkdir -p visualization_step_2/fusion_visualization/output
    
    log "Data structure setup completed ✓"
}

# Run demos
run_demos() {
    log "Running demonstration scripts..."
    
    # MIL Classification quick test
    info "Testing MIL Classification module..."
    cd training_step_1/mil_classification
    if [ -f "quick_training_test.py" ]; then
        python quick_training_test.py || warn "MIL quick test failed"
    else
        warn "MIL quick test script not found"
    fi
    cd ../..
    
    # Self-supervised learning demo
    info "Testing Self-Supervised Learning module..."
    cd training_step_2/self_supervised_training
    if [ -f "demo.py" ]; then
        python demo.py || warn "SSL demo failed"
    else
        warn "SSL demo script not found"
    fi
    
    if [ -f "test_installation.py" ]; then
        python test_installation.py || warn "SSL installation test failed"
    fi
    cd ../..
    
    log "Demo runs completed ✓"
}

# Main menu
show_menu() {
    echo ""
    echo -e "${CYAN}==================== MAIN MENU ====================${NC}"
    echo "1. 🔧 Full Setup (Environment + Dependencies)"
    echo "2. 📊 Setup Data Structure Only"
    echo "3. ✅ Verify Installation"
    echo "4. 🚀 Run Demos"
    echo "5. 🧬 Train MIL Classification"
    echo "6. 🔄 Train Self-Supervised Learning"
    echo "7. 👁️  Run XAI Visualization"
    echo "8. 🔮 Run Fusion Visualization"
    echo "9. 📓 Start Jupyter Notebook"
    echo "10. 🐳 Docker Setup"
    echo "11. 📋 System Information"
    echo "0. 🚪 Exit"
    echo -e "${CYAN}==================================================${NC}"
    echo ""
}

# Individual training functions
train_mil() {
    log "Starting MIL Classification Training..."
    cd training_step_1/mil_classification
    
    echo "Available options:"
    echo "1. Quick training test"
    echo "2. Full training with default parameters"
    echo "3. Custom training (interactive)"
    read -p "Choose option [1-3]: " mil_option
    
    case $mil_option in
        1)
            python quick_training_test.py
            ;;
        2)
            python train.py \
                --root_dir ../../dataset/training_dataset_step_1/tma_core \
                --cancer_labels_path ../../dataset/cancer.txt \
                --normal_labels_path ../../dataset/normal.txt \
                --batch_size 8 \
                --num_epochs 50
            ;;
        3)
            echo "Enter custom parameters (or press Enter for defaults):"
            read -p "Batch size [8]: " batch_size
            read -p "Number of epochs [50]: " epochs
            read -p "Learning rate [0.001]: " lr
            read -p "Max patches per patient [100]: " max_patches
            
            python train.py \
                --root_dir ../../dataset/training_dataset_step_1/tma_core \
                --cancer_labels_path ../../dataset/cancer.txt \
                --normal_labels_path ../../dataset/normal.txt \
                --batch_size ${batch_size:-8} \
                --num_epochs ${epochs:-50} \
                --learning_rate ${lr:-0.001} \
                --max_patches ${max_patches:-100}
            ;;
    esac
    cd ../..
}

train_ssl() {
    log "Starting Self-Supervised Learning Training..."
    cd training_step_2/self_supervised_training
    
    echo "Available options:"
    echo "1. Demo run"
    echo "2. Full training with default parameters"
    echo "3. Custom training (interactive)"
    read -p "Choose option [1-3]: " ssl_option
    
    case $ssl_option in
        1)
            python demo.py
            ;;
        2)
            python train.py \
                --data_dir ../../dataset/training_dataset_step_2/core_image \
                --epochs 100 \
                --batch_size 4
            ;;
        3)
            echo "Enter custom parameters (or press Enter for defaults):"
            read -p "Batch size [4]: " batch_size
            read -p "Number of epochs [100]: " epochs
            read -p "Learning rate [0.001]: " lr
            read -p "Hidden dimension [128]: " hidden_dim
            
            python train.py \
                --data_dir ../../dataset/training_dataset_step_2/core_image \
                --epochs ${epochs:-100} \
                --batch_size ${batch_size:-4} \
                --lr ${lr:-0.001} \
                --hidden_dim ${hidden_dim:-128}
            ;;
    esac
    cd ../..
}

run_xai() {
    log "Starting XAI Visualization..."
    cd visualization_step_1/xai_visualization
    
    python main.py \
        --model_path ../../training_step_1/mil_classification/output/best_model.pth \
        --data_path ../../dataset/visualization_dataset/core_image \
        --output_dir ./output/xai_results
    
    cd ../..
}

run_fusion() {
    log "Starting Fusion Visualization..."
    cd visualization_step_2/fusion_visualization
    
    echo "Available fusion methods:"
    echo "1. Multi-level fusion"
    echo "2. Final fusion"
    read -p "Choose method [1-2]: " fusion_option
    
    case $fusion_option in
        1)
            python main_multi_level_fusion.py \
                --model_path ../../training_step_2/self_supervised_training/output/best_model.pt \
                --mil_model_path ../../training_step_1/mil_classification/output/best_model.pth \
                --dataset_dir ../../dataset/visualization_dataset/core_image \
                --mask_dir ../../dataset/visualization_dataset/mask
            ;;
        2)
            python main_final_fusion.py \
                --cam_method fullgrad \
                --fusion_method confidence \
                --calculate_metrics True
            ;;
    esac
    cd ../..
}

start_jupyter() {
    log "Starting Jupyter Notebook..."
    
    # Activate environment
    source graphite_env/bin/activate || source graphite_env/Scripts/activate
    
    # Install Jupyter if not present
    pip install jupyter ipykernel
    
    # Register kernel
    python -m ipykernel install --user --name=graphite --display-name="GRAPHITE"
    
    # Start Jupyter
    info "Jupyter will start on http://localhost:8888"
    jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
}

docker_setup() {
    log "Setting up Docker environment..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
    fi
    
    echo "Docker setup options:"
    echo "1. Build and run container"
    echo "2. Run with Jupyter"
    echo "3. Run with TensorBoard"
    echo "4. Stop containers"
    read -p "Choose option [1-4]: " docker_option
    
    case $docker_option in
        1)
            docker-compose up --build
            ;;
        2)
            docker-compose --profile jupyter up --build
            ;;
        3)
            docker-compose --profile tensorboard up --build
            ;;
        4)
            docker-compose down
            ;;
    esac
}

show_system_info() {
    echo -e "${CYAN}==================== SYSTEM INFO ====================${NC}"
    echo "OS: $(uname -s)"
    echo "Python: $(python --version)"
    echo "Pip: $(pip --version)"
    
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    else
        echo "GPU: Not detected"
    fi
    
    echo "Available disk space: $(df -h . | tail -1 | awk '{print $4}')"
    echo "Available memory: $(free -h | grep Mem | awk '{print $7}')"
    echo -e "${CYAN}==================================================${NC}"
}

# Main execution
main() {
    print_banner
    
    while true; do
        show_menu
        read -p "Please select an option [0-11]: " choice
        
        case $choice in
            1)
                check_prerequisites
                setup_environment
                install_dependencies
                verify_installation
                setup_data_structure
                run_demos
                ;;
            2)
                setup_data_structure
                ;;
            3)
                verify_installation
                ;;
            4)
                run_demos
                ;;
            5)
                train_mil
                ;;
            6)
                train_ssl
                ;;
            7)
                run_xai
                ;;
            8)
                run_fusion
                ;;
            9)
                start_jupyter
                ;;
            10)
                docker_setup
                ;;
            11)
                show_system_info
                ;;
            0)
                log "Thank you for using GRAPHITE! 🔬"
                exit 0
                ;;
            *)
                error "Invalid option. Please choose 0-11."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@" 