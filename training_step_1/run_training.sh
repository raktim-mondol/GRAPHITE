#!/bin/bash

# MIL Training Script for Histopathology Image Classification
# This script provides an easy interface to run the MIL training pipeline

set -e  # Exit on any error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIL_DIR="${SCRIPT_DIR}/mil_classification"
LOG_FILE="${MIL_DIR}/training_$(date +%Y%m%d_%H%M%S).log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DEFAULT_DATA_DIR=""
DEFAULT_CANCER_LABELS="../../dataset/cancer.txt"
DEFAULT_NORMAL_LABELS="../../dataset/normal.txt"
DEFAULT_BATCH_SIZE=8
DEFAULT_MAX_PATCHES=100
DEFAULT_EPOCHS=100
DEFAULT_LEARNING_RATE=0.001
DEFAULT_TEST_SIZE=0.3
DEFAULT_RANDOM_STATE=78
DEFAULT_PATIENCE=10
DEFAULT_METRICS="auc"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

MIL Training Script for Histopathology Image Classification

OPTIONS:
    -h, --help                     Show this help message
    -d, --data_dir DIR             Root directory containing patient folders
    -c, --cancer_labels FILE       Path to cancer patient labels file (default: $DEFAULT_CANCER_LABELS)
    -n, --normal_labels FILE       Path to normal patient labels file (default: $DEFAULT_NORMAL_LABELS)
    -b, --batch_size SIZE          Batch size for training (default: $DEFAULT_BATCH_SIZE)
    -p, --max_patches NUM          Maximum patches per patient (default: $DEFAULT_MAX_PATCHES)
    -e, --epochs NUM               Number of training epochs (default: $DEFAULT_EPOCHS)
    -l, --learning_rate RATE       Learning rate (default: $DEFAULT_LEARNING_RATE)
    -t, --test_size RATIO          Test set proportion (default: $DEFAULT_TEST_SIZE)
    -s, --random_state SEED        Random seed (default: $DEFAULT_RANDOM_STATE)
    --patience NUM                 Early stopping patience (default: $DEFAULT_PATIENCE)
    --metrics METRIC               Metric to monitor (default: $DEFAULT_METRICS)
    --color_norm                   Enable Macenko color normalization
    --balanced_sampler             Use balanced batch sampling
    --quick_test                   Run quick test (2 epochs, 50 patches)
    --gpu_id ID                    Specify GPU ID to use
    --dry_run                      Show command without executing

EXAMPLES:
    # Basic training with default parameters
    $0

    # Training with custom data directory and parameters
    $0 --data_dir /path/to/data --epochs 150 --batch_size 16

    # Quick test run
    $0 --quick_test

    # Training with color normalization and balanced sampling
    $0 --color_norm --balanced_sampler --epochs 200

    # Training on specific GPU
    $0 --gpu_id 1 --batch_size 16

EOF
}

# Function to check requirements
check_requirements() {
    print_info "Checking requirements..."
    
    # Check if we're in the right directory
    if [[ ! -d "$MIL_DIR" ]]; then
        print_error "mil_classification directory not found. Please run this script from training_step_1 directory."
        exit 1
    fi
    
    # Check if train.py exists
    if [[ ! -f "$MIL_DIR/train.py" ]]; then
        print_error "train.py not found in mil_classification directory."
        exit 1
    fi
    
    # Check Python
    if ! command -v python &> /dev/null; then
        print_error "Python not found. Please install Python 3.8+."
        exit 1
    fi
    
    # Check Python version
    python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        print_error "Python 3.8+ required. Found: $python_version"
        exit 1
    fi
    
    print_success "Requirements check passed."
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    if [[ -f "$MIL_DIR/requirements.txt" ]]; then
        pip install -r "$MIL_DIR/requirements.txt" || {
            print_error "Failed to install dependencies."
            exit 1
        }
        print_success "Dependencies installed successfully."
    else
        print_warning "requirements.txt not found. Assuming dependencies are already installed."
    fi
}

# Function to verify data structure
verify_data() {
    local data_dir="$1"
    local cancer_labels="$2"
    local normal_labels="$3"
    
    if [[ -n "$data_dir" ]]; then
        if [[ ! -d "$data_dir" ]]; then
            print_error "Data directory does not exist: $data_dir"
            exit 1
        fi
        print_success "Data directory found: $data_dir"
    fi
    
    if [[ -f "$cancer_labels" ]]; then
        print_success "Cancer labels file found: $cancer_labels"
    else
        print_warning "Cancer labels file not found: $cancer_labels"
    fi
    
    if [[ -f "$normal_labels" ]]; then
        print_success "Normal labels file found: $normal_labels"
    else
        print_warning "Normal labels file not found: $normal_labels"
    fi
}

# Function to set up GPU environment
setup_gpu() {
    local gpu_id="$1"
    
    if [[ -n "$gpu_id" ]]; then
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        print_info "Using GPU: $gpu_id"
    fi
    
    # Check CUDA availability
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || {
        print_warning "Could not check CUDA availability. PyTorch may not be installed."
    }
}

# Function to run quick test
run_quick_test() {
    print_info "Running quick training test..."
    
    cd "$MIL_DIR"
    python quick_training_test.py || {
        print_error "Quick test failed."
        exit 1
    }
    
    print_success "Quick test completed successfully."
    cd "$SCRIPT_DIR"
}

# Function to build training command
build_training_command() {
    local cmd="python train.py"
    
    # Add parameters
    [[ -n "$DATA_DIR" ]] && cmd="$cmd --root_dir '$DATA_DIR'"
    [[ -n "$CANCER_LABELS" ]] && cmd="$cmd --cancer_labels_path '$CANCER_LABELS'"
    [[ -n "$NORMAL_LABELS" ]] && cmd="$cmd --normal_labels_path '$NORMAL_LABELS'"
    [[ -n "$BATCH_SIZE" ]] && cmd="$cmd --batch_size $BATCH_SIZE"
    [[ -n "$MAX_PATCHES" ]] && cmd="$cmd --max_patches $MAX_PATCHES"
    [[ -n "$EPOCHS" ]] && cmd="$cmd --num_epochs $EPOCHS"
    [[ -n "$LEARNING_RATE" ]] && cmd="$cmd --learning_rate $LEARNING_RATE"
    [[ -n "$TEST_SIZE" ]] && cmd="$cmd --test_size $TEST_SIZE"
    [[ -n "$RANDOM_STATE" ]] && cmd="$cmd --random_state $RANDOM_STATE"
    [[ -n "$PATIENCE" ]] && cmd="$cmd --early_stopping_patience $PATIENCE"
    [[ -n "$METRICS" ]] && cmd="$cmd --metrics_to_monitor '$METRICS'"
    [[ "$COLOR_NORM" == "true" ]] && cmd="$cmd --use_color_normalization"
    [[ "$BALANCED_SAMPLER" == "true" ]] && cmd="$cmd --use_balanced_sampler"
    
    echo "$cmd"
}

# Function to run training
run_training() {
    local cmd="$1"
    local dry_run="$2"
    
    print_info "Training configuration:"
    echo "  Data directory: ${DATA_DIR:-'Default (PBS_JOBFS)'}"
    echo "  Cancer labels: ${CANCER_LABELS}"
    echo "  Normal labels: ${NORMAL_LABELS}"
    echo "  Batch size: ${BATCH_SIZE}"
    echo "  Max patches: ${MAX_PATCHES}"
    echo "  Epochs: ${EPOCHS}"
    echo "  Learning rate: ${LEARNING_RATE}"
    echo "  Test size: ${TEST_SIZE}"
    echo "  Random state: ${RANDOM_STATE}"
    echo "  Patience: ${PATIENCE}"
    echo "  Metrics to monitor: ${METRICS}"
    echo "  Color normalization: ${COLOR_NORM:-false}"
    echo "  Balanced sampler: ${BALANCED_SAMPLER:-false}"
    echo "  Log file: ${LOG_FILE}"
    echo ""
    
    print_info "Command to execute:"
    echo "  $cmd"
    echo ""
    
    if [[ "$dry_run" == "true" ]]; then
        print_info "Dry run mode - command not executed."
        return 0
    fi
    
    print_info "Starting training..."
    cd "$MIL_DIR"
    
    # Create output directory
    mkdir -p output
    
    # Run training with logging
    eval "$cmd" 2>&1 | tee "$LOG_FILE" || {
        print_error "Training failed. Check log file: $LOG_FILE"
        exit 1
    }
    
    print_success "Training completed successfully!"
    print_info "Check outputs in: $MIL_DIR/output/"
    print_info "Full log available at: $LOG_FILE"
    
    cd "$SCRIPT_DIR"
}

# Parse command line arguments
QUICK_TEST=false
DRY_RUN=false
COLOR_NORM=false
BALANCED_SAMPLER=false
GPU_ID=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -d|--data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -c|--cancer_labels)
            CANCER_LABELS="$2"
            shift 2
            ;;
        -n|--normal_labels)
            NORMAL_LABELS="$2"
            shift 2
            ;;
        -b|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -p|--max_patches)
            MAX_PATCHES="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        -t|--test_size)
            TEST_SIZE="$2"
            shift 2
            ;;
        -s|--random_state)
            RANDOM_STATE="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --color_norm)
            COLOR_NORM=true
            shift
            ;;
        --balanced_sampler)
            BALANCED_SAMPLER=true
            shift
            ;;
        --quick_test)
            QUICK_TEST=true
            shift
            ;;
        --gpu_id)
            GPU_ID="$2"
            shift 2
            ;;
        --dry_run)
            DRY_RUN=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Set default values if not provided
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
CANCER_LABELS="${CANCER_LABELS:-$DEFAULT_CANCER_LABELS}"
NORMAL_LABELS="${NORMAL_LABELS:-$DEFAULT_NORMAL_LABELS}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
MAX_PATCHES="${MAX_PATCHES:-$DEFAULT_MAX_PATCHES}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
TEST_SIZE="${TEST_SIZE:-$DEFAULT_TEST_SIZE}"
RANDOM_STATE="${RANDOM_STATE:-$DEFAULT_RANDOM_STATE}"
PATIENCE="${PATIENCE:-$DEFAULT_PATIENCE}"
METRICS="${METRICS:-$DEFAULT_METRICS}"

# Quick test overrides
if [[ "$QUICK_TEST" == "true" ]]; then
    EPOCHS=2
    MAX_PATCHES=50
    BATCH_SIZE=4
    print_info "Quick test mode enabled (2 epochs, 50 patches, batch size 4)"
fi

# Main execution
main() {
    print_info "Starting MIL Training Pipeline"
    print_info "=============================="
    
    # Check requirements
    check_requirements
    
    # Install dependencies
    install_dependencies
    
    # Set up GPU
    setup_gpu "$GPU_ID"
    
    # Verify data
    verify_data "$DATA_DIR" "$CANCER_LABELS" "$NORMAL_LABELS"
    
    # Run quick test if requested
    if [[ "$QUICK_TEST" == "true" ]] && [[ "$DRY_RUN" != "true" ]]; then
        run_quick_test
    fi
    
    # Build and run training command
    training_cmd=$(build_training_command)
    run_training "$training_cmd" "$DRY_RUN"
    
    print_success "MIL training pipeline completed!"
}

# Run main function
main "$@" 