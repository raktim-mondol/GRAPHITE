#!/bin/bash

# GRAPHITE Training Script for Core Images
# This script runs self-supervised training on the core images dataset

echo "============================================================"
echo "GRAPHITE: Self-Supervised Training on Core Images"
echo "============================================================"

# Activate Python environment
echo "Activating Python environment..."
source /home/raktim/upython/bin/activate

# Check if the environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate Python environment"
    exit 1
fi

# Verify CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else \"None\"}')"

# Create output directory if it doesn't exist
mkdir -p output/core_images_training

# Run training
echo "Starting training..."
echo "Command: python train.py --data_dir dataset/core_images --epochs 50 --batch_size 1 --lr 0.001 --output_dir output/ss_training --num_workers 0 --verbose"
echo ""

python train.py \
    --data_dir dataset/core_images \
    --epochs 50 \
    --batch_size 1 \
    --lr 0.001 \
    --output_dir output/core_images_training \
    --num_workers 0 \
    --verbose

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "Training completed successfully!"
    echo "Check output/core_images_training/ for results"
    echo "============================================================"
else
    echo ""
    echo "============================================================"
    echo "Training failed or was interrupted"
    echo "Check the logs for details"
    echo "============================================================"
    exit 1
fi
