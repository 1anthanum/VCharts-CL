#!/bin/bash
# ============================================================
# Lambda Cloud Setup Script for VTBench
# ============================================================
#
# Run this after SSH into a Lambda Cloud instance:
#   ssh ubuntu@<ip>
#   bash lambda_setup.sh
#
# Assumes: Ubuntu 22.04, NVIDIA GPU, CUDA pre-installed
# ============================================================

set -e

echo "=== Lambda Cloud VTBench Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA: $(nvcc --version | tail -1)"

# --- 1. System packages ---
sudo apt-get update -qq
sudo apt-get install -y -qq git unzip

# --- 2. Clone / upload project ---
# Option A: If you've uploaded vtbench.tar.gz
if [ -f "vtbench.tar.gz" ]; then
    echo "Extracting vtbench.tar.gz..."
    tar xzf vtbench.tar.gz
    cd vtbench
# Option B: If using git
elif [ -d "vtbench" ]; then
    cd vtbench
else
    echo "ERROR: No vtbench directory or archive found."
    echo "Upload vtbench.tar.gz or clone the repo first."
    exit 1
fi

# --- 3. Python environment ---
# Lambda Cloud typically has conda pre-installed
conda create -n vtbench python=3.10 -y
conda activate vtbench || source activate vtbench

# PyTorch with CUDA (Lambda has CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project dependencies
pip install -e .
pip install imbalanced-learn

# --- 4. Verify ---
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# --- 5. Pre-generate images (if not uploaded) ---
if [ ! -d "chart_images" ] || [ -z "$(ls -A chart_images 2>/dev/null)" ]; then
    echo ""
    echo "No chart_images found. Generating baseline images..."
    echo "This will take ~30 min on Lambda Cloud."
    python scripts/pregenerate_images.py --phase baseline
    echo ""
    echo "Generating variant images for experiment datasets..."
    python scripts/pregenerate_images.py --phase variants
fi

echo ""
echo "=== Setup Complete ==="
echo "Run experiments with:"
echo "  nohup bash scripts/run_all_experiments.sh > experiment_all.log 2>&1 &"
echo "  tail -f experiment_all.log"
