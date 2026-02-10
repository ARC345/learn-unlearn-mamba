#!/bin/bash
# Vast.ai GPU instance setup for learn-unlearn-mamba
#
# Recommended Vast.ai template:
#   Image: pytorch/pytorch:2.4.1-cuda12.4-cudnn9-devel
#   GPU:   RTX 3090 or 4090 (24GB VRAM)
#   Disk:  30 GB
#
# Usage:
#   1. Rent an instance on vast.ai with the image above
#   2. SSH in: ssh -p PORT root@IP
#   3. Run:    bash setup.sh
#   OR one-liner after cloning:
#   git clone https://github.com/ARC345/learn-unlearn-mamba.git && cd learn-unlearn-mamba && bash setup.sh
set -e

echo "=== Vast.ai Setup for learn-unlearn-mamba ==="

# Vast.ai instances run as root, ensure HOME is set
export HOME="${HOME:-/root}"

# Install pixi if not present
if ! command -v pixi &> /dev/null; then
    echo "Installing pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
else
    echo "pixi already installed."
fi

# Clone repo if not already in it
if [ ! -f "pixi.toml" ]; then
    echo "Cloning repository..."
    git clone https://github.com/ARC345/learn-unlearn-mamba.git
    cd learn-unlearn-mamba
fi

# Install dependencies
echo "Installing dependencies with pixi..."
pixi install

# Verify GPU is visible
echo "GPU check:"
pixi run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

# Run full experiment (all 4 noise types)
echo "Starting experiment..."
pixi run experiment-1-4b
