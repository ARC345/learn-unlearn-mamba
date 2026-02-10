#!/bin/bash
# Bootstrap script for E2E Networks GPU instance.
# Usage:
#   ssh into E2E instance, then:
#   curl -sSL <raw-github-url>/setup_e2e.sh | bash
#   OR: git clone repo && bash setup_e2e.sh
set -e

echo "=== E2E Networks Setup for learn-unlearn-mamba ==="

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

# Run full experiment
echo "Starting experiment..."
pixi run experiment-1-4b
