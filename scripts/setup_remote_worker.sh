#!/bin/bash
# Setup a remote worker for Wayfinder Mathlib benchmarks.
# Usage: ssh macpro < scripts/setup_remote_worker.sh
#    or: ssh homebridge < scripts/setup_remote_worker.sh
set -e

WORK_DIR="$HOME/wayfinder_worker"
echo "=== Setting up Wayfinder worker at $WORK_DIR ==="
mkdir -p "$WORK_DIR"

# 1. Install elan (Lean version manager)
if ! command -v elan &>/dev/null && ! test -f "$HOME/.elan/bin/elan"; then
    echo "Installing elan..."
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none
    export PATH="$HOME/.elan/bin:$PATH"
else
    export PATH="$HOME/.elan/bin:$PATH"
    echo "elan already installed: $(elan --version 2>/dev/null || echo 'in PATH')"
fi

# 2. Install Lean 4.27.0 (matches Pantograph binary)
echo "Installing Lean 4.27.0..."
elan toolchain install leanprover/lean4:v4.27.0
elan default leanprover/lean4:v4.27.0
lean --version

# 3. Install Python packages
echo "Installing Python packages..."
pip3 install --user numpy PyYAML 2>/dev/null || python3 -m pip install --user numpy PyYAML
# sentence-transformers pulls torch if not installed
pip3 install --user sentence-transformers 2>/dev/null || echo "sentence-transformers install may need manual attention"

# 4. Install PyPantograph from source (builds against Lean 4.27.0)
echo "Installing PyPantograph..."
pip3 install --user 'pantograph @ git+https://github.com/lenianiva/PyPantograph@v0.3.13' 2>/dev/null || \
    python3 -m pip install --user 'pantograph @ git+https://github.com/lenianiva/PyPantograph@v0.3.13'

# 5. Verify
echo ""
echo "=== Verification ==="
echo "Lean: $(lean --version 2>/dev/null)"
python3 -c "import pantograph; print(f'Pantograph: {pantograph.__version__}')" 2>/dev/null || echo "Pantograph: FAILED"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || echo "PyTorch: not installed (needed for inference)"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy: FAILED"
python3 -c "import yaml; print(f'PyYAML: OK')" 2>/dev/null || echo "PyYAML: FAILED"

echo ""
echo "=== Setup complete ==="
echo "Next: sync Wayfinder files to $WORK_DIR"
echo "  scp -r src/ scripts/ configs/ $HOST:$WORK_DIR/"
echo "  scp models/NAV-002_step5000.pt $HOST:$WORK_DIR/models/"
echo "  scp -r data/lean_project/ $HOST:$WORK_DIR/data/lean_project/"
echo "  scp data/proof_network.db $HOST:$WORK_DIR/data/"
echo "  scp data/mathlib_benchmark_50.jsonl $HOST:$WORK_DIR/data/"
