#!/bin/bash
# Try different Python environments to find one with pandas

echo "Checking Python environments..."

# Try standard python3
if python3 -c "import pandas" 2>/dev/null; then
    echo "✅ Standard python3 has pandas"
    python3 --version
    python3 -c "import pandas; print('pandas:', pandas.__version__)"
    exit 0
fi

# Try with PYTHONPATH
if PYTHONPATH=src python3 -c "import sys; sys.path.insert(0, 'src'); import pandas" 2>/dev/null; then
    echo "✅ python3 with PYTHONPATH=src has pandas"
    exit 0
fi

# Check for conda
if command -v conda &> /dev/null; then
    echo "Conda available, checking environments..."
    for env in $(conda env list | grep -v "^#" | awk '{print $1}' | grep -v "^$"); do
        if conda run -n "$env" python -c "import pandas" 2>/dev/null; then
            echo "✅ Conda environment '$env' has pandas"
            exit 0
        fi
    done
fi

echo "❌ No Python environment with pandas found"
exit 1
