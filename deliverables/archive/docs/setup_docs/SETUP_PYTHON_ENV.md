# Python Environment Setup for Publication Package Generation

**Date:** 2025-11-24

---

## Problem

The publication package generation scripts require:
- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `scipy`
- `scikit-learn`

These packages are not currently available in the default Python environment.

---

## Solution Options

### Option 1: Install in Existing Conda Environment (Recommended)

```bash
# Activate nih_project environment
conda activate nih_project

# Install required packages
pip install pandas matplotlib seaborn numpy scipy scikit-learn

# Verify installation
python -c "import pandas, matplotlib, seaborn, numpy, scipy, sklearn; print('✅ All packages installed')"
```

### Option 2: Create New Conda Environment

```bash
# Create new environment
conda create -n lpm-publication python=3.10 -y
conda activate lpm-publication

# Install packages
pip install pandas matplotlib seaborn numpy scipy scikit-learn

# Verify
python -c "import pandas, matplotlib, seaborn, numpy, scipy, sklearn; print('✅ All packages installed')"
```

### Option 3: Use System Python (if available)

```bash
# Check if system Python has packages
python3 -c "import pandas, matplotlib, seaborn, numpy, scipy, sklearn" && echo "✅ Available"

# If not, install (may require --user flag)
python3 -m pip install --user pandas matplotlib seaborn numpy scipy scikit-learn
```

---

## After Setup

Once packages are installed, you can generate the publication package:

```bash
cd lpm-evaluation-framework-v2
bash publication_package/run_publication_generation.sh
```

---

## Verification

The `run_publication_generation.sh` script will automatically:
1. Detect the correct Python environment
2. Verify required packages are available
3. Run all generation scripts
4. Provide logs for troubleshooting

