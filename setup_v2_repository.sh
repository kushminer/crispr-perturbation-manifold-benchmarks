#!/bin/bash
# Setup script for creating v2 resampling-enabled repository

set -e  # Exit on error

echo "============================================================"
echo "Sprint 11 - V2 Repository Setup"
echo "============================================================"
echo ""

# Get the parent directory
PARENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="perturbench-resampling"
V2_DIR="${PARENT_DIR}/${REPO_NAME}"

echo "Parent directory: ${PARENT_DIR}"
echo "Repository name: ${REPO_NAME}"
echo "Target directory: ${V2_DIR}"
echo ""

# Check if directory already exists
if [ -d "${V2_DIR}" ]; then
    echo "⚠️  Directory ${V2_DIR} already exists."
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing directory..."
        rm -rf "${V2_DIR}"
    else
        echo "Aborting. Please remove or rename the existing directory."
        exit 1
    fi
fi

echo "Creating v2 repository directory..."
mkdir -p "${V2_DIR}"

echo "Copying files from evaluation_framework..."
cd "$(dirname "${BASH_SOURCE[0]}")"

# Copy all files and directories, preserving structure
rsync -av \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results/*' \
    --exclude='*.log' \
    --exclude='.pytest_cache' \
    --exclude='.DS_Store' \
    . "${V2_DIR}/"

# Update README for v2
if [ -f "${V2_DIR}/V2_RESAMPLING_README.md" ]; then
    echo "Updating README for v2..."
    mv "${V2_DIR}/V2_RESAMPLING_README.md" "${V2_DIR}/README.md"
fi

echo ""
echo "Initializing git repository..."
cd "${V2_DIR}"

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
else
    echo "Git repository already exists"
fi

# Create .gitignore if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo "Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/
.pytest_cache/
.coverage
htmlcov/

# Virtual environments
venv/
venv_*/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# R
.Rhistory
.RData
.Ruserdata

# Results and outputs
results/
*.log

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.bak
EOF
fi

echo ""
echo "Preparing initial commit..."

# Add all files
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "No changes to commit"
else
    git commit -m "Initial commit: v1 baseline (Sprint 11 - before resampling enhancements)"
    echo "✅ Initial commit created"
fi

echo ""
echo "============================================================"
echo "Repository Setup Complete!"
echo "============================================================"
echo ""
echo "Repository location: ${V2_DIR}"
echo ""
echo "Next steps:"
echo ""
echo "1. Create GitHub repository:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: ${REPO_NAME}"
echo "   - Description: 'Resampling-enabled evaluation engine (v2) for linear perturbation prediction'"
echo "   - Do NOT initialize with README, .gitignore, or license"
echo "   - Click 'Create repository'"
echo ""
echo "2. Push to GitHub:"
echo "   cd ${V2_DIR}"
echo "   git remote add origin <your-github-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Verify setup:"
echo "   cd ${V2_DIR}"
echo "   PYTHONPATH=src python verify_sprint11_implementation.py"
echo ""
echo "============================================================"

