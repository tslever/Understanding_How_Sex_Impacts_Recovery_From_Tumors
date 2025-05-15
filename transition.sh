#!/bin/bash

# Transition script to help users move to the new directory structure

echo "=== ICI Sex Project Directory Transition ==="
echo "This script will help you transition to the new directory structure."
echo ""

# Check if the user is in the right directory
if [ ! -d "project_root" ]; then
    echo "Error: project_root directory not found."
    echo "Please run this script from the original project directory."
    exit 1
fi

echo "The project has been reorganized into a cleaner structure."
echo "All code and data have been moved to the project_root directory."
echo ""
echo "To use the new structure:"
echo ""
echo "1. Change to the project_root directory:"
echo "   cd project_root"
echo ""
echo "2. Activate the conda environment:"
echo "   source miniconda3/etc/profile.d/conda.sh"
echo "   conda activate ici_sex"
echo ""
echo "3. Run analyses from the project root, for example:"
echo "   python src/icb_analysis/icb_main.py"
echo ""
echo "The new directory structure is as follows:"
echo ""
echo "project_root/"
echo "├── data/                  # Data directory"
echo "│   └── processed/         # Processed data files"
echo "├── docs/                  # Documentation"
echo "├── output/                # Analysis outputs and results"
echo "├── src/                   # Source code"
echo "│   ├── cd8_analysis/      # CD8+ T cell analysis"
echo "│   ├── data_processing/   # Data processing utilities"
echo "│   ├── icb_analysis/      # Immune checkpoint blockade analysis"
echo "│   ├── immune_analysis/   # Immune signature analysis"
echo "│   ├── sex_stratified/    # Sex-stratified analysis"
echo "│   └── utils/             # Shared utility functions"
echo "├── requirements.txt       # Python dependencies"
echo "└── setup.sh               # Environment setup script"
echo ""
echo "For more information, see the README.md file in the project_root directory."
echo ""

# Ask if the user wants to change to the new directory
read -p "Would you like to change to the new directory now? (y/n): " answer
if [[ $answer == "y" || $answer == "Y" ]]; then
    cd project_root
    echo "Changed to project_root directory."
    echo "You can now activate the conda environment and run analyses."
else
    echo "You can change to the new directory later with 'cd project_root'."
fi 