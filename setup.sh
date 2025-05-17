#!/bin/bash

# Set the project directory
PROJECT_DIR="/project/orien/data/aws/24PRJ217UVA_IORIG/Understanding_How_Sex_Impacts_Recovery_From_Tumors"
CONDA_DIR="$PROJECT_DIR/miniconda3"

# Download and install Miniconda if not already installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Downloading Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    
    echo "Installing Miniconda..."
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR
    
    # Initialize conda for bash shell
    echo "Initializing conda..."
    source "$CONDA_DIR/etc/profile.d/conda.sh"
    
    # Add conda initialization to .bashrc if not already present
    if ! grep -q "conda initialize" ~/.bashrc; then
        $CONDA_DIR/bin/conda init bash
    fi
    
    # Clean up
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Make sure conda is initialized
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Create new environment if it doesn't exist
if ! conda env list | grep -q "ici_sex"; then
    echo "Creating ici_sex environment..."
    conda create -y -n ici_sex python=3.9
fi

# Activate the environment
conda activate ici_sex

# Install required packages
echo "Installing required packages..."
conda install -y numpy matplotlib scikit-learn
conda install -y pandas scipy
conda install -y seaborn lifelines rpy2
conda install -y statsmodels

# Install R and Bioconductor packages through conda
echo "Installing R packages..."
conda install -y -c conda-forge r-base r-essentials
conda install -y -c conda-forge r-devtools r-remotes
conda install -y -c bioconda bioconductor-gsva
conda install -y -c bioconda bioconductor-gseabase
conda install -y -c bioconda bioconductor-biobase
conda install -y -c bioconda bioconductor-summarizedexperiment
conda install -y -c bioconda bioconductor-biomart
conda install -y -c bioconda bioconductor-org.hs.eg.db

# Install xCell and prepare data
conda run -n ici_sex R -e '
  .libPaths(paste0(Sys.getenv("CONDA_PREFIX"), "/lib/R/library"));
  if (!require("devtools")) install.packages("devtools", lib=.libPaths()[1]);
  
  # Install xCell from GitHub
  devtools::install_github("dviraran/xCell", lib=.libPaths()[1], force=TRUE);
  
  # Download and save xCell data
  library(xCell);
  data_url <- "https://raw.githubusercontent.com/dviraran/xCell/master/data/xCell.data.rda";
  data_path <- file.path(.libPaths()[1], "xCell", "data", "xCell.data.rda");
  dir.create(dirname(data_path), recursive=TRUE, showWarnings=FALSE);
  download.file(data_url, data_path);
  
  # Verify data
  load(data_path);
  if (exists("xCell.data")) {
    print("xCell data loaded successfully");
    print(paste("Number of reference genes:", length(xCell.data$genes)));
  } else {
    stop("Failed to load xCell data");
  }
'

echo "Environment setup complete!" 
