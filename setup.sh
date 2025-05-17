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

conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# Install required packages
echo "Installing required packages..."
conda install -y numpy matplotlib scikit-learn
conda install -y pandas scipy
conda install -y seaborn rpy2
if ! conda install -y lifelines; then
    echo "lifelines not found on conda; installing with pip"
    pip install lifelines
fi
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
echo "=== [3/4] Installing R package 'xCell' and reference data ================"

ENV_NAME="ici_sex"

conda run -n "$ENV_NAME" R --vanilla -s <<'RSCRIPT'
options(repos = c(CRAN="https://cloud.r-project.org"))
.libPaths(c(file.path(Sys.getenv("CONDA_PREFIX"), "lib/R/library"), .libPaths()))

if (!requireNamespace("devtools", quietly = TRUE))
    install.packages("devtools")
if (!requireNamespace("curl", quietly = TRUE))
    install.packages("curl")

# (Re)install xCell from GitHub only if not already present
if (!requireNamespace("xCell", quietly = TRUE)) {
    devtools::install_github("dviraran/xCell", upgrade = "never", quiet = TRUE)
}

library(curl)
library(xCell)

data_url  <- "https://raw.githubusercontent.com/dviraran/xCell/master/data/xCell.data.rda"
data_dir  <- file.path(.libPaths()[1], "xCell", "data")
data_path <- file.path(data_dir, "xCell.data.rda")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

needs_download <- !file.exists(data_path) || file.info(data_path)$size < 1e5

if (needs_download) {
  message("Downloading xCell reference data ...")
  tries <- 0; ok <- FALSE
  while (tries < 5 && !ok) {
    tries <- tries + 1
    try({
      curl_download(url = data_url,
                    destfile = data_path,
                    quiet = FALSE,
                    handle = curl::new_handle("retry" = 5))
      ok <- TRUE
    }, silent = TRUE)
    if (!ok) {
      message("  -> Retry #", tries, " failed; sleeping ", tries * 3, " s")
      Sys.sleep(tries * 3)
    }
  }
  if (!ok) stop("Failed to retrieve xCell.data.rda after ", tries, " attempts")
} else {
  message("xCell reference data already present – skipping download")
}

load(data_path)
stopifnot(exists("xCell.data"))
message("xCell data loaded – ", length(xCell.data$genes), " reference genes available.")
RSCRIPT

echo "Environment setup complete!" 
