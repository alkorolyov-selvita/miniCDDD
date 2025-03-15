#!/bin/bash
ENV_NAME="cddd"

# Function to check the source of `conda`
conda_installed() {
    local conda_path
    conda_path=$(which conda 2>/dev/null)

    if [[ -z "$conda_path" ]]; then
        echo "Conda is not installed."
        return 1
    else
      echo "Conda is already installed and located in ${conda_path}"
      return 0
    fi
}

# Download and install Miniforge
install_conda() {
    # Download
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-"$(uname)"-"$(uname -m)".sh -b

    # Source Conda and Mamba initialization scripts
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    source "${HOME}/miniforge3/etc/profile.d/mamba.sh"

    conda init
    conda config --set auto_activate_base false
    rm Miniforge3-"$(uname)"-"$(uname -m)".sh
}

# install dependencies
sudo apt-get install curl git-lfs -y

# Check if Miniforge is already installed
if conda_installed; then
    echo "Skipping Conda installation."
else
    echo "Downloading and installing Miniforge..."
    install_conda
fi

# install lfs and fetch encoder model
git lfs install
git lfs fetch --include="models/encoder_model.h5"
git lfs checkout "models/encoder_model.h5"


conda create -n "$ENV_NAME" python=3.10 notebook ipykernel zstandard pandas=2.1.4 joblib tqdm -y
conda run -n "$ENV_NAME" python -m ipykernel install --name "$ENV_NAME" --user
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt


