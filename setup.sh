#!/bin/bash

# setup.sh

set -e 

echo "[SETUP] Starting setup process..."
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate base && echo 'Ready.'; exec </dev/tty
mkdir -p tools





echo "[SETUP] Extracting OpenROAD binary..."

#tar -xzf ../openroad_bin.tar.gz

#mkdir -p tools/OpenROAD

#mv OpenROAD/build tools/OpenROAD/





echo "[SETUP] Extracting Conda environment..."

#mkdir -p tools/my_env

#tar -xzf ../my_env.tar.gz -C tools/my_env





echo "[SETUP] Fixing Conda paths..."

source tools/my_env/bin/activate

conda-unpack




