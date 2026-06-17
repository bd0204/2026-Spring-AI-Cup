#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -o infer.out
#PBS -e infer.err

cd $PBS_O_WORKDIR

# 避免讀到 ~/.local 裡面的套件
export PYTHONNOUSERSITE=1

# 使用符合目前 driver 的 CUDA module
module load CUDA/cuda-11.8/x86-64

# 直接指定 b12202057 conda environment 裡的 Python
PYTHON=/home/eegroup/eefrank/.conda/envs/b12202057/bin/python

echo "===== Environment check ====="
echo "Working directory: $(pwd)"
$PYTHON --version
$PYTHON -c "import sys; print('python executable:', sys.executable)"
$PYTHON -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
echo "============================="

# -u 讓輸出即時寫到 infer.out
$PYTHON -u inference.py
