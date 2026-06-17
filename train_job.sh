#!/bin/bash
#PBS -l select=1:ncpus=8:ngpus=1
#PBS -q ee
#PBS -o train.out
#PBS -e train.err

cd $PBS_O_WORKDIR

# 避免讀到 ~/.local 裡面的套件，防止 numpy / pandas / torch 混用
export PYTHONNOUSERSITE=1

# 使用符合目前 driver 的 CUDA module
module load CUDA/cuda-11.8/x86-64

# 直接指定 b12202057 conda environment 裡的 Python
PYTHON=/home/eegroup/eefrank/.conda/envs/b12202057/bin/python

echo "===== Environment check ====="
echo "Working directory: $(pwd)"
$PYTHON --version
$PYTHON -c "import sys; print('python executable:', sys.executable)"
$PYTHON -c "import numpy; print('numpy', numpy.__version__, numpy.__file__)"
$PYTHON -c "import pandas; print('pandas', pandas.__version__, pandas.__file__)"
$PYTHON -c "import sklearn; print('sklearn', sklearn.__version__, sklearn.__file__)"
$PYTHON -c "import torch; print('torch', torch.__version__); print('torch cuda', torch.version.cuda); print('cuda available', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
echo "============================="

# 訓練 v41 ShuttleNet 5-fold（含 fold-level resume：PBS 12hr walltime 中斷後重新 qsub 會接續未完成的 fold）
# -u = unbuffered output，方便即時看到 loss
$PYTHON -u train.py
