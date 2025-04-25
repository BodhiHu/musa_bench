#!/bin/bash

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ENV_DIR/../../ultralytics:$PYTHONPATH"
export PYTHONPATH="$ENV_DIR/../../NeuroTrim:$PYTHONPATH"
# export LD_LIBRARY_PATH=/home/bodhi/bodhi/tmp/msys:$LD_LIBRARY_PATH

export ENABLE_MUSA_MMA=1

# export TORCH_COMPILE_DEBUG=1
# export TORCHINDUCTOR_DUMP=1
# export TORCH_LOGS=+all

# Set GPU freq
sudo ac_tool register w32 -a 0x280FD248 -v 1400
