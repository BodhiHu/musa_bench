#!/bin/bash

ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$ENV_DIR/../../ultralytics:$PYTHONPATH"
export PYTHONPATH="$ENV_DIR/../../NeuroTrim:$PYTHONPATH"

# export TORCH_COMPILE_DEBUG=1
# export TORCHINDUCTOR_DUMP=1
# export TORCH_LOGS=+all
