
## Setup

* Python: **3.10**

Install numpy:
```
pip install numpy==1.23
```

AiBook 1.2.0 latest packages：
```bash
# install torch & torch_musa
# NOTE: for M1000 developers, you need to get the torch_musa v1.3.2/m1000-dev branch and build pip wheel from source.
# see torch_musa for how to build & install torch & torch_musa from source.
# pip insall https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torch_musa-1.3.2-cp310-cp310-linux_aarch64.whl
# pip insall https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torch-2.2.0-cp310-cp310-linux_aarch64.whl

# install torchvision & torchaudio
pip insall https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip insall https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl

# install triton:
wget https://oss.mthreads.com/product-release/release_M1000_1.2.2/20250319/M1000_triton.tar.gz
tar -zxvf M1000_triton.tar.gz
pip install <extracted-whl-pkg>
```

(Optional):

```
MTT：      https://oss.mthreads.com/ai-product/aipc/mtt/mttransformer-20240402.dev67+g1196b79-py3-none-any.whl
vllm：     https://oss.mthreads.com/ai-product/aipc/mtt/vllm-0.4.2+musa-cp310-cp310-linux_aarch64.whl
```

## YOLO training & inference

This will show how to train YOLO model and run inference on MUSA GPU.

Install yolo pip package:

```sh
# pip install ultralytics
pip install setuptools wheel build
git clone git@sh-code.mthreads.com:m1000-sw/ultralytics.git
cd ultralytics
git checkout v8.3.96/musa
python -m build --wheel
pip install dist/<biult-whl-pkg>
```

Download sample imgs：
```sh
wget https://ultralytics.com/images/bus.jpg
```

### Sample code (yolo_train_infer.py)

```py
#!/usr/bin/env python3
import torch_compat as torch
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="coco8.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("./bus.jpg")
results[0].show()
```

### Sample code (yolo-perf.py)

You can just run `python yolo-perf.py` to perf benchmarks.
