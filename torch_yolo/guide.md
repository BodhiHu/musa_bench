# M1000 用户指南 - YOLO training & inference for PyTorch on MUSA

本指南旨在帮助用户在 M1000 设备上配置和使用 PyTorch完成YOLO training & inference，以充分利用 MUSA 加速能力。

## 1. 准备环境：PyTorch on MUSA

本节介绍如何在 M1000 环境中安装 PyTorch，以支持 MUSA 加速。

**前提条件**：

*   Python 版本：**3.10**
*   PyTorch MUSA版本：**2.2.0**
*   numpy 版本：**1.23.x**

**安装步骤**：

您可以选择以下两种方式安装 PyTorch 及其 MUSA 兼容组件：**在线安装** 或 **离线安装**。

### 1.1 安装方式一：在线安装

如果您可以访问互联网，推荐使用在线安装方式，命令如下：

```sh
pip install https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torch-2.2.0-cp310-cp310-linux_aarch64.whl
pip install https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torch_musa-1.3.2-cp310-cp310-linux_aarch64.whl
pip install https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
pip install https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip install numpy==1.23
```

AiBook 1.2.0 最新包：
```
torch_musa： https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torch_musa-1.3.2-cp310-cp310-linux_aarch64.whl
pytorch：    https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torch-2.2.0-cp310-cp310-linux_aarch64.whl
torchvision：https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
torchaudio： https://oss.mthreads.com/mt-ai-data/ci-release/torch_musa/AIBook/20250113/torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
MTT：        https://oss.mthreads.com/ai-product/aipc/mtt/mttransformer-20240402.dev67+g1196b79-py3-none-any.whl
vllm：       https://oss.mthreads.com/ai-product/aipc/mtt/vllm-0.4.2+musa-cp310-cp310-linux_aarch64.whl
triton：     https://oss.mthreads.com/ai-product/aipc/triton/triton-3.0.0-cp310-cp310-linux_aarch64.whl
```

### 1.2 安装方式二：离线安装

如果您无法访问互联网，或者网络环境不稳定，请使用离线安装方式。

**步骤 1：下载 whl 文件**

首先，使用 `wget` 命令下载所有必要的 whl 安装包：

```sh
wget https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torch-2.2.0-cp310-cp310-linux_aarch64.whl
wget https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torch_musa-1.3.2-cp310-cp310-linux_aarch64.whl
wget https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
wget https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/pytorch/1.3.2/torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
```

**步骤 2：离线安装 whl 文件**

下载完成后，使用 `pip install` 命令安装这些 whl 包：

```sh
pip install torch-2.2.0-cp310-cp310-linux_aarch64.whl
pip install torch_musa-1.3.2-cp310-cp310-linux_aarch64.whl
pip install torchaudio-2.2.2+cefdb36-cp310-cp310-linux_aarch64.whl
pip install torchvision-0.17.2+c1d70fe-cp310-cp310-linux_aarch64.whl
pip install numpy==1.23   # numpy默认安装2.2会有报错，请使用1.23.x版本
```

**注意**：

*   请务必安装使用Numpy 1.23.x版本，以避免与 PyTorch 2.2.0 版本可能存在的兼容性问题。

## 2. 安装：torch 兼容包

为了更好地兼容现有的 PyTorch 代码，您需要安装 `torch_compat` 兼容包。

**安装命令**：

```sh
pip install https://apollo-appstore-pre.tos-cn-beijing.volces.com/appstore/release/pip/torch_compat/torch_compat-1.0.0-cp310-cp310-linux_aarch64.whl
```

## 3. 最佳实践

本节提供了一些在 M1000 上使用 PyTorch 的最佳实践示例。

### 3.1 基础用例

以下是一个基础的 PyTorch 用例，展示了如何在 MUSA 设备上运行张量运算。

**示例代码 (test_cuda.py)**：

```py
#!/usr/bin/env python3
import torch_compat as torch        # 导入torch_compat包，并重命名为torch，其余保持不变

x = torch.randn(2, 3)

y1 = x.to("cuda:0")
print(f"y1 device: {y1.device}")

y2 = x.to(device="cuda:0")
print(f"y2 device: {y2.device}")

y3 = x.to("cpu") # Should not be changed
print(f"y3 device: {y3.device}")

y4 = x.to(0) # Assuming 0 is cuda device index, might need adjustment based on your setup if indices are different
print(f"y4 device: {y4.device}")

y5 = x.to(device=0) # Assuming 1 is cuda device index
print(f"y5 device: {y5.device}")

y6 = x.to(dtype=torch.float64, device="cuda:0") # Mixed args and kwargs
print(f"y6 device: {y6.device}, dtype: {y6.dtype}")

y7 = x.to(device="cuda:0", non_blocking=True) # More kwargs
print(f"y7 device: {y7.device}, non_blocking: {y7.is_pinned()}")

x = torch.tensor([1]).to('cuda:0')
print(x.device)

print(torch.__version__)
print(torch.cuda.is_available())

device = torch.device('cuda:0')
print(device)
```

**运行结果**：

正常运行上述代码，您将看到以下输出信息，表明 PyTorch 代码已在 MUSA 设备 (`musa:0`) 上成功运行：

```txt
$ ./test_cuda.py
y1 device: musa:0
y2 device: musa:0
y3 device: cpu
y4 device: musa:0
y5 device: musa:0
y6 device: musa:0, dtype: torch.float64
y7 device: musa:0, non_blocking: False
musa:0
2.2.0
True
musa:0
```

### 3.2 案例：YOLO训推一体

本节展示如何将流行的目标检测模型 YOLO 适配到 MUSA 环境上运行。

#### 3.2.1 准备工作

在运行 YOLO 示例之前，请确保完成以下准备工作：

**步骤 1：安装 ultralytics**

使用 pip 安装 YOLOv8 官方库 `ultralytics`：

```sh
pip install ultralytics
```

**步骤 2：下载测试图片**

下载一张用于测试的图片 `bus.jpg`：

```sh
wget https://ultralytics.com/images/bus.jpg
```

#### 3.2.2 示例代码 (yolo_train_infer.py)

以下是 YOLO 适配 MUSA 的示例代码，您可以保存为 `yolo_train_infer.py` 文件：

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

#### 3.2.3 运行与问题解决

**运行代码**：

直接运行上述 `yolo_train_infer.py` 脚本：

```sh
chmod +x yolo_train_infer.py
./yolo_train_infer.py
```

**问题排查**：

在首次运行时，您可能会遇到一些报错。以下列出了一些常见问题及其解决方案：

*   **[问题 1: Numpy 版本冲突](#4.1)**:  `NumPy 2.2.0 as it may crash. To support both 1.x and 2.x`
*   **[问题 2: isinstance 函数报错](#4.2)**: `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`

#### 3.2.4 成功运行结果

解决上述问题后，再次运行 `yolo_demo.py`，您应该能看到类似以下的运行结果，这表明 YOLO 模型已成功在 MUSA 上运行，并完成了训练、验证和推理过程：

```txt
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     98/100      1.18G     0.5199     0.5667      1.002         13        640: 100%|██████████| 1/1 [00:00<00:00,  2.04it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  5.84it/s
                   all          4         17      0.839      0.464      0.529       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100      1.18G     0.5653     0.4987      1.011         13        640: 100%|██████████| 1/1 [00:00<00:00,  2.17it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  5.87it/s
                   all          4         17      0.839      0.464      0.529       0.28

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100      1.18G     0.4481     0.4065     0.9326         13        640: 100%|██████████| 1/1 [00:00<00:00,  1.81it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  5.85it/s
                   all          4         17      0.856      0.461      0.556      0.321

100 epochs completed in 0.042 hours.
WARNING ⚠️ Skipping runs/detect/train10/weights/last.pt, not a valid Ultralytics model: isinstance() arg 2 must be a type, a tuple of types, or a union
WARNING ⚠️ Skipping runs/detect/train10/weights/best.pt, not a valid Ultralytics model: isinstance() arg 2 must be a type, a tuple of types, or a union

Validating runs/detect/train10/weights/best.pt...
Ultralytics 8.3.85 🚀 Python-3.10.16 torch-2.2.0 CUDA:0 (M1000, 31795MiB)
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  7.58it/s
                   all          4         17       0.65      0.783      0.913      0.652
                person          3         10      0.624        0.7       0.67      0.328
                   dog          1          1      0.527          1      0.995      0.796
                 horse          1          2      0.644          1      0.995      0.676
              elephant          1          2      0.551          1      0.828      0.322
              umbrella          1          1      0.552          1      0.995      0.895
          potted plant          1          1          1          0      0.995      0.895
Speed: 0.4ms preprocess, 18.3ms inference, 0.0ms loss, 3.2ms postprocess per image
Results saved to runs/detect/train10
Ultralytics 8.3.85 🚀 Python-3.10.16 torch-2.2.0 CUDA:0 (M1000, 31795MiB)
YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs
val: Scanning /data/repo/musa-yolov5/datasets/coco8/labels/val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%|██████████| 4/4 [00:0
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:00<00:00,  6.16it/s
                   all          4         17      0.649      0.783      0.913      0.653
                person          3         10      0.623        0.7      0.671       0.33
                   dog          1          1      0.527          1      0.995      0.796
                 horse          1          2      0.643          1      0.995      0.676
              elephant          1          2      0.551          1      0.828      0.323
              umbrella          1          1      0.551          1      0.995      0.895
          potted plant          1          1          1          0      0.995      0.895
Speed: 0.9ms preprocess, 16.2ms inference, 0.0ms loss, 4.2ms postprocess per image
Results saved to runs/detect/train102

image 1/1 /data/tmp/pip/torch_compat/bus.jpg: 640x480 4 persons, 1 bus, 33.4ms
Speed: 5.7ms preprocess, 33.4ms inference, 4.5ms postprocess per image at shape (1, 3, 640, 480)
```

### 3.3 案例：YOLO on GPU推理性能测试

本节展示如何评估 YOLO on GPU推理性能

#### 3.3.1 准备工作

在运行 YOLO 示例之前，请确保完成以下准备工作：

**步骤 1：安装 ultralytics**

使用 pip 安装 YOLOv8 官方库 `ultralytics`：

```sh
pip install ultralytics
```

#### 3.3.2 示例代码 (yolo_infer_perf.py)

以下是 YOLO 适配 MUSA 的示例代码，您可以保存为 `yolo_infer_perf.py` 文件：

```py
#!/usr/bin/env python3

import torch_compat as torch;
from ultralytics import YOLO
from ultralytics.cfg import TASK2DATA, TASK2METRIC
from ultralytics.engine.exporter import export_formats
from ultralytics.utils import ASSETS, LOGGER
from ultralytics.utils.checks import check_imgsz, check_yolo
from ultralytics.utils.files import file_size
from ultralytics.utils.torch_utils import select_device

import argparse
import time
from itertools import product
from typing import List
from pathlib import Path

DEFAULT_MODELS = [
    "yolov5n.pt",
    "yolov5s.pt",
    "yolov5m.pt",
    "yolov5l.pt",

    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",

    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",
    "yolov10l.pt",

    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",

    "yolo12n.pt",
    "yolo12s.pt",
    "yolo12m.pt",
    "yolo12l.pt",
    ]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_DTYPES = [False, True]  # False: fp32, True: fp16

def benchmark(
    model,
    data=None,
    imgsz=160,
    batch=1,
    half=False,
    int8=False,
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="-",
):
    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], "end2end", False)
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect

    y = []
    t0 = time.time()

    format_arg = format.lower()
    if format_arg:
        formats = frozenset(export_formats()["Argument"])
        assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."
    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):
        emoji, filename = "❌", None  # export defaults
        try:
            if format_arg and format_arg != format:
                continue

            # Export
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = model  # PyTorch format
            else:
                filename = model.export(
                    imgsz=imgsz, format=format, half=half, int8=int8, data=data, device=device, verbose=False
                )
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

            # Predict
            assert model.task != "pose" or i != 7, "GraphDef Pose inference is not supported"
            assert i not in {9, 10}, "inference not supported"  # Edge TPU and TF.js are unsupported
            assert i != 5 or platform.system() == "Darwin", "inference only supported on macOS>=10.13"  # CoreML
            if i in {13}:
                assert not is_end2end, "End-to-end torch.topk operation is not supported for NCNN prediction yet"
            exported_model.predict(ASSETS / "bus.jpg", imgsz=imgsz, device=device, half=half, verbose=False)

            # Validate
            results = exported_model.val(
                data=data, batch=batch, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # frames per second
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e) is AssertionError, f"Benchmark failure for {name}: {e}"
            LOGGER.warning(f"ERROR ❌️ Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])

    name = model.model_name
    dt = time.time() - t0
    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df.fillna('-')}\n"
    LOGGER.info(s)
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df

def main(models: List[str], batches: List[int], dtypes: List[bool], dataset: str = "coco128.yaml", imgsz: int = 640, device: str = "cuda:0"):
    for half, model, batch in product(dtypes, models, batches):
        print(f"Benchmarking model: {model} with half:{half} batch:{batch} dataset:{dataset} imgsz:{imgsz}")
        benchmark(model=model, data=dataset, imgsz=imgsz, batch=batch, half=half, int8=True, device=device)
        time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to benchmark.")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES, help="List of batch to test.")
    parser.add_argument("--dtypes", nargs="+", type=lambda x: x.lower() == 'true', default=DEFAULT_DTYPES, help="List of dtypes to test (False for fp32, True for fp16).")
    parser.add_argument("--dataset", default="coco128.yaml", help="Dataset configuration file (e.g., coco128.yaml).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default="cuda:0", help="Device to run on.")

    args = parser.parse_args()

    main(args.models, args.batches, args.dtypes, args.dataset, args.imgsz, args.device)
```

#### 3.3.3 运行与问题解决

**运行代码**：

直接运行上述 `yolo_infer_perf.py` 脚本：

```sh
chmod +x yolo_infer_perf.py
./yolo_infer_perf.py
```

**问题排查**：

在首次运行时，您可能会遇到一些报错。以下列出了一些常见问题及其解决方案：

*   **[问题 1: Numpy 版本冲突](#4.1)**:  `NumPy 2.2.0 as it may crash. To support both 1.x and 2.x`
*   **[问题 2: isinstance 函数报错](#4.2)**: `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union`

#### 3.3.4 成功运行结果

解决上述问题后，再次运行 `yolo_demo.py`，您应该能看到类似以下的运行结果，这表明 YOLO 模型已成功在 MUSA 上运行，并完成了训练、验证和推理过程：

```txt
$ ./yolo_infer_perf.py
Benchmarking model: yolo12n.pt with half:False batch:1 dataset:coco128.yaml imgsz:640
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo12n.pt to 'yolo12n.pt'...
  8%|████████████▌                                                                                                                                            | 448k/5.34M [00:00<00:03,  12%|██████████████████▊                                                                                                                                      | 672k/5.34M [00:00<00:04,  19%|████████████████████████████▉                                                                                                                           | 1.02M/5.34M [00:00<00:02,  27%|████████████████████████████████████████▉                                                                                                               | 1.44M/5.34M [00:00<00:01,  32%|████████████████████████████████████████████████▎                                                                                                       | 1.70M/5.34M [00:00<00:01,  36%|███████████████████████████████████████████████████████▍                                                                                                | 1.95M/5.34M [00:01<00:01,  41%|██████████████████████████████████████████████████████████████                                                                                          | 2.18M/5.34M [00:01<00:01,  45%|████████████████████████████████████████████████████████████████████▌                                                                                   | 2.41M/5.34M [00:01<00:01,  49%|███████████████████████████████████████████████████████████████████████████▏                                                                            | 2.64M/5.34M [00:01<00:01,  54%|█████████████████████████████████████████████████████████████████████████████████▍                                                                      | 2.86M/5.34M [00:01<00:01,  58%|████████████████████████████████████████████████████████████████████████████████████████▌                                                               | 3.11M/5.34M [00:01<00:01,  62%|██████████████████████████████████████████████████████████████████████████████████████████████▌                                                         | 3.32M/5.34M [00:01<00:00,  67%|█████████████████████████████████████████████████████████████████████████████████████████████████████▍                                                  | 3.56M/5.34M [00:01<00:00,  71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                            | 3.78M/5.34M [00:01<00:00,  75%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋                                      | 3.99M/5.34M [00:02<00:00,  83%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊                         | 4.45M/5.34M [00:02<00:00,  88%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉                  | 4.70M/5.34M [00:02<00:00,  93%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▋           | 4.94M/5.34M [00:02<00:00,  97%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉     | 5.16M/5.34M [00:02<00:00, 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.34M/5.34M [00:02<00:00, 1.98MB/s]
val: Scanning /data/repo/musa-yolov5/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 128/128 [00:06<00:00, 19.14it/s]
                   all        128        929      0.692      0.616      0.683      0.528
Speed: 0.7ms preprocess, 30.8ms inference, 0.0ms loss, 4.0ms postprocess per image
Setup complete ✅ (12 CPUs, 31.0 GB RAM, 60.3/125.2 GB disk)

Benchmarks complete for yolo12n.pt on coco128.yaml at imgsz=640 (10.60s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
    Format Status❔  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)    FPS
0  PyTorch       ✅        5.3               0.5277                   30.84  32.43

Benchmarking model: yolo12n.pt with half:False batch:2 dataset:coco128.yaml imgsz:640
val: Scanning /data/repo/musa-yolov5/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 64/64 [00:04<00:00, 15.98it/s]
                   all        128        929      0.697      0.619      0.684      0.529
Speed: 0.5ms preprocess, 19.1ms inference, 0.0ms loss, 5.7ms postprocess per image
Setup complete ✅ (12 CPUs, 31.0 GB RAM, 60.3/125.2 GB disk)

Benchmarks complete for yolo12n.pt on coco128.yaml at imgsz=640 (4.88s)
Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed
    Format Status❔  Size (MB)  metrics/mAP50-95(B)  Inference time (ms/im)    FPS
0  PyTorch       ✅        5.3               0.5285                   19.06  52.46
```

## 4. 常见问题 FAQ

本节汇总了用户在使用过程中可能遇到的常见问题及解决方案，以便快速排查和解决问题。

### <span id="4.1">4.1 Numpy 版本冲突</span>：NumPy 2.2.0 可能导致程序崩溃

**问题描述**：

运行程序时，出现类似以下错误信息，提示 NumPy 版本 2.2.0 可能导致程序崩溃，建议降级到 NumPy 1.x 版本。

```txt
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.

If you are a user of the module, the easiest solution will be to
downgrade to 'numpy<2' or try to upgrade the affected module.
We expect that some modules will need time to support NumPy 2.

Traceback (most recent call last):  File "/data/tmp/pip/torch_compat/./yolo_demo.py", line 3, in <module>
    import torch_compat as torch
  ... (省略部分 traceback) ...
/home/mt/miniconda3/envs/ultralytics/lib/python3.10/site-packages/torch/nn/modules/transformer.py:20: UserWarning: Failed to initialize NumPy:
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.2.0 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
Some module may need to rebuild instead e.g. with 'pybind11>=2.12'.
```

**解决方案**：

安装指定的 `numpy==1.23.5` 版本。

```sh
pip install numpy==1.23.5       # 安装numpy 1.23.x 版本
```

### <span id="4.2">4.2 `isinstance` 函数报错</span>：TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union

**问题描述**：

运行 YOLO 等模型时，出现 `TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union` 错误，指向 `ultralytics/utils/torch_utils.py` 文件。

```txt
$ ./yolo_demo.py
Traceback (most recent call last):
  File "/data/tmp/pip/torch_compat/./yolo_demo.py", line 10, in <module>
    train_results = model.train(
  ... (省略部分 traceback) ...
  File "/home/mt/miniconda3/envs/ultralytics/lib/python3.10/site-packages/ultralytics/utils/torch_utils.py", line 166, in select_device
    if isinstance(device, torch.device) or str(device).startswith("tpu"):
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

**解决方案**：

根据错误提示，定位到报错文件 `ultralytics/utils/torch_utils.py` 和报错行号（例如第 166 行）。

1.  **打开报错文件**：使用编辑器打开 `ultralytics/utils/torch_utils.py` 文件。
2.  **定位报错代码**：找到报错行，通常是类似 `isinstance(device, torch.device)` 的代码。
3.  **修改源码**：将 `isinstance(device, torch.device)` 修改为 `isinstance(device, torch._C.device)`。

**修改示例**：

将以下代码：

```python
if isinstance(device, torch.device) or str(device).startswith("tpu"):
```

修改为：

```python
if isinstance(device, torch._C.device) or str(device).startswith("tpu"):
```

**完成修改后，重新运行程序即可解决该问题。**

---
