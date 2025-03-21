#!/usr/bin/env python3

from datetime import datetime
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
    #"yolov5n.pt",
    #"yolov5s.pt",
    #"yolov5m.pt",
    #"yolov5l.pt",

    # "yolov8n.pt",
    # "yolov8s.pt",
    "yolov8m.pt",
    # "yolov8l.pt",

    #"yolov10n.pt",
    #"yolov10s.pt",
    #"yolov10m.pt",
    #"yolov10l.pt",

    #"yolo11n.pt",
    #"yolo11s.pt",
    #"yolo11m.pt",
    #"yolo11l.pt",

    #"yolo12n.pt",
    #"yolo12s.pt",
    #"yolo12m.pt",
    #"yolo12l.pt",
    ]
DEFAULT_BATCHES = [1, 2, 4, 8, 16, 32]
DEFAULT_DTYPES = [False, True]  # False: fp32, True: fp16

torch._logging.set_logs(dynamo=50, inductor=50)

def benchmark(
    bf,
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
    triton=True
):

    model_name = model

    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    model = YOLO(model)
    # using triton musa backend
    if triton:
        model.model = torch.compile(model.model, backend="inductor", mode="max-autotune")
        exit(0)
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
                if triton:
                    exported_model.model = torch.compile(exported_model.model, backend="inductor", mode="max-autotune")
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

    row = y[0]
    print_table_row(bf, model_name, row[2], half, batch, data, imgsz, triton, row[3], row[4], row[5])

    legend = "Benchmarks legend:  - ✅ Success  - ❎ Export passed but validation failed  - ❌️ Export failed"
    s = f"\nBenchmarks complete for {name} on {data} at imgsz={imgsz} ({dt:.2f}s)\n{legend}\n{df.fillna('-')}\n"
    with open("benchmarks.log", "a", errors="ignore", encoding="utf-8") as f:
        f.write(s)

    if verbose and isinstance(verbose, float):
        metrics = df[key].array  # values to compare to floor
        floor = verbose  # minimum metric floor to pass, i.e. = 0.29 mAP for YOLOv5n
        assert all(x > floor for x in metrics if pd.notna(x)), f"Benchmark failure: metric(s) < floor {floor}"

    return df

def print_table_head(bf):
    bf.write(
"""
<style>
table { border-collapse: collapse; }
th, td { border: 1px solid rgba(128,128,128,0.5); }
tr:nth-child(even) { background-color: rgba(99,99,99,0.3); }
</style>

""")
    bf.write(f"| Model            | Size             | half             | batch            | dataset          | imgsz            | compile(triton)  | mAP50-95(B)      | ms/im            | FPS              |\n")
    bf.write(f"| :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |\n")
    bf.flush()

def print_table_row(bf, model, size, half, batch, dataset, imgsz, triton, metrics_mAP50, ms_per_img, fps):
    bf.write(f"| {model:<17}| {size:<17}| {half:<17}| {batch:<17}| {dataset:<17}| {imgsz:<17}| {triton:<17}| {metrics_mAP50:<17}| {ms_per_img:<17}| {fps:<17}|\n")
    bf.flush()

def main(models: List[str], batches: List[int], dtypes: List[bool], dataset: str = "coco128.yaml", imgsz: int = 640, device: str = "cuda:0"):
    current_ts = datetime.now().strftime("%Y%m%d:%H%M")
    with open(f"benchmarks-table-{current_ts}.md", "w", errors="ignore", encoding="utf-8") as bf:
        print_table_head(bf)
        for half, model, batch, triton, in product(dtypes, models, batches, (False, True)):
            benchmark(bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch, half=half, int8=True, device=device, triton=triton)
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
