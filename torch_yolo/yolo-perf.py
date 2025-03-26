#!/usr/bin/env python3

import os
import numpy as np
import logging
import platform
import torch
import torch_musa
from datetime import datetime
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
from typing import Literal

try:
    import torch_musa.cuda_compat
except Exception as exc:
    print("WARN: could not import torch_musa.cuda_compat: ", exc)
    print("WARN: fall back to manual cuda compat")
    # fallback to manual cuda compat
    del torch.cuda
    torch.cuda = torch.musa

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
DEFAULT_BATCHES = [
    1, #2, 4, 8, 16, 32
]
DEFAULT_DTYPES = [
    # False, # fp32
    True   # fp16
]
TRITON_TOGGLES = [True]
DEFAULT_DEVICE = "cuda:0"
# DEFAULT_DEVICE = "musa:0"
CWD = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to benchmark.")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES, help="List of batch to test.")
    parser.add_argument("--dtypes", nargs="+", type=lambda x: x.lower() == 'true', default=DEFAULT_DTYPES, help="List of dtypes to test (False for fp32, True for fp16).")
    parser.add_argument("--dataset", default="coco128.yaml", help="Dataset configuration file (e.g., coco128.yaml).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device to run on.")
    parser.add_argument("-tt", "--triton-toggles", action="store_true", help="If also perf without triton.")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode.")
    parser.add_argument("-v", "--verify-musa", action="store_true", help="verify musa env only.")

    args = parser.parse_args()

    if args.triton_toggles:
        TRITON_TOGGLES.insert(0, False)
    return args

args = parse_args()


if args.debug:
    logging.basicConfig(
        level=logging.DEBUG, 
        # format="%(asctime)s - %(levelname)s - %(message)s",
    )
    # Ensure the root logger captures debug messages
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    torch._logging.set_logs(
        all                 = logging.DEBUG,
        dynamo              = logging.DEBUG,
        aot                 = logging.DEBUG,
        dynamic             = logging.DEBUG,
        inductor            = logging.DEBUG,
        distributed         = logging.DEBUG,
        onnx                = logging.DEBUG,

        bytecode            = True,
        aot_graphs          = True,
        aot_joint_graph     = True,
        ddp_graphs          = True,
        graph               = True,
        graph_code          = True,
        graph_breaks        = True,
        graph_sizes         = True,
        guards              = True,
        recompiles          = True,
        recompiles_verbose  = True,
        trace_source        = True,
        trace_call          = True,
        output_code         = True,
        schedule            = True,
        perf_hints          = True,
        post_grad_graphs    = True,
        onnx_diagnostics    = True,
        fusion              = True,
        overlap             = True,
    )

def verify_musa(choice: Literal["simple", "naive-yolo"], verify_only = False):
    if choice == "simple":
        def simple_fn(x):
            return x.sin()

        x = torch.randn(10, device="cuda")
        compiled_fn = torch.compile(simple_fn, backend="inductor")
        compiled_fn(x)
        print("INFO: >> ✔ simple_fn test pass")

    if choice == "naive-yolo":
        model_path = "yolov8m.pt"
        model_data = torch.load(model_path, map_location="cuda")
        model = model_data["model"].float().eval()
        print(f">> model = {type(model)}\n")
        model = torch.compile(model, backend="inductor", mode="max-autotune")
        input_tensor = np.load('img_0.npy')
        input_tensor = torch.from_numpy(input_tensor).to("cuda")
        with torch.no_grad():
            model(input_tensor)
            print("INFO: >> ✔ yolov8m naive test pass")

    if verify_only:
        exit(0)

if args.verify_musa:
    # verify_musa("simple")
    verify_musa("naive-yolo", verify_only=False)
    exit(0)


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

    def may_load_triton_model(model: YOLO):
        # using triton musa backend
        if triton:
            print(f"INFO: using triton backend for model {type(model.model)} on {device}")
            # model.model.to(device)
            model.model.to(device).eval()
            model.model = torch.compile(
                model.model,
                backend="inductor",
                mode="max-autotune"
            )

    model = YOLO(model)
    may_load_triton_model(model)

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
            assert i == 0

            # Export
            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = model  # PyTorch format
            else:
                filename = model.export(
                    imgsz=imgsz, format=format, half=half, int8=int8, data=data, device=device, verbose=False
                )
                exported_model = YOLO(filename, task=model.task)
                may_load_triton_model(exported_model)

                assert suffix in str(filename), "export failed"
            emoji = "❎"  # indicates export succeeded

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


def main(models: List[str], batches: List[int], dtypes: List[bool], dataset: str = "coco128.yaml", imgsz: int = 640, device: str = DEFAULT_DEVICE):
    current_ts = datetime.now().strftime("%Y%m%d:%H%M")
    os.makedirs(f"{CWD}/benchmarks/", exist_ok=True)
    with open(f"{CWD}/benchmarks/benchmarks-table-{current_ts}.md", "w", errors="ignore", encoding="utf-8") as bf:
        print_table_head(bf)
        for half, model, batch, triton, in product(dtypes, models, batches, TRITON_TOGGLES):
            benchmark(bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch, half=half, int8=True, device=device, triton=triton)
            time.sleep(5)


if __name__ == "__main__":
    main(args.models, args.batches, args.dtypes, args.dataset, args.imgsz, args.device)
