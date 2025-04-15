#!/usr/bin/env python3

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.abspath(f"{current_dir}/.."))
sys.path.append(os.path.abspath(f"{current_dir}/../.."))

import copy
from functools import partial
import inspect
import re
import traceback
import numpy as np
import logging
import platform
import torch
import torch_musa
from torch import fx
from tqdm import tqdm
from datetime import datetime
import ultralytics
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
from torch.ao.quantization import quantize_fx, HistogramObserver, MinMaxObserver, QConfig
from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from musa_bench.utils.general import get_device_name

try:
    import torch_musa.cuda_compat
except Exception as exc:
    print("WARN: could not import torch_musa.cuda_compat: ", exc)
    print("WARN: fall back to manual cuda compat")
    # fallback to manual cuda compat
    del torch.cuda
    torch.cuda = torch.musa
    torch.cuda.CUDAGraph = torch.musa.MUSAGraph


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
    1,
    # 2, 4, 8, 16, 32
]
DEFAULT_DTYPES = [
    "fp32",
    "fp16",
    # "int8"
]
TRITON_TOGGLES = [True]
GRAPH_TOGGLES = [True, False]
DEFAULT_DEVICE = "cuda:0"
# DEFAULT_DEVICE = "musa:0"
COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
DEFAULT_ROUNDS = 30
CWD = os.path.dirname(os.path.abspath(__file__))
DEVICE_NAME = get_device_name()

def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to benchmark.")
    parser.add_argument("--batches", type=lambda s: list(map(int, s.split(","))), default=DEFAULT_BATCHES, help="List of batch to test.")
    parser.add_argument("--dataset", default="coco128.yaml", help="Dataset configuration file (e.g., coco128.yaml).")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Device to run on.")
    parser.add_argument("--dtypes", default=DEFAULT_DTYPES, type=lambda s: s.split(","), help="dtypes (e.g. int8,fp16,fp32)")
    parser.add_argument("--qmethod", default="neuro-fx", type=str, help="quant method (dynamic, static, fx, neuro-fx)")
    parser.add_argument("--cmp-modes", default=COMPILE_MODES, type=lambda s: s.split(","), help="compile modes")
    parser.add_argument("-tt", "--triton-toggles", action="store_true", help="If also perf without triton.")
    parser.add_argument("--no-compile", action="store_true", help="trun off compiling.")
    parser.add_argument("-d", "--debug", action="store_true", help="turn on debug mode.")
    parser.add_argument("-v", "--verify-musa", action="store_true", help="verify musa env.")
    parser.add_argument("-ng", "--no-graph", action="store_true", help="turn off cuda/musa graph.")
    parser.add_argument("--rounds", default=DEFAULT_ROUNDS, type=int, help="rounds to run")
    parser.add_argument("-p", "--profiling", action="store_true", help="turn on perf profiling mode.")
    parser.add_argument("-pm", "--print-model", action="store_true", help="print model info.")
    parser.add_argument("-pp", "--print-layers", action="store_true", help="print model layers.")

    args = parser.parse_args()

    global TRITON_TOGGLES
    if args.triton_toggles:
        TRITON_TOGGLES.append(False)
    if args.no_compile:
        TRITON_TOGGLES = [False]

    global GRAPH_TOGGLES
    if args.no_graph:
        GRAPH_TOGGLES = [False]

    return args

args = parse_args()


if args.debug:
    torch._logging.set_logs(
        # all                 = logging.DEBUG,
        # dynamo              = logging.DEBUG,
        # aot                 = logging.DEBUG,
        # dynamic             = logging.DEBUG,
        # inductor            = logging.DEBUG,
        # distributed         = logging.DEBUG,
        # onnx                = logging.DEBUG,

        # bytecode            = True,
        # aot_graphs          = True,
        # aot_joint_graph     = True,
        # ddp_graphs          = True,
        # graph               = True,
        # graph_code          = True,
        # graph_breaks        = True,
        # graph_sizes         = True,
        # guards              = True,
        recompiles          = True,
        # recompiles_verbose  = True,
        # trace_source        = True,
        # trace_call          = True,
        output_code         = True,
        # schedule            = True,
        # perf_hints          = True,
        # post_grad_graphs    = True,
        # onnx_diagnostics    = True,
        # fusion              = True,
        # overlap             = True,
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
    dtype="fp32",
    device="cpu",
    verbose=False,
    eps=1e-3,
    format="-",
    triton=True,
    compile_mode="default",
    warmup=False,
    use_graph=False,
    profiling = False,
):

    model_name = model

    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)

    def print_model_layers_info(_model):
        if args.print_layers:
            print("\n==== model layers info ==========================================================")
        traceable_clszs   = set()
        untraceable_clszs = set()
        for name, module in _model.named_modules():
            is_traceable = False
            try:
                torch.fx.symbolic_trace(module)
                is_traceable = True
                traceable_clszs.add(type(module))
            except Exception as exc:
                untraceable_clszs.add(type(module))

            if args.print_layers:
                print(f"{name:<40} :: {str(type(module)):<50} | is_traceable {'✔ ' if is_traceable else '❌'} |")

        print("\ntraceable_clszs =\n"   + '\n'.join(str(item) for item in traceable_clszs))
        print("\nuntraceable_clszs =\n" + '\n'.join(str(item) for item in untraceable_clszs))
        print("\n")

        return (traceable_clszs, untraceable_clszs)

    def may_compile_and_quant(model: YOLO):
        # fuse conv2d and bn
        model.fuse()

        if triton:
            if type(model.model) == str:
                print(f"WARN: will not call compile for model.model of type = {type(model.model)}")
                return

            print(f"INFO: compile model {type(model.model)} on {device}, mode = {compile_mode}")
            model.model.to(device).eval()
            model.model = torch.compile(
                model.model,
                backend="inductor",
                mode=compile_mode
            )
            # if warmup:
            #     input_tensor = np.load('img_0.npy')
            #     input_tensor = torch.from_numpy(input_tensor).to("cuda")
            #     with torch.no_grad():
            #         model.model(input_tensor)

        quant_method = args.qmethod
        if dtype == "int8":
            print(f"INFO: quantize model {type(model.model)} to {dtype} using {quant_method} method")
            model.model.to("cpu").eval()

            backend = "qnnpack"
            input_tensor = np.load('img_0.npy')
            input_tensor = torch.from_numpy(input_tensor)
            print_mod = args.print_model
            print_layers = args.print_layers

            _model = model.model

            if print_mod:
                print("==== BEFORE QUANT (**after fuse**) ===============================================")
                print(_model)

            # Dynamic quant:
            if quant_method == "dynamic":
                model.model = torch.quantization.quantize_dynamic(
                    _model,
                    dtype=torch.qint8
                )

            # Static quant:
            if quant_method == "static":
                # Quant to int8
                # torch.quantization.get_default_qconfig("qnnpack")
                qcfg = QConfig(
                    activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.qint8),
                    weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
                )
                quant_list = [
                    torch.nn.modules.conv.Conv2d,
                    torch.nn.modules.pooling.MaxPool2d,
                    # torch.mul,
                    # torch.add,
                    # torch.nn.Sigmoid,
                    # torch.split,
                ]
                full_quant = False
                calibration_on = False

                if full_quant:
                    quant_list = None

                _model.qconfig = qcfg
                for name, module in _model.named_modules():
                    if quant_list is None or type(module) in quant_list:
                        module.qconfig = qcfg

                model_prepared = torch.quantization.prepare(_model, allow_list=quant_list)

                if calibration_on:
                    from musa_bench.utils.quant_utils import calibrate
                    task = "val"  # path to train/val/test images
                    stride = max(int(_model.stride.max()), 32)
                    single_cls=False
                    pad = 0.5
                    rect = True
                    workers = 8
                    calib_data = yaml_load(f"cfg/datasets/coco128.yaml")
                    dataloader = create_dataloader(
                        "datasets/coco128/" + calib_data[task],
                        imgsz,
                        batch,
                        stride,
                        single_cls,
                        pad=pad,
                        rect=rect,
                        workers=workers,
                        prefix=colorstr(f"{task}: "),
                    )[0]
                    calibrate(model_prepared, dataloader)

                model_prepared(input_tensor)
                qmodel = torch.quantization.convert(model_prepared, allow_list=quant_list)
                if hasattr(_model, "stride"):
                    qmodel.stride = _model.stride
                qmodel.names = _model.module.names if hasattr(_model, "module") else _model.names
                # from musa_bench.utils.replace_quant_modules import replace_nnq_modules
                # for name, module in qmodel.named_modules():
                #     if type(module) in quant_list:
                #         wrapped_m = replace_nnq_modules(module)
                model.model = qmodel # .half()

            # FX Graph Mode Quant:
            if quant_method == "fx":

                _model = model.model
                input_tensor = np.load('img_0.npy')
                input_tensor = torch.from_numpy(input_tensor).to(device)
                _model.to(device)
                with torch.no_grad():
                    _model(input_tensor)

                print_model_layers_info(_model)

                qcfg = QConfig(
                    activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.qint8),
                    weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
                )
                # _ = torch.ao.quantization.get_default_qconfig(backend)
                qconfig_dict = {
                    # "": torch.ao.quantization.get_default_qconfig(backend),  # Apply to all ops
                    # "": qcfg,
                    torch.nn.modules.conv.Conv2d:       qcfg,
                    torch.nn.modules.pooling.MaxPool2d: qcfg,
                }
                qconfig_mapping = QConfigMapping() \
                    .set_global(qcfg) \
                    .set_object_type(torch.nn.modules.conv.Conv2d, qcfg) \
                    .set_object_type(torch.nn.modules.pooling.MaxPool2d, qcfg)

                prepare_custom_config = PrepareCustomConfig() \
                    .set_non_traceable_module_classes([
                        # ultralytics.nn.tasks.DetectionModel,
                        # torch.nn.modules.container.Sequential,
                        ultralytics.nn.modules.head.Detect,
                        # ultralytics.nn.modules.conv.Conv,
                        # torch.nn.modules.container.ModuleList,
                        # ultralytics.nn.modules.conv.Concat,
                        ultralytics.nn.modules.block.C2f,
                    ])
                model_prepared = quantize_fx.prepare_fx(
                    _model,
                    qconfig_mapping, # qconfig_dict,
                    (input_tensor),
                    prepare_custom_config=prepare_custom_config,
                )
                model_prepared(input_tensor)
                model_quantized = quantize_fx.convert_fx(model_prepared)
                # print(model_quantized)
                if hasattr(_model, "stride"):
                    model_quantized.stride = _model.stride
                model_quantized.names = _model.module.names if hasattr(_model, "module") else _model.names
                model.model = model_quantized.half()

            if quant_method == "neuro-fx":
                from neurotrim.compression.config import Config
                from neurotrim.compression.builder import build_compressor
                from neurotrim.graph.graph_optimizer import GraphOptimizer
                from musa_bench.utils.dataloaders import create_dataloader
                from musa_bench.utils.general import colorstr, yaml_load
                from musa_bench.utils.quant_utils import CustomedTracer, calibrate, postprocess

                _model = model.model
                _model.to("cpu").eval()
                task = "val"  # path to train/val/test images
                stride = max(int(_model.stride.max()), 32)
                single_cls=False
                pad = 0.5
                rect = True
                workers = 8
                calib_data = yaml_load(f"cfg/datasets/coco128.yaml")
                dataloader = create_dataloader(
                    "datasets/coco128/" + calib_data[task],
                    imgsz,
                    batch,
                    stride,
                    single_cls,
                    pad=pad,
                    rect=rect,
                    workers=workers,
                    prefix=colorstr(f"{task}: "),
                )[0]

                nhwc = False
                if nhwc:
                    _model = _model.to(memory_format=torch.channels_last)

                (_, untraceable_clszs) = print_model_layers_info(_model)

                config = Config("yolov8_static_quant_config.json")
                config.device = device.type
                use_trace = len(untraceable_clszs) <= 0
                compressor = build_compressor(_model, config, trace=False) if not use_trace else \
                             build_compressor(_model, config, tracer=CustomedTracer())

                calib_steps = 3 # 30
                compressor.compress(
                    calib_data=dataloader, calib_func=partial(calibrate, steps=calib_steps)
                )
                graph_opt = GraphOptimizer(compressor)
                qmodel = graph_opt.optimize()
                qmodel.names = _model.names
                postprocess(qmodel)
                model.model = qmodel

            if print_mod or print_layers:
                print("==== AFTER  QUANT ==========================================================")
            if print_mod:
                print(model.model)
            if print_layers:
                print("")
                for name, module in model.model.named_modules():
                    print(f"{name:<40} :: {type(module)}")
                print("")

            quanzied_model_file = f"{model.model_name.replace('.pt', '')}-{dtype}-{quant_method}.pth"
            if os.path.exists(quanzied_model_file):
                os.remove(quanzied_model_file)
            torch.save(model.model, quanzied_model_file)
            print(f"INFO: saved quantized model({type(model.model)}) to {quanzied_model_file}, size = {os.path.getsize(quanzied_model_file) / (1024**2):.2f}MB")
            model.model.to(device).eval()

    model = YOLO(model)

    is_end2end = getattr(model.model.model[-1], "end2end", False)
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect

    y = []
    t0 = time.time()

    format_arg = format.lower()

    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):
        emoji, filename = "❌", None  # export defaults
        try:
            if format_arg and format_arg != format:
                continue

            if format == "-":
                filename = model.pt_path or model.ckpt_path or model.model_name
                exported_model = model  # PyTorch format
            elif format == 'torchscript':
                print(f"INFO: export model to torchscript")
                filename = model.export(
                    imgsz=imgsz, format="torchscript",
                    batch=batch, optimize=True,
                    half=half,
                    int8=int8,
                    data=data,
                    device="cpu",
                    verbose=True
                )
                print("INFO: exported to", filename)
                exported_model = YOLO(filename, task=model.task)

            may_compile_and_quant(exported_model)

            emoji = "❎"  # indicates export succeeded

            print(f"INFO: format = {format}, use_graph = {use_graph}, batch = {batch}, triton = {compile_mode if triton else triton}, dtype = {dtype}, half={half}, int8={int8}")

            rounds = args.rounds
            if profiling:
                rounds = 1

            def rounds_predict(warmup = False):
                _rounds = rounds if not warmup else 10
                print("INFO: rounds =", _rounds)
                for _ in range(_rounds):
                    exported_model.predict(
                        ASSETS / "bus.jpg",
                        imgsz=imgsz, device=device, half=half, int8=int8, verbose=False,
                        use_graph=use_graph
                    )

            if warmup:
                print("INFO: warming up model ...")
                rounds_predict(warmup=True)
                return

            current_ts = datetime.now().strftime("%Y%m%d-%H%M")
            if profiling:
                print(f"INFO: start profiling")
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU, 
                        # torch.profiler.ProfilerActivity.CUDA
                        torch.profiler.ProfilerActivity.MUSA
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True,
                    use_musa=True
                ) as prof:
                    rounds_predict()
                print(prof.key_averages().table(sort_by="self_musa_time_total", row_limit=100))
                trace_file_name = f"perf-trace-{DEVICE_NAME}-{current_ts}.json"
                prof.export_chrome_trace(trace_file_name)
                exit(0)
            else:
                rounds_predict()

            # Validate
            results = exported_model.val(
                data=data, batch=batch, imgsz=imgsz, plots=False, device=device, half=half, int8=int8, verbose=False
            )
            metric, speed = results.results_dict[key], results.speed["inference"]
            fps = round(1000 / (speed + eps), 2)  # frames per second
            y.append([name, "✅", round(file_size(filename), 1), round(metric, 4), round(speed, 2), fps])
        except Exception as e:
            print(f"Benchmark failure for {name}: {e}")
            traceback.print_exc()
            LOGGER.warning(f"\nERROR ❌️ Benchmark failure for {name}: {e}")
            y.append([name, emoji, round(file_size(filename), 1), None, None, None])  # mAP, t_inference

    # Print results
    check_yolo(device=device)  # print system info
    df = pd.DataFrame(y, columns=["Format", "Status❔", "Size (MB)", key, "Inference time (ms/im)", "FPS"])

    name = model.model_name
    dt = time.time() - t0

    row = y[0]
    print_table_row(bf, model_name, row[2], use_graph, dtype, batch, data, imgsz, triton, compile_mode, row[3], row[4], row[5])

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
    bf.write(f"| Model       | Size      | dtype    | batch   | dataset          | imgsz           | graph    | compile(triton)  | mAP50-95(B)      | ms/im            | FPS              |\n")
    bf.write(f"| :---------: | :-------: | :------: | :-----: | :--------------: | :--------------:| :------: | :--------------: | :--------------: | :--------------: | :--------------: |\n")
    bf.flush()

def print_table_row(bf, model, size, graph, dtype, batch, dataset, imgsz, triton, compile_mode, metrics_mAP50, ms_per_img, fps):
    triton_mode = compile_mode if triton else triton
    bf.write(f"| {model:<12}| {size:<10}| {dtype:<9}| {batch:<8}| {dataset:<17}| {imgsz:<17}| {graph:<9}| {(triton_mode):<17}| {metrics_mAP50:<17}| {ms_per_img:<17}| {fps:<17}|\n")
    bf.flush()


def main(models: List[str],
         batches: List[int],
         dataset: str = "coco128.yaml",
         imgsz: int = 640,
         device: str = DEFAULT_DEVICE,
         dtypes: List[bool] = DEFAULT_DTYPES,
         cmp_modes = COMPILE_MODES,
         profiling = False
         ):

    current_ts = datetime.now().strftime("%Y%m%d:%H%M")
    os.makedirs(f"{CWD}/benchmarks/", exist_ok=True)

    with open(f"{CWD}/benchmarks/benchmarks-table-{current_ts}.md", "w", errors="ignore", encoding="utf-8") as bf:
        print_table_head(bf)
        for dtype, model, batch, triton, graph_on in product(dtypes, models, batches, TRITON_TOGGLES, GRAPH_TOGGLES):
            int8 = dtype == "int8"
            half = dtype == "fp16"

            def _benchmark(warmup=False, compile_mode=None):
                benchmark(
                    bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch,
                    half=half, int8=int8, dtype=dtype, device=device, triton=triton, compile_mode=compile_mode,
                    warmup=warmup, profiling=profiling, use_graph=graph_on
                )

            if triton:
                print("INFO: torch compile warming up...")
                _benchmark(warmup=True, compile_mode="default")
                for compile_mode in cmp_modes:
                    print("\n")
                    _benchmark(compile_mode=compile_mode)
            else:
                print("\n")
                _benchmark()

            time.sleep(1)


if __name__ == "__main__":
    main(args.models, args.batches, args.dataset, args.imgsz, args.device, args.dtypes,
         cmp_modes=args.cmp_modes,
         profiling=args.profiling,
         )
