#!/usr/bin/env python3

import copy
from functools import partial
import os
import re
import sys
import traceback
import numpy as np
import logging
import platform
import torch
import torch_musa
from tqdm import tqdm
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
from torch.ao.quantization import quantize_fx, HistogramObserver, MinMaxObserver, QConfig

try:
    import torch_musa.cuda_compat
except Exception as exc:
    print("WARN: could not import torch_musa.cuda_compat: ", exc)
    print("WARN: fall back to manual cuda compat")
    # fallback to manual cuda compat
    del torch.cuda
    torch.cuda = torch.musa


current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.abspath(f"{current_dir}/.."))
sys.path.append(os.path.abspath(f"{current_dir}/../.."))

# torch._dynamo.config.cache_size_limit = 128

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
DEFAULT_DEVICE = "cuda:0"
# DEFAULT_DEVICE = "musa:0"
COMPILE_MODES = [
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
CWD = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO benchmarks.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="List of models to benchmark.")
    parser.add_argument("--batches", nargs="+", type=int, default=DEFAULT_BATCHES, help="List of batch to test.")
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
    parser.add_argument("-p", "--profiling", action="store_true", help="turn on perf profiling mode.")

    args = parser.parse_args()

    global TRITON_TOGGLES
    if args.triton_toggles:
        TRITON_TOGGLES.append(False)
    if args.no_compile:
        TRITON_TOGGLES = [False]

    return args

args = parse_args()


if args.debug:
    # logging.basicConfig(
    #     level=logging.DEBUG, 
    #     # format="%(asctime)s - %(levelname)s - %(message)s",
    # )
    # # Ensure the root logger captures debug messages
    # logger = logging.getLogger()
    # logger.setLevel(logging.DEBUG)
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
    profiling = False,
):

    model_name = model

    imgsz = check_imgsz(imgsz)
    assert imgsz[0] == imgsz[1] if isinstance(imgsz, list) else True, "benchmark() only supports square imgsz."

    import pandas as pd  # scope for faster 'import ultralytics'

    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)

    def may_compile_and_quant(model: YOLO):
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
            full_quant = True

            # Dynamic quant:
            if quant_method == "dynamic":
                model.model = torch.quantization.quantize_dynamic(
                    model.model,
                    dtype=torch.qint8
                )

            # Static quant:
            if quant_method == "static":
                Conv2d      = torch.nn.modules.conv.Conv2d
                BatchNorm2d = torch.nn.modules.batchnorm.BatchNorm2d

                # Quant to int8
                qcfg = QConfig(
                    activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.qint8),
                    weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
                )
                quant_list = [
                    torch.nn.modules.conv.Conv2d,
                    torch.nn.MaxPool2d,
                    torch.mul,
                    torch.add,
                    torch.nn.Sigmoid,
                    torch.split,
                ]
                if full_quant:
                    quant_list = None

                print_mod = False
                print_layers = False
                fuse_ops = False
                matched_conv2d_name = None
                matched_bn2d_name = None

                if print_mod:
                    print(model.model)
                for name, module in model.model.named_modules():
                    if print_layers:
                        print(f"{name:<24} :: {type(module)}")
                    if quant_list is None or type(module) in quant_list:
                        module.qconfig = qcfg
                    if fuse_ops:
                        # if re.search("activation.*", str(type(module)), re.IGNORECASE):
                        #     print("  => will not quantize")
                        #     matched_conv2d_name, matched_bn2d_name = None, None
                        #     continue

                        if isinstance(module, Conv2d):
                            matched_conv2d_name = name
                            continue
                        if isinstance(module, BatchNorm2d):
                            matched_bn2d_name = name
                            assert matched_conv2d_name is not None and matched_bn2d_name is not None
                            # fuse conv2d and bn2d
                            print(f"  => Fusing {matched_conv2d_name} and {matched_bn2d_name}")
                            torch.quantization.fuse_modules(
                                model.model,
                                [ matched_conv2d_name, matched_bn2d_name ],
                                inplace=True
                            )
                        matched_conv2d_name, matched_bn2d_name = None, None

                # model.model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
                model.model.qconfig = qcfg
                # model.model.qconfig_dict = qconfig_dict
                model_prepared = torch.quantization.prepare(model.model, allow_list=quant_list)
                model_prepared(input_tensor)
                model.model = torch.quantization.convert(model_prepared, allow_list=quant_list)

            # FX Graph Mode Quant:
            if quant_method == "fx":
                qcfg = QConfig(
                    activation=HistogramObserver.with_args(reduce_range=False, dtype=torch.qint8),
                    weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8)
                )
                qconfig_dict = {
                    # "": torch.ao.quantization.get_default_qconfig(backend),  # Apply to all ops
                    # "": qcfg,
                    torch.nn.Conv2d:    qcfg,
                    torch.nn.MaxPool2d: qcfg,
                    torch.mul:          qcfg,
                    torch.add:          qcfg,
                    torch.nn.Sigmoid:   qcfg,
                    torch.split:        qcfg
                }
                model_prepared = quantize_fx.prepare_fx(model.model, qconfig_dict, (input_tensor))
                model_prepared(input_tensor)
                model_quantized = quantize_fx.convert_fx(model_prepared)
                model.model = model_quantized

            if quant_method == "neuro-fx":
                from torch import fx
                from neurotrim.compression.config import Config
                from neurotrim.compression.builder import build_compressor
                from neurotrim.graph.graph_optimizer import GraphOptimizer
                from musa_bench.utils.dataloaders import create_dataloader
                from musa_bench.utils.general import colorstr, yaml_load

                class CustomedTracer(fx.Tracer):
                    """
                    ``Tracer`` is the class that implements the symbolic tracing functionality
                    of ``torch.fx.symbolic_trace``. A call to ``symbolic_trace(m)`` is equivalent
                    to ``Tracer().trace(m)``.
                    This Tracer override the ``is_leaf_module`` function to make symbolic trace
                    right in some cases.
                    """

                    def __init__(self, *args, customed_leaf_module=None, **kwargs):
                        super().__init__(*args, **kwargs)
                        self.customed_leaf_module = customed_leaf_module

                    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                        """
                        A method to specify whether a given ``nn.Module`` is a "leaf" module.
                        Leaf modules are the atomic units that appear in
                        the IR, referenced by ``call_module`` calls. By default,
                        Modules in the PyTorch standard library namespace (torch.nn)
                        are leaf modules. All other modules are traced through and
                        their constituent ops are recorded, unless specified otherwise
                        via this parameter.
                        Args:
                            m (Module): The module being queried about
                            module_qualified_name (str): The path to root of this module. For example,
                                if you have a module hierarchy where submodule ``foo`` contains
                                submodule ``bar``, which contains submodule ``baz``, that module will
                                appear with the qualified name ``foo.bar.baz`` here.
                        """
                        if self.customed_leaf_module and isinstance(m, self.customed_leaf_module):
                            return True

                        if hasattr(m, "_is_leaf_module") and m._is_leaf_module:
                            return True

                        return m.__module__.startswith("torch.nn") and not isinstance(
                            m, torch.nn.Sequential
                        )

                def calibrate(model, dataloader, steps=30):
                    for batch_i, (imgs, _, _, _) in tqdm(enumerate(dataloader)):
                        imgs = imgs.float()
                        imgs /= 255  # 0 - 255 to 0.0 - 1.0

                        # Inference
                        _, _ = model(imgs)  # inference, loss outputs
                        if batch_i >= steps:
                            return

                def _getattr(model, name):
                    """customize getattr function to recursive get attribute for pytorch module"""
                    name_list = name.split(".")
                    for name in name_list:  # pylint: disable=redefined-argument-from-local
                        model = getattr(model, name)
                    return model

                def postprocess(model: fx.GraphModule):
                    """replace float add to quantized one, reduce dequantize ops"""
                    for node in model.graph.nodes:
                        if node.op != "call_module":
                            continue
                        if not isinstance(_getattr(model, node.target), torch.nn.Upsample):
                            continue
                        args = node.args
                        assert len(args) == 1
                        up_arg = args[0]
                        if up_arg.target == "dequantize":
                            quant_arg = up_arg.args[0]
                        else:
                            continue
                        assert len(node.users) == 1
                        cat_node = list(node.users.keys())[0]
                        assert cat_node.target == torch.cat
                        cat_inputs = cat_node.args[0]
                        flag = True
                        for cat_inp in cat_inputs:
                            if cat_inp is node:
                                continue
                            if cat_inp.target == "dequantize":
                                cat_quant = cat_inp.args[0]
                                cat_inp.replace_all_uses_with(cat_quant)
                                model.graph.erase_node(cat_inp)
                            else:
                                flag = False
                        if flag and len(cat_node.users) == 1:
                            up_arg.replace_all_uses_with(quant_arg)
                            model.graph.erase_node(up_arg)
                            late_node = list(cat_node.users.keys())[0]
                            assert late_node.target == torch.quantize_per_tensor
                            late_node.replace_all_uses_with(cat_node)
                            model.graph.erase_node(late_node)

                    model.graph.lint()
                    model.recompile()

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

                config = Config("static_quant_config.json")
                config.device = device.type
                compressor = build_compressor(_model, config, trace=CustomedTracer())
                compressor.compress(
                    calib_data=dataloader, calib_func=partial(calibrate, steps=30)
                )
                graph_opt = GraphOptimizer(compressor)
                qmodel = graph_opt.optimize()
                qmodel.names = _model.names
                postprocess(qmodel)
                model.model = qmodel

            if print_layers:
                print("")
                print("=============================================================================")
                print("Model after quantization:")
                for name, module in model.model.named_modules():
                    print(f"{name:<24} :: {type(module)}")
                print("=============================================================================")
                print("")

            quanzied_model_file = f"{model.model_name.replace('.pt', '')}-{dtype}-{quant_method}.pth"
            if os.path.exists(quanzied_model_file):
                os.remove(quanzied_model_file)
            torch.save(model.model, quanzied_model_file)
            print(f"INFO: saved quantized model({type(model.model)}) to {quanzied_model_file}")
            model.model.to(device).eval()

    model = YOLO(model)
    may_compile_and_quant(model)

    is_end2end = getattr(model.model.model[-1], "end2end", False)
    data = data or TASK2DATA[model.task]  # task to dataset, i.e. coco8.yaml for task=detect
    key = TASK2METRIC[model.task]  # task to metric, i.e. metrics/mAP50-95(B) for task=detect

    y = []
    t0 = time.time()

    format_arg = format.lower()
    # if format_arg:
    #     formats = frozenset(export_formats()["Argument"])
    #     assert format in formats, f"Expected format to be one of {formats}, but got '{format_arg}'."

    for i, (name, format, suffix, cpu, gpu, _) in enumerate(zip(*export_formats().values())):
        emoji, filename = "❌", None  # export defaults
        try:
            if format_arg and format_arg != format:
                continue

            print(f"INFO: format = {format}, dtype = {dtype}, half={half}, int8={int8}")
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

            if warmup:
                print("INFO: warming up model ...")
                for _half in (True, False):
                    for _ in range(3):
                        exported_model.predict(
                            ASSETS / "bus.jpg",
                            imgsz=imgsz, device=device, half=_half, int8=int8, verbose=False
                        )
                return

            if profiling:
                print(f"INFO: start profiling")
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU, 
                        torch.profiler.ProfilerActivity.CUDA
                        # torch.profiler.ProfilerActivity.MUSA
                    ],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    with_flops=True,
                    with_modules=True,
                ) as prof:
                    exported_model.predict(
                        ASSETS / "bus.jpg",
                        imgsz=imgsz, device=device, half=half, int8=int8, verbose=False
                    )
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
                return
            else:
                exported_model.predict(
                    ASSETS / "bus.jpg",
                    imgsz=imgsz, device=device, half=half, int8=int8, verbose=False
                )

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
    print_table_row(bf, model_name, row[2], dtype, batch, data, imgsz, triton, compile_mode, row[3], row[4], row[5])

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
    bf.write(f"| Model            | Size             | dtype            | batch            | dataset          | imgsz            | compile(triton)  | mAP50-95(B)      | ms/im            | FPS              |\n")
    bf.write(f"| :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |\n")
    bf.flush()

def print_table_row(bf, model, size, dtype, batch, dataset, imgsz, triton, compile_mode, metrics_mAP50, ms_per_img, fps):
    if triton:
        bf.write(f"| {model:<17}| {size:<17}| {dtype:<17}| {batch:<17}| {dataset:<17}| {imgsz:<17}| {(compile_mode):<17}| {metrics_mAP50:<17}| {ms_per_img:<17}| {fps:<17}|\n")
    else:
        bf.write(f"| {model:<17}| {size:<17}| {dtype:<17}| {batch:<17}| {dataset:<17}| {imgsz:<17}| {(triton):<17}| {metrics_mAP50:<17}| {ms_per_img:<17}| {fps:<17}|\n")
    bf.flush()


def main(models: List[str],
         batches: List[int],
         dataset: str = "coco128.yaml",
         imgsz: int = 640,
         device: str = DEFAULT_DEVICE,
         dtypes: List[bool] = DEFAULT_DTYPES,
         cmp_modes = COMPILE_MODES,
         profiling = False):

    current_ts = datetime.now().strftime("%Y%m%d:%H%M")
    os.makedirs(f"{CWD}/benchmarks/", exist_ok=True)
    with open(f"{CWD}/benchmarks/benchmarks-table-{current_ts}.md", "w", errors="ignore", encoding="utf-8") as bf:
        print_table_head(bf)
        for dtype, model, batch, triton, in product(dtypes, models, batches, TRITON_TOGGLES):
            half = dtype == "fp16"
            int8 = dtype == "int8"
            if triton:
                print("INFO: torch compile warming up...")
                benchmark(
                    bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch,
                    half=half, int8=int8, dtype=dtype, device=device, triton=triton, compile_mode="default",
                    warmup=True
                )
                for compile_mode in cmp_modes:
                    print("\n")
                    benchmark(
                        bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch,
                        half=half, int8=int8, dtype=dtype, device=device, triton=triton, compile_mode=compile_mode,
                        profiling=profiling
                    )
            else:
                print("\n")
                benchmark(
                    bf=bf, model=model, data=dataset, imgsz=imgsz, batch=batch,
                    half=half, int8=int8, dtype=dtype, device=device, triton=triton,
                )

            time.sleep(1)


if __name__ == "__main__":
    main(args.models, args.batches, args.dataset, args.imgsz, args.device, args.dtypes,
         cmp_modes=args.cmp_modes, profiling=args.profiling
        )
