import argparse
from datetime import datetime
import os
import re
from ultralytics import YOLO
import torch
import torch_musa
import torch_musa.cuda_compat
import numpy as np
import subprocess
from musa_bench.utils.general import get_device_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Profile yolo model")
    parser.add_argument("--dtype",      default="fp16", type=str, help="dtype (e.g. int8|fp16|fp32)")
    parser.add_argument("--no-compile", action="store_true", help="trun off compiling.")
    parser.add_argument("--profiler",   default="plain", type=str, help="use plain or autograd profiler")
    parser.add_argument("--graph",      action="store_true", help="turn on musa graph")
    parser.add_argument("--rounds",     default=1, type=int, help="rounds to profiles")

    args = parser.parse_args()
    half = args.dtype == "fp16"

    model = YOLO('yolov8m.pt').to('cuda')
    model.model.to('musa').eval()

    print(f"INFO: using half: {half}")

    if not args.no_compile:
        print("INFO: compiling model ...")
        model.model = torch.compile(model.model, backend="inductor", mode="default")

    # warmup model
    for i in range(10):
        results = model.predict('images/bus.jpg', half=half, use_graph=args.graph)

    current_ts = datetime.now().strftime("%Y%m%d-%H%M")
    device_name = get_device_name()
    rounds = args.rounds or 1

    if args.profiler == "plain":
        print("INFO: using plain profiler")
        print("INFO: using chrome_trace profiler")
        trace_file_name = f"profiler-trace-{device_name}-{current_ts}.json"

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
            for i in range(rounds):
                results = model.predict('images/bus.jpg', half=half, use_graph=args.graph)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
        prof.export_chrome_trace(trace_file_name)
    elif args.profiler == "autograd":
        print("INFO: using autograd profiler")
        trace_file_name = f"profiler-trace-autograd-{device_name}-{current_ts}.json"

        with torch.autograd.profiler.profile(
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            with_modules=True,
            use_musa=True
        ) as prof:
            for i in range(rounds):
                results = model.predict('images/bus.jpg', half=half, use_graph=args.graph)
        prof.export_chrome_trace(trace_file_name)
