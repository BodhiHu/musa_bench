import os, sys, time
from datetime import datetime

start_time = datetime.now()
print(f"INFO: start_time = {start_time}")

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
sys.path.append(os.path.abspath(f"{current_dir}/.."))
sys.path.append(os.path.abspath(f"{current_dir}/../.."))

import argparse
import re
from ultralytics import YOLO
import torch
import torch_musa
import torch_musa.cuda_compat
import numpy as np
import subprocess
from musa_bench.utils.general import get_device_name


parser = argparse.ArgumentParser(description="Profile yolo model")
parser.add_argument("--dtype", default="fp16", type=str, help="dtype (fp16 | fp32)")
parser.add_argument("--graph", action="store_true", help="turn on musa graph")
args = parser.parse_args()

half = args.dtype == "fp16"

print(f"INFO: using half       : {half}")
print(f"INFO: using musa graph : {args.graph}")
model = YOLO('yolov8m.pt')
model.fuse()
model.model.to('musa').eval()

print("INFO: warming up")
for _ in range(3):
    results = model.predict('images/bus.jpg', half=True, use_graph=args.graph, verbose=False)

torch.musa.synchronize()
_time = datetime.now()
delta = _time - start_time
print("\nINFO: load time:", delta)
# exit(0)
time.sleep(10)

for _ in range(30):
    results = model.predict('images/bus.jpg', half=True, use_graph=args.graph, verbose=True)
torch.musa.synchronize()
time.sleep(3)
