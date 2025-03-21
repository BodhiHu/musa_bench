#!/usr/bin/env python3
import torch_compat as torch
#import torch_cuda

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
