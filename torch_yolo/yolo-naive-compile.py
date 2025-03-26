import torch
import torch_musa
import numpy as np

try:
    import torch_musa.cuda_compat
except Exception as exc:
    print("WARN: could not import torch_musa.cuda_compat: ", exc)
    print("WARN: fall back to manual cuda compat")
    # fallback to manual cuda compat
    del torch.cuda
    torch.cuda = torch.musa

device = "musa"

# 加载模型
model_path = "yolov8m.pt"
model_data = torch.load(model_path, map_location=device)
model = model_data["model"].float().eval()
# print(f">> model = {type(model)}\n", model)
# exit(0)

model = torch.compile(model, backend="inductor", mode="max-autotune")

# 预处理
input_tensor = np.load('img_0.npy')
input_tensor = torch.from_numpy(input_tensor).to(device)

# 推理
with torch.no_grad():
    outputs = model(input_tensor)
