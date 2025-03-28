from ultralytics import YOLO
import torch
import torch_musa
import numpy as np

# # 创建字典来保存各层输出
# outputs = {}

# # 定义钩子函数
# def hook_fn(name):
#     def hook(module, input, output):
#         if isinstance(output, tuple):  # 如果输出是元组
#             outputs[name] = [x.detach() if isinstance(x, torch.Tensor) else x for x in output]
#         else:  # 如果是单个张量
#             outputs[name] = output.detach()
#     return hook

# # 注册钩子
# for name, layer in model.model.named_modules():
#     layer.register_forward_hook(hook_fn(name))


model = YOLO('yolov8m.pt').to('cuda')  # 必须移到GPU
model.model.to('musa').eval()
print("INFO: compiling model ...")
model.model = torch.compile(model.model, backend="inductor", mode="default")
# 运行推理（注意：需通过原始YOLO接口调用）
for i in range(9):
    results = model.predict('images/bus.jpg')  # 可结合FP16


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
) as prof:
    results = model.predict('images/bus.jpg')  # 可结合FP16
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))

# # 现在outputs字典包含了每一层的输出
# for name, output in outputs.items():
#     if isinstance(output, list):
#         for i in range(len(output)):
#             if isinstance(output[i], list):
#                 for j in range(len(output[i])):
#                     print(f"Layer: {name}, Output {i} {j} shape: {output[i][j].shape}")
#             else:
#                 print(f"Layer: {name}, Output {i} shape: {output[i].shape}")
#     else:
#         print(f"Layer: {name}, Output shape: {output.shape}")