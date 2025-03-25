# import torch_compat as torch
import torch
import torch.nn as nn
import torch_musa
import torch.profiler


USE_INDUCTOR = True
PROFILING_ON = True
device = torch.device("musa")

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        # self.fc1 = nn.Linear(input_size, hidden_size)
        # self.relu = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

# model = SimpleNN(input_size=10, hidden_size=20, output_size=1).to(device)
model = SimpleNN(input_size=2, hidden_size=2, output_size=2).to(device)
for param in model.parameters():
    param.requires_grad = False
if USE_INDUCTOR:
    model = torch.compile(model, backend="inductor", mode="max-autotune")
# print(f"\n>> backend = \n{model._compiled_model.backend}\n\n")
# print(f"\n>> codegen = \n{torch._inductor.codegen}\n\n")
# torch._dynamo.utils.save_code_for_debugging(model)

# x = torch.randn(5, 10).to(device)
x = torch.randn(2, 2).to(device)

if not PROFILING_ON:
    with torch.no_grad():
        output = model(x)
else:
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
        with torch.no_grad():
            output = model(x)
    print(prof.key_averages().table(sort_by="musa_time_total", row_limit=10))


print(f"\n\n>> input  =\n{x}\n")
print(f"\n>> output =\n{output}\n")
