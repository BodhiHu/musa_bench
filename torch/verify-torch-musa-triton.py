# import torch_compat as torch
import torch
import torch.nn as nn
import torch_musa

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


device = torch.device("musa")
# model = SimpleNN(input_size=10, hidden_size=20, output_size=1).to(device)
model = SimpleNN(input_size=2, hidden_size=2, output_size=2).to(device)
model = torch.compile(model, backend="inductor", mode="max-autotune")
# print(f"\n>> backend = \n{model._compiled_model.backend}\n\n")
# print(f"\n>> codegen = \n{torch._inductor.codegen}\n\n")
# torch._dynamo.utils.save_code_for_debugging(model)

# x = torch.randn(5, 10).to(device)
x = torch.randn(2, 2).to(device)
output = model(x)

print(f"\n>> input  =\n{x}\n")
print(f"\n>> output =\n{output}\n")
