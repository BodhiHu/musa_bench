import torch_compat as torch
import torch.nn as nn

torch._logging.set_logs(dynamo=50, inductor=50)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda")
model = SimpleNN(input_size=10, hidden_size=20, output_size=1).to(device)
model = torch.compile(model, backend="inductor", mode="max-autotune")
# print(f"\n>> backend = \n{model._compiled_model.backend}\n\n")
# print(f"\n>> codegen = \n{torch._inductor.codegen}\n\n")
# torch._dynamo.utils.save_code_for_debugging(model)

x = torch.randn(5, 10).to(device)
output = model(x)

print(f"\b>> output = {output}\n")
