import torch
import torch_musa
import torch_musa.cuda_compat
import torch.nn as nn

# Ensure CUDA is available
assert torch.cuda.is_available(), "CUDA is not available"

DEVICE = "cuda:0"

# Simple model (Linear layer)
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

# Instantiate and move to CUDA
model = SimpleModel().to(DEVICE)
model.eval()

# Fixed input
x_real = torch.randn(1, 10, device=DEVICE)
x_static = torch.empty_like(x_real)

# Warm-up to ensure all allocations done
with torch.no_grad():
    for _ in range(3):
        _ = model(x_real)

# Output containers
static_output = [None]

# Create CUDA Graph
g = torch.cuda.CUDAGraph()

# Capture the graph
with torch.cuda.graph(g):
    static_output[0] = model(x_static)

# Run graph-based inference
x_static.copy_(x_real)
g.replay()
graph_output = static_output[0]

# Compare with normal inference
with torch.no_grad():
    eager_output = model(x_real)

# Validate output is close
if torch.allclose(graph_output, eager_output, atol=1e-6):
    print("✅ CUDA Graph is working correctly!")
else:
    print("❌ CUDA Graph output does not match standard inference.")
