import torch
import torch_musa
from torch._inductor.select_algorithm import extern_kernels
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor

def test_addmm(addmm=torch.addmm, grad=False):
  # addmm = torch.addmm
  # addmm = extern_kernels.addmm

  device = torch.device("musa")

  primals_2 = torch.tensor(
    [0.2069, 0.3700],
    device='musa:0', requires_grad=grad)
  primals_3 = torch.tensor(
    [[-1.4669,  0.1403],
     [-0.7819,  0.4707]], device='musa:0')
  primals_1 = torch.tensor(
    [[ 0.5870,  0.4790],
     [-0.2002,  0.3976]], device='musa:0', requires_grad=grad)

  buf0 = torch.empty((2, 2), device='musa', dtype=torch.float32)

  addmm(
    primals_2,
    primals_3,
    primals_1,
    alpha=1, beta=1,
    out=buf0
  )
  print(">> buf0 =\n", buf0)

  buf1 = torch.empty((2, 2), device='musa', dtype=torch.float32)
  A = reinterpret_tensor(primals_2, (2, 2), (0, 1), 0)
  B = reinterpret_tensor(primals_1, (2, 2), (1, 2), 0)
  print(">> reinterpreted tensors:")
  print(">> A:\n", A)
  print(">> B:\n", B)
  addmm(
    A,#reinterpret_tensor(primals_2, (2, 2), (0, 1), 0),
    primals_3,
    B,#reinterpret_tensor(primals_1, (2, 2), (1, 2), 0),
    alpha=1, beta=1,
    out=buf0
  )
  print(">> buf1 =\n", buf1)


# for grad in (False, True):
test_addmm(grad=False)
test_addmm(grad=True)
