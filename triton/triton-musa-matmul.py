import torch
import torch_musa
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(A_ptr, B_ptr, C_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    A_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float16)
    B_block = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_N), dtype=tl.float16)
    C_block = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        mask_A = tl.where(offs_m[:, None] < M, k + offs_k[None, :] < K, False)
        mask_B = tl.where(k + offs_k[:, None] < K, offs_n[None, :] < N, False)
        A_block = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak, mask=mask_A, other=0.0)
        B_block = tl.load(B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_B, other=0.0)
        
        C_block += tl.dot(A_block, B_block)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, C_block, mask=mask)

def triton_matmul(A: torch.Tensor, B: torch.Tensor):
    M, K = A.shape
    K, N = B.shape
    C = torch.empty((M, N), device='musa', dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    matmul_kernel[grid](
        A, B, C, M, N, K, 
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=16
    )
    return C

def cpu_matmul(A: torch.Tensor, B: torch.Tensor):
    A_cpu = A.cpu()  # 确保 A 在 CPU
    B_cpu = B.cpu()  # 确保 B 在 CPU
    return torch.mm(A_cpu, B_cpu)  # PyTorch 在 CPU 上执行矩阵乘法

def singleTest():
  # 测试
  A = torch.ones(32, 16, dtype=torch.float16)
  B = torch.ones(16, 32, dtype=torch.float16)

  C_cpu = cpu_matmul(A, B).to(torch.float32)  # CPU 计算
  cpu_nonzero_indices = C_cpu.nonzero(as_tuple=True)
  C_gpu = triton_matmul(A.to('musa'), B.to('musa')).cpu()  # GPU 计算，转换回 CPU
  gpu_nonzero_indices = C_gpu.nonzero(as_tuple=True)

  torch.set_printoptions(precision=3)
  # torch.set_printoptions(threshold=10000)
  # torch.set_printoptions(linewidth=2000)
  # torch.set_printoptions(sci_mode=False)
  # 结果对比
  print("<<<----------------------------------------------------------")
  print("C_cpu:")
  print(C_cpu)
  print("C_cpu_none_zeros:")
  for idx in zip(*cpu_nonzero_indices):
    print(f"索引: {idx}, 值: {C_gpu[idx]}")
  print("...----------------------------------------------------------")
  print("C_gpu:")
  print(C_gpu)
  print("C_gpu_none_zeros:")
  for idx in zip(*gpu_nonzero_indices):
    print(f"索引: {idx}, 值: {C_gpu[idx]}")
  print("...----------------------------------------------------------")
  print("误差:", torch.allclose(C_cpu, C_gpu, atol=1e-2))
  # print("误差:", torch.testing.assert_close(C_cpu, C_gpu, atol=16*1e-4, rtol=1e-3, equal_nan=False))
  print("最大误差:", torch.max(torch.abs(C_cpu - C_gpu)))
  print("---------------------------------------------------------->>>")

def main():
  singleTest()
  return 0

if __name__ == "__main__":
    exit(main())
