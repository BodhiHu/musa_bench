import torch
import torch_musa
import time

def measure_transfer_bandwidth():
    if not torch.musa.is_available():
        print("CUDA not available!")
        return

    device = torch.device('musa')
    print(f"Testing on: {torch.musa.get_device_name(0)}")

    size_mb = 10
    while size_mb <= 1024:
        num_elements = size_mb * 1024 * 1024 // 4
        cpu_data = torch.randn(num_elements, dtype=torch.float32)
        
        _ = torch.randn(10, device=device)  # Warmup

        # Test CPU -> GPU (2 runs)
        musa_times = []
        for _ in range(2):
            torch.musa.synchronize()
            start = time.time()
            gpu_data = cpu_data.to(device)
            torch.musa.synchronize()
            musa_times.append(time.time() - start)

        # Test GPU -> CPU (2 runs)
        cpu_times = []
        for _ in range(2):
            torch.musa.synchronize()
            start = time.time()
            _ = gpu_data.to('cpu')
            torch.musa.synchronize()
            cpu_times.append(time.time() - start)

        # Compute average bandwidth
        avg_musa_bw = size_mb / (sum(musa_times) / len(musa_times))
        avg_cpu_bw = size_mb / (sum(cpu_times) / len(cpu_times))
        
        print(f"Size: {size_mb:4d} MB | CPU->GPU: {avg_musa_bw:8.2f} MB/s | GPU->CPU: {avg_cpu_bw:8.2f} MB/s")
        size_mb *= 2

if __name__ == "__main__":
    measure_transfer_bandwidth()
