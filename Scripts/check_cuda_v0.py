import subprocess

import torch


def main():
    try:
        nvidia_smi = subprocess.check_output(["nvidia-smi"], text=True)
        print(nvidia_smi, flush=True)
    except FileNotFoundError:
        print("nvidia-smi not found", flush=True)

    print(f"torch={torch.__version__}", flush=True)
    print(f"torch_cuda={torch.version.cuda}", flush=True)
    print(f"cudnn={torch.backends.cudnn.version()}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    assert torch.cuda.is_available()
    print(f"device_count={torch.cuda.device_count()}", flush=True)
    print(f"device_name={torch.cuda.get_device_name(0)}", flush=True)
    tensor = torch.ones((2, 2), device="cuda")
    print(f"cuda_tensor_sum={float(tensor.sum().item())}", flush=True)


if __name__ == "__main__":
    main()
