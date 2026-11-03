"""Проверка бэкенда torch: версии, доступность GPU, замер matmul CPU vs GPU."""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import torch

print(f"torch: {torch.__version__}")
print(f"torch.version.cuda: {torch.version.cuda}")
print(f"torch.version.hip: {torch.version.hip}")
print(f"cuda is_available: {torch.cuda.is_available()}")
print(f"device_count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"device[0]: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  память: {props.total_memory / 1024 ** 3:.1f} ГБ, "
          f"SM/CU: {getattr(props, 'multi_processor_count', '?')}")

SIZE = 4096
REPEATS = 10
a = torch.randn(SIZE, SIZE)
b = torch.randn(SIZE, SIZE)


def bench(device: str) -> float:
    x, y = a.to(device), b.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    for _ in range(2):                      # прогрев
        x @ y
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        x @ y
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / REPEATS * 1000.0


cpu_ms = bench("cpu")
print(f"\nmatmul {SIZE}x{SIZE}: CPU {cpu_ms:.1f} мс")
if torch.cuda.is_available():
    gpu_ms = bench("cuda")
    print(f"matmul {SIZE}x{SIZE}: GPU {gpu_ms:.1f} мс (ускорение {cpu_ms / gpu_ms:.1f}x)")
else:
    print("GPU-бэкенд недоступен — обучение останется на CPU")
