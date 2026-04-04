#!/usr/bin/env python3
"""GPU device and memory management utilities for dual 3090 setup."""
import gc
import torch


def get_device(gpu_id=0):
    """Return CUDA device, falling back to CPU."""
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')


def clear_gpu(gpu_id=None):
    """Free cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        if gpu_id is not None:
            with torch.cuda.device(gpu_id):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()


def unload_model(model):
    """Move model to CPU, delete, and free GPU memory."""
    if model is not None:
        model.cpu()
        del model
    clear_gpu()


def gpu_memory_report():
    """Print memory usage for all GPUs."""
    if not torch.cuda.is_available():
        print("No CUDA GPUs available")
        return
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        free = total - cached
        name = torch.cuda.get_device_name(i)
        print(f"  GPU {i} ({name}): {alloc:.1f}GB alloc / {cached:.1f}GB reserved / {free:.1f}GB free / {total:.1f}GB total")
