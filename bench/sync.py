"""CUDA synchronization helpers.

The protocol requires torch.cuda.synchronize() before and after every
measured CUDA region. These helpers fail LOUDLY when that contract is
violated (the requirement says: no silent skip).
"""

from __future__ import annotations


def is_cuda_device(device: str) -> bool:
    return str(device).startswith("cuda")


def cuda_sync(device: str) -> None:
    """Synchronize the CUDA device. Raises if device is CUDA and sync fails.

    Returns immediately for CPU/MPS/TPU devices.
    """
    if not is_cuda_device(device):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required to synchronize CUDA but is not importable."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"device={device!r} requested but torch.cuda.is_available() is False"
        )
    torch.cuda.synchronize()


def assert_cuda_available_if_required(device: str) -> None:
    if not is_cuda_device(device):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("CUDA device requested but torch not installed") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
