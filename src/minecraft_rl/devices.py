"""PyTorch device selection shared by the neural stages."""

import torch

DEVICES = ("auto", "cpu", "mps", "cuda")


def select_device(requested: str) -> torch.device:
    """Resolve `auto` to the best available device; unavailable devices raise."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is not available on this host")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is not available on this host")
    if requested not in DEVICES:
        raise ValueError(f"unknown device {requested!r}")
    return torch.device(requested)
