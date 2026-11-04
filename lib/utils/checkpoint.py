"""Load trusted TGTrack training checkpoints across supported PyTorch versions."""
import inspect
import torch


def load_checkpoint(path, map_location="cpu", weights_only=False):
    # Training checkpoints include Settings and optimizer state, not just tensors.
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = weights_only
    return torch.load(path, **kwargs)
