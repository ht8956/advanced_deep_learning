from .base_vlm import BaseVLM
from .clip import load as load_clip
from .data import VQADataset, benchmark


def load_vlm(*args, **kwargs):
    # Lazy import avoids module re-import warning when running `python -m homework.finetune`.
    from .finetune import load

    return load(*args, **kwargs)


def train(*args, **kwargs):
    # Lazy import avoids module re-import warning when running `python -m homework.finetune`.
    from .finetune import train as finetune_train

    return finetune_train(*args, **kwargs)

__all__ = ["BaseVLM", "VQADataset", "benchmark", "train", "load_vlm", "load_clip"]
