from .base_vlm import BaseVLM
from .data import VQADataset, benchmark


def load_clip(*args, **kwargs):
    from .clip import load

    return load(*args, **kwargs)


def load_vlm(*args, **kwargs):
    from .finetune import load

    return load(*args, **kwargs)


def train(*args, **kwargs):
    from .finetune import train as _train

    return _train(*args, **kwargs)

__all__ = ["BaseVLM", "VQADataset", "benchmark", "train", "load_vlm", "load_clip"]
