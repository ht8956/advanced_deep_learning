from .base_vlm import BaseVLM
from .clip import load as load_clip
from .data import VQADataset, benchmark


def load_vlm(*args, **kwargs):
    from .finetune import load

    return load(*args, **kwargs)


def train(*args, **kwargs):
    from .finetune import train as _train

    return _train(*args, **kwargs)

__all__ = ["BaseVLM", "VQADataset", "benchmark", "train", "load_vlm", "load_clip"]
