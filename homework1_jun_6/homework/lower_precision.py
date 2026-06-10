from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 1D tensor into 3-bit groups.

    We pack 8 three-bit values into 3 bytes. With group_size=32, each group stores
    32 weights in 12 bytes plus a float16 normalization, which averages to 3.5 bits/value.
    """
    assert x.dim() == 1
    assert group_size % 8 == 0
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    normalization = torch.where(normalization == 0, torch.ones_like(normalization), normalization)

    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 7).round().clamp_(0, 7).to(torch.int32)
    blocks = x_quant_8.view(-1, group_size // 8, 8)

    packed = torch.zeros(blocks.size(0), blocks.size(1), dtype=torch.int32, device=x.device)
    for index in range(8):
        packed |= (blocks[:, :, index] & 0x7) << (3 * index)

    x_quant_3 = torch.empty(blocks.size(0), blocks.size(1) * 3, dtype=torch.int8, device=x.device)
    x_quant_3[:, 0::3] = (packed & 0xFF).to(torch.int8)
    x_quant_3[:, 1::3] = ((packed >> 8) & 0xFF).to(torch.int8)
    x_quant_3[:, 2::3] = ((packed >> 16) & 0xFF).to(torch.int8)
    return x_quant_3, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """Reverse block_quantize_3bit."""
    assert x_quant_3.dim() == 2

    normalization = normalization.to(torch.float32).view(-1, 1, 1)
    blocks = x_quant_3.size(1) // 3
    bytes_ = x_quant_3.view(-1, blocks, 3).to(torch.int32)
    packed = (bytes_[:, :, 0] & 0xFF) | ((bytes_[:, :, 1] & 0xFF) << 8) | ((bytes_[:, :, 2] & 0xFF) << 16)

    x_quant_8 = torch.empty(bytes_.size(0), blocks, 8, dtype=torch.int32, device=x_quant_3.device)
    for index in range(8):
        x_quant_8[:, :, index] = (packed >> (3 * index)) & 0x7

    x_norm = x_quant_8.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, group_size // 8 * 3, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)

        self.bias = None
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]
            weight_flat = weight.view(-1).to(torch.float32)
            weight_q3, weight_norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(weight_q3.to(self.weight_q3.device))
            self.weight_norm.copy_(weight_norm.to(self.weight_norm.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm).view(self._shape)
            return torch.nn.functional.linear(x, weight, self.bias)


class LowerBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
                torch.nn.ReLU(),
                Linear3Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None):
    net = LowerBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
