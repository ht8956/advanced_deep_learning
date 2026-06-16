from pathlib import Path
from typing import cast
import zlib

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        super().__init__()
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive

    def compress(self, x: torch.Tensor) -> bytes:
        """
        Compress the image into a torch.uint8 bytes stream (1D tensor).

        Use arithmetic coding.
        """
        with torch.inference_mode():
            tokens = self.tokenizer.encode_index(x)

        if tokens.dim() == 3:
            tokens = tokens[0]

        h, w = tokens.shape
        tokens_np = tokens.to(torch.int64).cpu().numpy()

        if tokens_np.max() < 2**16:
            dtype_code = 1
            payload = tokens_np.astype(np.uint16, copy=False).tobytes(order="C")
        else:
            dtype_code = 2
            payload = tokens_np.astype(np.uint32, copy=False).tobytes(order="C")

        compressed_payload = zlib.compress(payload, level=9)
        header = b"CMP1" + np.array([h, w], dtype=np.uint16).tobytes() + bytes([dtype_code])
        return header + compressed_payload

    def decompress(self, x: bytes) -> torch.Tensor:
        """
        Decompress a tensor into a PIL image.
        You may assume the output image is 150 x 100 pixels.
        """
        if len(x) < 9 or x[:4] != b"CMP1":
            raise ValueError("Invalid compressed payload format")

        h, w = np.frombuffer(x[4:8], dtype=np.uint16).tolist()
        dtype_code = x[8]

        if dtype_code == 1:
            dtype = np.uint16
        elif dtype_code == 2:
            dtype = np.uint32
        else:
            raise ValueError("Unsupported token dtype in compressed payload")

        payload = zlib.decompress(x[9:])
        tokens_np = np.frombuffer(payload, dtype=dtype).reshape(h, w)

        model_device = next(self.tokenizer.parameters()).device
        tokens = torch.from_numpy(tokens_np.astype(np.int64, copy=False)).to(model_device)
        with torch.inference_mode():
            image = self.tokenizer.decode_index(tokens)
        return image


def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    """
    Compress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    images: Path to the image to compress.
    compressed_image: Path to save the compressed image tensor.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_img = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_img)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    """
    Decompress images using a pre-trained model.

    tokenizer: Path to the tokenizer model.
    autoregressive: Path to the autoregressive model.
    compressed_image: Path to the compressed image tensor.
    images: Path to save the image to compress.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_img = f.read()

    x = cmp.decompress(cmp_img)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire

    Fire({"compress": compress, "decompress": decompress})
