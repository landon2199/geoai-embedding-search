"""Chip embeddings from the SSL4EO-S12 MoCo ResNet-50 in TorchGeo.

The weights were pretrained on 13-band Sentinel-2 L1C top-of-atmosphere
imagery (Wang et al. 2022). TorchGeo's transform for these weights
(`ResNet50_Weights.SENTINEL2_ALL_MOCO.transforms`) is Resize(256) ->
CenterCrop(224) -> Normalize(mean=0, std=10000), reproduced here
exactly. We feed L2A surface reflectance on
the same 0-10000 scale and insert a zero-filled B10 (cirrus) band, which
L2A does not carry. The L1C/L2A mismatch is a documented limitation.
"""

from __future__ import annotations

import numpy as np

# Band order expected by TorchGeo's SENTINEL2_ALL_* ResNet weights.
MODEL_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
               "B08", "B8A", "B09", "B10", "B11", "B12"]
SCALE = 10000.0
RESIZE = 256
MODEL_INPUT = 224
WEIGHTS_NAME = "SENTINEL2_ALL_MOCO"


def to_model_bands(chips: np.ndarray, bands: list[str]) -> np.ndarray:
    """Reorder (N, C, H, W) chips to MODEL_BANDS, zero-filling missing bands."""
    out = np.zeros((chips.shape[0], len(MODEL_BANDS)) + chips.shape[2:], dtype=np.float32)
    for i, b in enumerate(MODEL_BANDS):
        if b in bands:
            out[:, i] = chips[:, bands.index(b)]
    return out


def normalize(x):
    """TorchGeo SSL4EO-S12 Sentinel-2 normalisation: (x - 0) / 10000."""
    return x / SCALE


def load_model(weights_name: str = WEIGHTS_NAME, device: str = "cuda"):
    """Backbone with the classifier removed, returning 2048-D pooled features."""
    from torchgeo.models import ResNet50_Weights, resnet50

    weights = getattr(ResNet50_Weights, weights_name)
    model = resnet50(weights=weights, num_classes=0)
    return model.eval().to(device), weights


def embed_chips(chips: np.ndarray, model, device: str = "cuda",
                batch_size: int = 128) -> np.ndarray:
    """Embed model-ordered chips (N, 13, h, w) on the 0-10000 scale."""
    import torch
    import torch.nn.functional as F

    feats = []
    use_amp = device.startswith("cuda")
    with torch.inference_mode():
        for i in range(0, len(chips), batch_size):
            x = torch.from_numpy(chips[i:i + batch_size]).to(device).float()
            x = normalize(x)
            x = F.interpolate(x, size=(RESIZE, RESIZE), mode="bilinear",
                              align_corners=False, antialias=True)
            off = (RESIZE - MODEL_INPUT) // 2
            x = x[..., off:off + MODEL_INPUT, off:off + MODEL_INPUT]
            with torch.autocast("cuda", enabled=use_amp):
                f = model(x)
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)
