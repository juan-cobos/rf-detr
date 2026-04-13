# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pathlib import Path
from typing import Optional

import torch
from torch import nn

size_to_entrypoint = {
    "tiny": "eupe_vitt16",
    "small": "eupe_vits16",
    "base": "eupe_vitb16",
}
size_to_embed_dim = {
    "tiny": 192,
    "small": 384,
    "base": 768,
}

DEFAULT_PATCH_SIZE = 16

# Path to the local EUPE repository, relative to this file's package root.
_EUPE_LOCAL_REPO = Path(__file__).resolve().parents[4] / "EUPE"


def load_model(size: str = "tiny", patch_size: int = 14) -> nn.Module:
    """Load an EUPE ViT model, preferring the local repo clone.

    Uses ``torch.hub.load`` with ``source='local'`` when the EUPE repo is
    present at ``<project_root>/EUPE``; falls back to loading from GitHub
    (``facebookresearch/EUPE``) otherwise.  Pretrained weights are downloaded
    from ``dl.fbaipublicfiles.com`` on first use and cached by torch.hub.

    Args:
        size: Model size — one of ``"tiny"``, ``"small"``, ``"base"``.
        patch_size: Patch size for the ViT. EUPE models default to 16 but can
            be adjusted to 14 for compatibility with RF-DETR.

    Returns:
        The loaded EUPE ViT model.
    """
    entrypoint = size_to_entrypoint[size]

    if _EUPE_LOCAL_REPO.exists():
        model = torch.hub.load(
            str(_EUPE_LOCAL_REPO),
            entrypoint,
            source="local",
            pretrained=True,
        )
    else:
        model = torch.hub.load(
            "facebookresearch/EUPE",
            entrypoint,
            pretrained=True,
        )

    return model


class EUPEEncoder(nn.Module):
    """EUPE ViT encoder compatible with RF-DETR's Backbone/MultiScaleProjector pipeline.

    Wraps the EUPE model and exposes intermediate spatial feature maps via
    ``forward_features``, matching the interface expected by
    ``rfdetr.models.backbone.backbone.Backbone``.

    Attributes:
        _out_feature_channels: List of channel widths, one per output feature level.
        shape: Target input shape (H, W) for position encoding interpolation.
        patch_size: Patch size used by the ViT.
        num_windows: Number of windows for windowed attention compatibility.
    """

    def __init__(
        self,
        size: str = "base",
        out_feature_indexes: Optional[list[int]] = None,
        shape: tuple[int, int] = (640, 640),
        patch_size: int = 14,
        num_windows: int = 4,
        gradient_checkpointing: bool = False,
    ):
        """Initialize EUPEEncoder.

        Args:
            size: Model size — one of ``"tiny"``, ``"small"``, ``"base"``.
            out_feature_indexes: Transformer block indices whose outputs are
                returned as spatial feature maps. Defaults to ``[9]``.
            shape: Target input shape (H, W) for the model.
            patch_size: Patch size for the ViT. Defaults to 14 for RF-DETR compatibility.
            num_windows: Number of windows for compatibility with windowed attention
                pipelines. Defaults to 4.
            gradient_checkpointing: Whether to use gradient checkpointing.
                Currently not supported for EUPE.
        """
        super().__init__()
        if out_feature_indexes is None:
            out_feature_indexes = [9]

        self.out_feature_indexes = out_feature_indexes
        self.shape = shape
        self.patch_size = patch_size
        self.num_windows = num_windows
        embed_dim = size_to_embed_dim[size]
        self._out_feature_channels = [embed_dim] * len(out_feature_indexes)

        if gradient_checkpointing:
            raise NotImplementedError("Gradient checkpointing is not yet supported for EUPE encoder.")

        self.encoder = load_model(size=size, patch_size=patch_size)
        self._export = False

    def export(self):
        """Prepare model for inference export.

        Handles position encoding interpolation to match target shape.
        """
        if self._export:
            return
        self._export = True

        # EUPE handles position encoding interpolation internally via
        # interpolate_pos_encoding in forward_features, so no explicit
        # interpolation is needed here. This method exists for interface
        # compatibility with DinoV2.

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate spatial features from EUPE ViT.

        Args:
            x: Input image tensor of shape ``[B, C, H, W]``. Images must be
                pre-normalised by the RF-DETR data pipeline (ImageNet stats).
                H and W must be divisible by ``patch_size * num_windows``.

        Returns:
            List of spatial feature tensors ``[B, C, H/ps, W/ps]``, one per
            entry in ``out_feature_indexes``, where ps is the effective patch
            stride (patch_size for EUPE).
        """
        block_size = self.patch_size * self.num_windows
        assert x.shape[2] % block_size == 0 and x.shape[3] % block_size == 0, (
            f"EUPEEncoder requires input H and W divisible by {block_size}, got {x.shape}"
        )

        # forward_features returns List[Dict[str, Tensor]] where each dict
        # contains features from a specific layer. We extract the normalized
        # patch tokens (x_norm_patchtokens) which have shape [B, N, C] where
        # N = (H/ps) * (W/ps), then reshape to [B, C, H/ps, W/ps].
        features_list = self.encoder.forward_features(
            x,
            masks=None,
        )

        # Extract features at specified indexes
        outputs = []
        for idx in self.out_feature_indexes:
            feat_dict = features_list[idx]
            # x_norm_patchtokens: [B, N, C] -> [B, H/ps, W/ps, C] -> [B, C, H/ps, W/ps]
            patch_tokens = feat_dict["x_norm_patchtokens"]
            B, N, C = patch_tokens.shape
            H_ps = x.shape[2] // self.patch_size
            W_ps = x.shape[3] // self.patch_size
            feat = patch_tokens.reshape(B, H_ps, W_ps, C).permute(0, 3, 1, 2)
            outputs.append(feat)

        return outputs


if __name__ == "__main__":
    model = EUPEEncoder(patch_size=14, num_windows=4)
    print("Loads correctly")

    # Test forward method with random input and intermediate layers
    batch_size = 1
    channels = 3
    height = 560  # Divisible by 14*4 = 56
    width = 560

    # Create random input tensor (simulating ImageNet-normalized images)
    x = torch.randn(batch_size, channels, height, width)

    # Run forward pass
    features = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Number of output features: {len(features)}")
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")

    # Verify output shapes
    expected_h = height // model.patch_size
    expected_w = width // model.patch_size
    embed_dim = size_to_embed_dim["base"]

    assert len(features) == len(model.out_feature_indexes), (
        f"Expected {len(model.out_feature_indexes)} features, got {len(features)}"
    )

    for i, feat in enumerate(features):
        assert feat.shape == (batch_size, embed_dim, expected_h, expected_w), (
            f"Feature {i} has unexpected shape: {feat.shape}, "
            f"expected ({batch_size}, {embed_dim}, {expected_h}, {expected_w})"
        )

    print("Forward pass test passed!")
