# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for EUPE backbone integration with RF-DETR."""

import pytest
import torch

from rfdetr.models.backbone.eupe import EUPEEncoder, load_model, size_to_embed_dim


class TestEUPEEncoder:
    """Test suite for EUPEEncoder backbone."""

    @pytest.mark.parametrize("size", ["tiny", "small", "base"])
    def test_eupe_encoder_initialization(self, size):
        """Test EUPEEncoder can be initialized with different sizes."""
        model = EUPEEncoder(size=size, patch_size=14, num_windows=4)
        assert model.patch_size == 14
        assert model.num_windows == 4
        assert hasattr(model, "_out_feature_channels")
        assert hasattr(model, "shape")
        assert hasattr(model, "encoder")

    @pytest.mark.parametrize(
        "size,out_feature_indexes,expected_num_features",
        [
            ("tiny", [3, 6, 9], 3),
            ("small", [3, 6, 9, 12], 4),
            ("base", [9], 1),
        ],
    )
    def test_out_feature_channels(self, size, out_feature_indexes, expected_num_features):
        """Test _out_feature_channels is correctly computed."""
        model = EUPEEncoder(
            size=size,
            out_feature_indexes=out_feature_indexes,
            patch_size=14,
            num_windows=4,
        )
        embed_dim = size_to_embed_dim[size]
        assert len(model._out_feature_channels) == expected_num_features
        assert all(ch == embed_dim for ch in model._out_feature_channels)

    @pytest.mark.parametrize("patch_size", [14, 16])
    @pytest.mark.parametrize("num_windows", [1, 2, 4])
    def test_forward_output_shapes(self, patch_size, num_windows):
        """Test forward pass produces correct output shapes."""
        model = EUPEEncoder(
            size="tiny",
            out_feature_indexes=[3, 6, 9],
            patch_size=patch_size,
            num_windows=num_windows,
        )
        model.eval()

        # Create input divisible by patch_size * num_windows
        block_size = patch_size * num_windows
        batch_size = 2
        height = 224
        width = 224
        # Adjust to be divisible by block_size
        height = (height // block_size) * block_size
        width = (width // block_size) * block_size

        x = torch.randn(batch_size, 3, height, width)

        with torch.no_grad():
            features = model(x)

        expected_h = height // patch_size
        expected_w = width // patch_size
        embed_dim = size_to_embed_dim["tiny"]

        assert len(features) == 3  # 3 output features
        for feat in features:
            assert feat.shape == (batch_size, embed_dim, expected_h, expected_w)

    def test_forward_invalid_input_size(self):
        """Test forward raises error for invalid input dimensions."""
        model = EUPEEncoder(
            size="tiny",
            patch_size=14,
            num_windows=4,
        )
        # 13 is not divisible by 14*4=56
        x = torch.randn(1, 3, 112, 112)

        with pytest.raises(AssertionError, match="divisible by"):
            model(x)

    def test_export_method(self):
        """Test export method exists and can be called."""
        model = EUPEEncoder(size="tiny", patch_size=14, num_windows=4)
        assert not model._export
        model.export()
        assert model._export
        # Calling export again should not raise error
        model.export()
        assert model._export

    def test_gradient_checkpointing_not_supported(self):
        """Test gradient checkpointing raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Gradient checkpointing"):
            EUPEEncoder(
                size="tiny",
                gradient_checkpointing=True,
            )


class TestEUPEBackboneIntegration:
    """Test EUPE integration with Backbone wrapper."""

    def test_backbone_with_eupe_encoder(self):
        """Test EUPEEncoder works within Backbone wrapper."""
        from rfdetr.models.backbone.backbone import Backbone

        backbone = Backbone(
            name="eupe_tiny",
            out_channels=256,
            out_feature_indexes=[3, 6, 9],
            projector_scale=["P3", "P4", "P5"],
            patch_size=14,
            num_windows=4,
            target_shape=(560, 560),
        )

        assert hasattr(backbone, "encoder")
        assert hasattr(backbone, "projector")
        assert isinstance(backbone.encoder, EUPEEncoder)

    def test_backbone_forward(self):
        """Test full backbone forward pass with EUPE."""
        from rfdetr.models.backbone.backbone import Backbone
        from rfdetr.utilities.tensors import NestedTensor

        backbone = Backbone(
            name="eupe_tiny",
            out_channels=256,
            out_feature_indexes=[3, 6, 9],
            projector_scale=["P4"],
            patch_size=14,
            num_windows=4,
            target_shape=(560, 560),
        )
        backbone.eval()

        batch_size = 2
        height = 560
        width = 560

        # Create NestedTensor input
        tensors = torch.randn(batch_size, 3, height, width)
        mask = torch.ones(batch_size, height, width, dtype=torch.bool)
        tensor_list = NestedTensor(tensors, mask)

        with torch.no_grad():
            outputs = backbone(tensor_list)

        assert len(outputs) == 1  # Only P4 scale
        assert isinstance(outputs[0], NestedTensor)
        # P4 scale should have spatial dims reduced by factor related to projector
        assert outputs[0].tensors.shape[0] == batch_size
        assert outputs[0].tensors.shape[1] == 256  # out_channels


class TestLoadModel:
    """Test EUPE model loading."""

    @pytest.mark.parametrize("size", ["tiny", "small", "base"])
    def test_load_model_returns_module(self, size):
        """Test load_model returns a valid nn.Module."""
        model = load_model(size=size)
        assert isinstance(model, torch.nn.Module)
        assert hasattr(model, "forward_features")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
