from __future__ import annotations

import torch

from models import BaselineLSTMTransformer, BaselineModelConfig


def test_baseline_model_forward_shape() -> None:
    model = BaselineLSTMTransformer(BaselineModelConfig())
    csi_window = torch.randn(2, 16, 2, 3, 114, 10)

    output = model(csi_window)

    assert output.shape == (2, 16, 17, 2)
    assert torch.isfinite(output).all()


def test_baseline_model_supports_dynamic_window_size() -> None:
    model = BaselineLSTMTransformer(BaselineModelConfig())
    csi_window = torch.randn(2, 8, 2, 3, 114, 10)

    output = model(csi_window)

    assert output.shape == (2, 8, 17, 2)
    assert torch.isfinite(output).all()


def test_model_config_uses_transformer_temporal_encoder() -> None:
    config = BaselineModelConfig()

    assert not hasattr(config, "gru_hidden_size")
    assert not hasattr(config, "gru_layers")
    assert config.transformer_layers == 4
    assert config.transformer_heads == 8
    assert config.transformer_ff_dim == 1024
