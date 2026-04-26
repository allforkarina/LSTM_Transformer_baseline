from __future__ import annotations

import torch

from models import BaselineLSTMTransformer, BaselineModelConfig


def test_baseline_model_forward_shape() -> None:
    model = BaselineLSTMTransformer(BaselineModelConfig())
    csi_window = torch.randn(2, 16, 2, 3, 114, 10)

    output = model(csi_window)

    assert output.shape == (2, 16, 17, 2)
    assert torch.isfinite(output).all()
