from __future__ import annotations

import torch

from models import BaselineLSTMTransformer, BaselineModelConfig


def test_baseline_model_forward_shape() -> None:
    model = BaselineLSTMTransformer(BaselineModelConfig())
    csi_amplitude = torch.randn(4, 3, 114, 10)
    csi_phase_cos = torch.randn(4, 3, 114, 10)

    output = model(csi_amplitude, csi_phase_cos)

    assert output.shape == (4, 17, 2)
    assert torch.isfinite(output).all()
