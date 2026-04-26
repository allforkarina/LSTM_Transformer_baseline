from __future__ import annotations

"""Baseline LSTM + Transformer model for frame-level MM-Fi pose regression."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class BaselineModelConfig:
    """Fixed model configuration for the first baseline implementation."""

    time_steps: int = 10
    input_modalities: int = 2
    antenna_dim: int = 3
    subcarrier_dim: int = 114
    cnn_hidden_channels: int = 32
    cnn_output_dim: int = 128
    lstm_hidden_size: int = 128
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_dim: int = 512
    dropout: float = 0.1
    num_keypoints: int = 17


class TimeStepEncoder(nn.Module):
    """Encode one CSI time slice into one compact embedding."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        # Each time step is treated as a small two-channel CSI map:
        # - channel 0: amplitude
        # - channel 1: cosine-transformed phase
        # The encoder stays deliberately shallow because this repository targets
        # a readable first baseline rather than a heavily engineered backbone.
        self.features = nn.Sequential(
            nn.Conv2d(config.input_modalities, config.cnn_hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.cnn_hidden_channels,
                config.cnn_hidden_channels * 2,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 16)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.cnn_hidden_channels * 2 * 16, config.cnn_output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.projection(x)


class BaselineLSTMTransformer(nn.Module):
    """Lightweight frame-level baseline using CSI time steps as the temporal axis."""

    def __init__(self, config: BaselineModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaselineModelConfig()

        # The model follows one simple pipeline:
        # 1. encode each of the 10 CSI time slices independently
        # 2. model short-range temporal order with a bidirectional LSTM
        # 3. refine the temporal tokens with a Transformer encoder
        # 4. average the token sequence and regress one 17 x 2 pose
        self.time_step_encoder = TimeStepEncoder(self.config)
        self.temporal_lstm = nn.LSTM(
            input_size=self.config.cnn_output_dim,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        transformer_dim = self.config.lstm_hidden_size * 2
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=self.config.transformer_heads,
            dim_feedforward=self.config.transformer_ff_dim,
            dropout=self.config.dropout,
            batch_first=True,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.config.time_steps, transformer_dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.transformer_layers,
        )
        self.regressor = nn.Sequential(
            nn.Linear(transformer_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.config.dropout),
            nn.Linear(128, self.config.num_keypoints * 2),
        )

    def forward(self, csi_amplitude: torch.Tensor, csi_phase_cos: torch.Tensor) -> torch.Tensor:
        """Predict one normalized 17 x 2 pose from one batch of CSI tensors.

        Expected input shape:
        - ``csi_amplitude``: ``B x 3 x 114 x 10``
        - ``csi_phase_cos``: ``B x 3 x 114 x 10``

        The final axis with length ``10`` is the temporal axis modeled by the LSTM
        and Transformer. The network predicts one pose per frame, so the output shape
        is ``B x 17 x 2``.
        """

        if csi_amplitude.shape != csi_phase_cos.shape:
            raise ValueError(
                "csi_amplitude and csi_phase_cos must have the same shape, "
                f"got {tuple(csi_amplitude.shape)} and {tuple(csi_phase_cos.shape)}"
            )

        batch_size, antenna_dim, subcarrier_dim, time_steps = csi_amplitude.shape
        expected_shape = (
            self.config.antenna_dim,
            self.config.subcarrier_dim,
            self.config.time_steps,
        )
        if (antenna_dim, subcarrier_dim, time_steps) != expected_shape:
            raise ValueError(
                "Unexpected CSI shape. "
                f"Expected (B, {expected_shape[0]}, {expected_shape[1]}, {expected_shape[2]}), "
                f"got {tuple(csi_amplitude.shape)}"
            )

        # Stack the two CSI modalities into one channel dimension, then move the
        # 10 time steps forward so each time step can be encoded independently.
        time_major = torch.stack((csi_amplitude, csi_phase_cos), dim=1)
        time_major = time_major.permute(0, 4, 1, 2, 3).contiguous()
        encoded = self.time_step_encoder(
            time_major.view(batch_size * self.config.time_steps, self.config.input_modalities, antenna_dim, subcarrier_dim)
        )

        # Restore the batch/time layout for sequence modeling.
        encoded = encoded.view(batch_size, self.config.time_steps, self.config.cnn_output_dim)
        lstm_output, _ = self.temporal_lstm(encoded)
        transformer_input = lstm_output + self.position_embedding[:, : self.config.time_steps]
        transformer_output = self.transformer(transformer_input)

        # Mean pooling keeps the regression head minimal and makes the final tensor
        # independent of any single time step token.
        pooled = transformer_output.mean(dim=1)
        keypoints = self.regressor(pooled)
        return keypoints.view(batch_size, self.config.num_keypoints, 2)
