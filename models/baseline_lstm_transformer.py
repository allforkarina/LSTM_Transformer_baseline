from __future__ import annotations

"""Sequence-level CSI pose model with a CNN frame encoder and joint queries."""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class BaselineModelConfig:
    """Configuration for the sequence-level MM-Fi pose baseline."""

    input_modalities: int = 2
    antenna_dim: int = 3
    subcarrier_dim: int = 114
    shot_dim: int = 10
    frame_feature_dim: int = 256
    cnn_channels: int = 64
    transformer_layers: int = 4
    transformer_heads: int = 8
    transformer_ff_dim: int = 1024
    joint_query_dim: int = 64
    decoder_hidden_dim: int = 256
    dropout: float = 0.1
    num_keypoints: int = 17


class FrameCSIEncoder(nn.Module):
    """Encode one frame's CSI tensor into one compact feature vector.

    The model treats the two physical modalities and three antenna streams as
    channels, then applies 2D convolutions over ``subcarrier x CSI time shot``.
    """

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        input_channels = config.input_modalities * config.antenna_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, config.cnn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels, config.cnn_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(config.cnn_channels * 2, config.cnn_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(config.cnn_channels * 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.cnn_channels * 4, config.frame_feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.features(x))


class TemporalEncoder(nn.Module):
    """Model pose-relevant changes across consecutive CSI frames."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.frame_feature_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
        self.output_norm = nn.LayerNorm(config.frame_feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = x + self._sinusoidal_position_encoding(
            window_size=x.shape[1],
            feature_dim=x.shape[2],
            device=x.device,
            dtype=x.dtype,
        )
        return self.output_norm(self.encoder(encoded))

    @staticmethod
    def _sinusoidal_position_encoding(
        window_size: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a dynamic sinusoidal encoding for the current window length."""

        positions = torch.arange(window_size, device=device, dtype=dtype).unsqueeze(1)
        dimensions = torch.arange(0, feature_dim, 2, device=device, dtype=dtype)
        div_term = torch.exp(dimensions * (-math.log(10000.0) / feature_dim))
        encoding = torch.zeros(1, window_size, feature_dim, device=device, dtype=dtype)
        encoding[0, :, 0::2] = torch.sin(positions * div_term)
        encoding[0, :, 1::2] = torch.cos(positions * div_term[: encoding[0, :, 1::2].shape[-1]])
        return encoding


class JointQueryPoseDecoder(nn.Module):
    """Predict each COCO-17 joint with its own learnable semantic query."""

    def __init__(self, config: BaselineModelConfig) -> None:
        super().__init__()
        self.num_keypoints = config.num_keypoints
        self.joint_queries = nn.Parameter(torch.randn(config.num_keypoints, config.joint_query_dim) * 0.02)
        self.regressor = nn.Sequential(
            nn.Linear(config.frame_feature_dim + config.joint_query_dim, config.decoder_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.decoder_hidden_dim, config.decoder_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(config.decoder_hidden_dim // 2, 2),
        )

    def forward(self, frame_features: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, feature_dim = frame_features.shape
        features = frame_features.unsqueeze(2).expand(batch_size, window_size, self.num_keypoints, feature_dim)
        queries = self.joint_queries.view(1, 1, self.num_keypoints, -1).expand(batch_size, window_size, -1, -1)
        decoder_input = torch.cat((features, queries), dim=-1)
        return self.regressor(decoder_input)


class BaselineLSTMTransformer(nn.Module):
    """Sequence CSI baseline that predicts one pose for each frame in a window.

    Expected input shape is ``B x T x 2 x 3 x 114 x 10``. The output shape is
    ``B x T x 17 x 2`` using the existing ``[0, 1]`` normalized coordinate target.
    """

    def __init__(self, config: BaselineModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BaselineModelConfig()
        self.frame_encoder = FrameCSIEncoder(self.config)
        self.temporal_encoder = TemporalEncoder(self.config)
        self.pose_decoder = JointQueryPoseDecoder(self.config)

    def forward(self, csi_window: torch.Tensor) -> torch.Tensor:
        if csi_window.ndim != 6:
            raise ValueError(f"Expected CSI window shape B x T x C x 3 x 114 x 10, got {tuple(csi_window.shape)}")

        batch_size, window_size, modalities, antenna_dim, subcarrier_dim, shot_dim = csi_window.shape
        expected_shape = (
            self.config.input_modalities,
            self.config.antenna_dim,
            self.config.subcarrier_dim,
            self.config.shot_dim,
        )
        if (modalities, antenna_dim, subcarrier_dim, shot_dim) != expected_shape:
            raise ValueError(
                "Unexpected CSI window shape. "
                f"Expected trailing dimensions {expected_shape}, got {tuple(csi_window.shape[2:])}"
            )

        frame_input = csi_window.reshape(
            batch_size * window_size,
            modalities * antenna_dim,
            subcarrier_dim,
            shot_dim,
        )
        frame_features = self.frame_encoder(frame_input)
        frame_features = frame_features.view(batch_size, window_size, self.config.frame_feature_dim)
        temporal_features = self.temporal_encoder(frame_features)
        return self.pose_decoder(temporal_features)
