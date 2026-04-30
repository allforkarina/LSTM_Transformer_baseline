from __future__ import annotations

"""CSI-to-COCO17 pose models with heatmap and regression decoders."""

import torch
from torch import nn
from torch.nn import functional as F


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class TemporalBlock(nn.Module):
    """Residual 1D temporal block over frame-level CSI features."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2     # dilation temporal convolution, padding to maintain the size
        self.net = nn.Sequential(                       # two cnn with norm and dropout, maintain the size.
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU(inplace=True)         # activate function: ReLU

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply one residual temporal convolution block."""

        return self.activation(features + self.net(features))


class CSI2PoseBackbone(nn.Module):
    """
    Encode CSI windows into per-frame temporal features.
    
    Input: CSI amplitude and phase-cosine, size = [B=64, T=297, A=3, C=2, F=114, t=10]
    batch_size, Time, antenna, channel, frequency, time_shot
    """

    def __init__(
        self,
        input_channels: int = 6,
        feature_dim: int = 128,
        temporal_layers: int = 3,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),    # 6 -> 32 channel, maintain size.
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),                           # frequency dimension downsampled by 2.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),                # 32 -> 64 channel, maintain size.
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 2)),                           # frequency dimension downsampled by 3, time dimension downsampled by 2.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),               # 64 -> 128 channel, maintain size.
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),                               #^ problem 01: global average pool the spatial and infra_time dimension, from [114, 10] -> [57, 10] -> [19, 5] -> [1, 1]
            nn.Flatten(),
            nn.Linear(128, feature_dim),                                # channel dim -> feature dim.
            nn.ReLU(inplace=True),
        )
        self.temporal_input = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.temporal_encoder = nn.Sequential(
            *[
                TemporalBlock(
                    channels=feature_dim,
                    kernel_size=temporal_kernel_size,
                    dilation=2 ** layer_index,              # dilation factor: 1, 2, 4.
                    dropout=dropout,
                )
                for layer_index in range(temporal_layers)   # 0, 1, 2.
            ]
        )

    def forward(self, csi_amplitude: torch.Tensor, csi_phase_cos: torch.Tensor) -> torch.Tensor:
        """Return temporal features shaped ``B x T x C``."""

        if csi_amplitude.shape != csi_phase_cos.shape:
            raise ValueError(
                "csi_amplitude and csi_phase_cos must share the same shape, "
                f"got {tuple(csi_amplitude.shape)} and {tuple(csi_phase_cos.shape)}"
            )
        if csi_amplitude.ndim != 5:
            raise ValueError(f"Expected CSI tensors shaped B x T x 3 x 114 x 10, got {tuple(csi_amplitude.shape)}")

        batch_size, window_size = csi_amplitude.shape[:2]                                                               # batch_size, window_size = 297
        csi_features = torch.cat([csi_amplitude, csi_phase_cos], dim=2)                                                 # concat the feature, channel size = 3 + 3 = 6
        frame_features = self.frame_encoder(csi_features.reshape(batch_size * window_size, *csi_features.shape[2:]))    # shape = [B*T, C*A, F, t], encode to [B*T, feature_dim]
        frame_features = frame_features.reshape(batch_size, window_size, -1)                                            # reshape back to [B, T, feature_dim]
        temporal_features = self.temporal_input(frame_features.transpose(1, 2))                                         # conv1d at Time dimension, [B, feature_dim, T].
        return self.temporal_encoder(temporal_features).transpose(1, 2)                                                 # [B, T, feature_dim], output the temporal features for each frame. 


class CSI2PoseHeatmapModel(nn.Module):
    """Estimate COCO17 pose with per-joint heatmaps and soft-argmax decoding."""

    def __init__(
        self,
        input_channels: int = 6,                    # C * A, Amp(3) + Phase_cos(3)
        feature_dim: int = 128,                     # feature dimension
        temporal_layers: int = 3,                   # inter-frame temporal layers: 3
        temporal_kernel_size: int = 3,              # kernel size of temporal convolution: 3
        dropout: float = 0.1,
        heatmap_size: int = 64,                     #^ problem 02: heatmap size, 64 is too small for keypoint
        num_joints: int = 17,                       # joint query number.
        softargmax_temperature: float = 0.05,       # temperature for softargmax.
    ) -> None:
        super().__init__()
        if heatmap_size <= 1:
            raise ValueError(f"heatmap_size must be greater than 1, got {heatmap_size}")
        if num_joints != len(COCO_KEYPOINT_NAMES):
            raise ValueError(f"Only COCO17 is supported, got num_joints={num_joints}")

        self.heatmap_size = int(heatmap_size)
        self.num_joints = int(num_joints)
        self.softargmax_temperature = float(softargmax_temperature)
        self.backbone = CSI2PoseBackbone(
            input_channels=input_channels,
            feature_dim=feature_dim,
            temporal_layers=temporal_layers,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.joint_queries = nn.Parameter(torch.randn(num_joints, feature_dim) * 0.02)  #* Learnable joint query, change different size to test better performance.
        self.joint_norm = nn.LayerNorm(feature_dim)
        self.heatmap_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, heatmap_size * heatmap_size),                        # linear regression to heatmap
        )

        coordinates = torch.linspace(0.0, 1.0, steps=heatmap_size)                      # coordinates from 0 to 1 with step = heatmap_size.
        grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")        # reshape to heatmap
        self.register_buffer("grid_x", grid_x.reshape(1, 1, 1, -1), persistent=False)
        self.register_buffer("grid_y", grid_y.reshape(1, 1, 1, -1), persistent=False)

    def forward(self, csi_amplitude: torch.Tensor, csi_phase_cos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run heatmap-based CSI pose estimation."""

        temporal_features = self.backbone(csi_amplitude, csi_phase_cos)                                         # [B, T, feature_dim]
        batch_size, window_size = temporal_features.shape[:2]                                                   # get shape
        joint_features = temporal_features.unsqueeze(2) + self.joint_queries.view(1, 1, self.num_joints, -1)    #^ problem 03: independent query [B, T, 1, feature_dim] + [1, 1, num_joint, query_dim]
        joint_features = self.joint_norm(joint_features)                                                        #^ problem 03: the query is add, not expand. [B, T, num_joint, feature_dim]
        heatmap_logits = self.heatmap_head(joint_features)                                                      # predict the heatmap
        heatmaps = heatmap_logits.reshape(
            batch_size,
            window_size,
            self.num_joints,
            self.heatmap_size,
            self.heatmap_size,
        )
        return {"keypoints": self._softargmax(heatmaps), "heatmaps": heatmaps}

    def _softargmax(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Convert heatmaps to normalized ``x, y`` coordinates by expectation."""

        flat_heatmaps = heatmaps.reshape(*heatmaps.shape[:3], -1)
        probabilities = F.softmax(flat_heatmaps / self.softargmax_temperature, dim=-1)
        x_coordinates = torch.sum(probabilities * self.grid_x, dim=-1)
        y_coordinates = torch.sum(probabilities * self.grid_y, dim=-1)
        return torch.stack([x_coordinates, y_coordinates], dim=-1)


class CSI2PoseRegressionModel(nn.Module):
    """Estimate COCO17 pose by direct regression from joint-query features."""

    def __init__(
        self,
        input_channels: int = 6,
        feature_dim: int = 128,
        temporal_layers: int = 3,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
        num_joints: int = 17,
    ) -> None:
        super().__init__()
        if num_joints != len(COCO_KEYPOINT_NAMES):
            raise ValueError(f"Only COCO17 is supported, got num_joints={num_joints}")

        self.num_joints = int(num_joints)
        self.backbone = CSI2PoseBackbone(               # output inter-frame temporal features, [B, T, feature_dim]
            input_channels=input_channels,
            feature_dim=feature_dim,
            temporal_layers=temporal_layers,
            temporal_kernel_size=temporal_kernel_size,
            dropout=dropout,
        )
        self.joint_queries = nn.Parameter(torch.randn(num_joints, feature_dim) * 0.02)
        self.joint_norm = nn.LayerNorm(feature_dim)
        self.regression_head = nn.Sequential(           # double linear connection layer.
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2),
        )

    def forward(self, csi_amplitude: torch.Tensor, csi_phase_cos: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run direct-regression CSI pose estimation."""

        temporal_features = self.backbone(csi_amplitude, csi_phase_cos)                                         # [B, T, feature_dim]
        joint_features = temporal_features.unsqueeze(2) + self.joint_queries.view(1, 1, self.num_joints, -1)    # add query feature
        joint_features = self.joint_norm(joint_features)
        return {"keypoints": self.regression_head(joint_features)}                                              # direct regression to keypoints, [B, T, num_joint, 2]


CSI2PoseModel = CSI2PoseHeatmapModel
