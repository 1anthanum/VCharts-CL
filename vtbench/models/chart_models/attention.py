"""
Channel Attention Modules for Chart Encoders
=============================================
SE-Net (Squeeze-and-Excitation) and CBAM (Convolutional Block Attention Module)
that can be applied to any CNN backbone to learn channel-wise importance.

Usage:
    model = get_chart_model('deepcnn_se', ...)   # DeepCNN + SE attention
    model = get_chart_model('resnet18_cbam', ...) # ResNet18 + CBAM
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block.

    Learns per-channel importance weights via global avg pool → FC → sigmoid.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class CBAMChannelAttention(nn.Module):
    """CBAM channel attention: max pool + avg pool → shared MLP → sigmoid."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w


class CBAMSpatialAttention(nn.Module):
    """CBAM spatial attention: channel max + avg → 7x7 conv → sigmoid."""

    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Full CBAM block: channel attention → spatial attention."""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = CBAMChannelAttention(channels, reduction)
        self.spatial = CBAMSpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


def wrap_model_with_attention(model, attention_type='se'):
    """
    Wrap a chart model's feature extractor with attention blocks.

    Inserts SE or CBAM after each conv block in the model's feature layers.
    Works with DeepCNN and ResNet18.

    Parameters
    ----------
    model : nn.Module
        A chart model from factory.py
    attention_type : str
        'se' for SE-Net, 'cbam' for CBAM

    Returns
    -------
    nn.Module
        Modified model with attention inserted.
    """
    AttentionClass = SEBlock if attention_type == 'se' else CBAM

    # For DeepCNN: insert after each conv block
    if hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        new_features = nn.Sequential()
        for i, layer in enumerate(model.features):
            new_features.add_module(str(i), layer)
            # Insert attention after ReLU layers (end of conv block)
            if isinstance(layer, nn.ReLU):
                # Find the preceding conv's out_channels
                for j in range(i - 1, -1, -1):
                    prev = model.features[j]
                    if isinstance(prev, (nn.Conv2d, nn.BatchNorm2d)):
                        channels = prev.num_features if isinstance(prev, nn.BatchNorm2d) else prev.out_channels
                        new_features.add_module(
                            f'attn_{i}', AttentionClass(channels)
                        )
                        break
        model.features = new_features

    return model
