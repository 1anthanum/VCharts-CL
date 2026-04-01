"""
EfficientNet-B0 chart encoder.

Uses torchvision's pretrained EfficientNet-B0 (5.3M params) as a
feature extractor for chart images. Much stronger than DeepCNN (0.3M)
while still being lightweight.

feature_dim = 1280  (EfficientNet-B0 output before classifier)
"""

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetB0(nn.Module):
    """EfficientNet-B0 chart encoder with optional pretrained weights."""

    feature_dim = 1280

    def __init__(self, input_channels=3, num_classes=None, pretrained=False):
        super().__init__()

        if pretrained:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            backbone = models.efficientnet_b0(weights=weights)
        else:
            backbone = models.efficientnet_b0(weights=None)

        # Replace first conv if input_channels != 3
        if input_channels != 3:
            old_conv = backbone.features[0][0]
            backbone.features[0][0] = nn.Conv2d(
                input_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
