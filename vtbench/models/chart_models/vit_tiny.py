"""
Vision Transformer (ViT-Tiny) chart encoder.

A lightweight ViT variant suitable for chart image classification.
Uses 12 transformer layers with embedding dim 192, 3 heads.

If pretrained=True, loads torchvision's ViT_B_16 weights and projects
down — OR uses a from-scratch tiny config. For simplicity, this
implementation wraps torchvision's vit_b_16 with a feature projection.

feature_dim = 192
"""

import torch
import torch.nn as nn
import math


class ViTTiny(nn.Module):
    """
    Minimal Vision Transformer for chart images.

    Architecture:
      - Patch embedding (16x16 patches)
      - 6 transformer encoder layers (dim=192, 3 heads)
      - CLS token → linear classifier

    feature_dim = 192
    """

    feature_dim = 192

    def __init__(self, input_channels=3, num_classes=None, pretrained=False,
                 image_size=128, patch_size=16, embed_dim=192, depth=6, num_heads=3):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token output
        x = x[:, 0]  # (B, embed_dim)

        x = self.classifier(x)
        return x
