"""Модель корректора центра: ResNet18 + эмбеддинг камеры.

Без номера камеры модель не может выучить разные смещения пивота: у cam1 и cam4
они отличаются на сотни пикселей. Эмбеддинг камеры даёт модели этот контекст.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class CamResNet(nn.Module):
    def __init__(self, n_cams: int, emb: int = 8, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.emb = nn.Embedding(n_cams, emb)
        self.head = nn.Linear(feat + emb, 2)

    def forward(self, x: torch.Tensor, cam: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        e = self.emb(cam)
        return self.head(torch.cat([f, e], dim=1))
