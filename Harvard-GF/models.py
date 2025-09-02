import torch
import torch.nn as nn
import torch.nn.functional as F
from models_resnet import ResNet18  # your ResNet implementation

class ClsResNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()
        self.backbone = ResNet18(in_channels=in_channels, num_classes=num_classes)

    def forward(self, x):
        return self.backbone(x)
