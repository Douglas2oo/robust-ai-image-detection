"""Torchvision ResNet-50 V2, one logit; compatible with the common label interface."""
def build_model(pretrained=True):
    from torch import nn
    from torchvision.models import resnet50, ResNet50_Weights
    m=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    m.fc=nn.Linear(m.fc.in_features,1)
    return m
