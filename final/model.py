import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def get_model(pretrained=True, num_classes=5):
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
