import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabV3Wrapper(nn.Module):
    def __init__(self, num_classes: int = 21, pretrained_backbone: bool = True):
        super().__init__()
        weights_backbone = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None

        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)["out"]