from vtbench.models.chart_models.simplecnn import SimpleCNN
from vtbench.models.chart_models.deepcnn import DeepCNN
from vtbench.models.chart_models.resnet18 import ResNet18
from vtbench.models.chart_models.efficientnet import EfficientNetB0
from vtbench.models.chart_models.vit_tiny import ViTTiny


def get_chart_model(name, input_channels=3, num_classes=None, pretrained=False,
                    image_size=128):
    """Factory for chart encoder models.

    Parameters
    ----------
    name : str
        Model name: simplecnn, deepcnn, resnet18, efficientnet, vit_tiny
    input_channels : int
        Number of input channels (default 3 for RGB).
    num_classes : int or None
        If None, model returns features only (for multimodal fusion).
    pretrained : bool
        Use pretrained weights (resnet18, efficientnet only).
    image_size : int
        Input image size (only used by vit_tiny for positional embeddings).
    """
    name = name.lower()
    if name == 'simplecnn':
        return SimpleCNN(input_channels=input_channels, num_classes=num_classes)
    elif name == 'deepcnn':
        return DeepCNN(input_channels=input_channels, num_classes=num_classes)
    elif name == 'resnet18':
        return ResNet18(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)
    elif name in ('efficientnet', 'efficientnet_b0'):
        return EfficientNetB0(input_channels=input_channels, num_classes=num_classes, pretrained=pretrained)
    elif name in ('vit_tiny', 'vit'):
        return ViTTiny(input_channels=input_channels, num_classes=num_classes,
                       pretrained=pretrained, image_size=image_size)
    else:
        raise ValueError(f"Unknown chart model: {name}")
