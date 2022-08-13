from efficientnet_pytorch import EfficientNet
from torch import nn
import torch


class StripEfficientNet(nn.Module):

    def __init__(self, version, pretrained=False, pretrained_model=None, in_channels=3, num_classes=2):

        super().__init__()
        self.model = EfficientNet.from_name(version, in_channels=in_channels, num_classes=num_classes, image_size=(1024,1024))
        if pretrained:
            self.model.load_state_dict(pretrained_model)

    def forward(self, x):

        return self.model.forward(x)


if __name__ == "__main__":

    pretrained_model = torch.load("../checkpoint/pretrained/efficientnet-b4-e116e8b3.pth")
    efficientnet = StripEfficientNet("efficientnet-b4", False, pretrained_model, in_channels=48, num_classes=2)
    print(efficientnet.model)
    print("test")