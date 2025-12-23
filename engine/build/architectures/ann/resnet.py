import torch
import torch.nn as nn
from ..quant_layers import QuantConv2d, QuantLinear


class ResidualBlock(nn.Module):

    def __init__(
        self,
        bits: int,
        num_in_channels: int,
        num_channels: int,
        stride: int,
        downsample = None
    ):
        super().__init__()
        self.conv1 = QuantConv2d(bits, num_in_channels, num_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.act1 = nn.ReLU()
        self.conv2 = QuantConv2d(bits, num_channels, num_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.downsample = downsample
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_branch = x
        if self.downsample:
            x_branch = self.downsample(x)
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        x = x + x_branch
        x = self.act2(x)
        return x


class ResNet(nn.Module):

    def __init__(
        self,
        bits: int,
        num_in_channels: int,
        num_in_features: int,
        num_classes: int,
        channels: list[int],
        blocks: list[int]
    ):
        super().__init__()
        if num_classes == 1000:
            self.conv1 = QuantConv2d(bits, 3, num_in_channels, 7, stride=2, padding=3, bias=False)
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.conv1 = QuantConv2d(bits, 3, num_in_channels, 3, stride=1, padding=1, bias=False)
            self.pool1 = nn.Identity()
        
        self.bn1 = nn.BatchNorm2d(num_in_channels)
        self.act1 = nn.ReLU()

        layers = []
        for i in range(len(blocks)):
            stride = 1 if i == 0 else 2
            layer, num_in_channels = self._make_layer(
                bits, num_in_channels, channels[i], stride, blocks[i]
            )
            layers.extend(layer)
        self.layers = nn.ModuleList(layers)
        
        self.pool2 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = QuantLinear(bits, num_in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.act1(x)
        x = self.pool1(x)
        for module in self.layers:
            x = module(x)
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    @staticmethod
    def _make_layer(
        bits,
        num_in_channels,
        num_channels,
        stride,
        num_blocks
    ) -> tuple[list, int]:
        
        downsample = None
        if stride != 1 or num_in_channels != num_channels:
            downsample = nn.Sequential(
                QuantConv2d(bits, num_in_channels, num_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels)
            )

        blocks = []
        blocks.append(ResidualBlock(bits, num_in_channels, num_channels, stride, downsample))
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(bits, num_channels, num_channels, stride=1))
        
        return blocks, num_channels


def resnet20(bits: int, num_classes: int=10):
    channels = [16, 32, 64]
    blocks = [3, 3, 3]
    return ResNet(bits, 16, 64, num_classes, channels, blocks)


if __name__ == '__main__':
    pass