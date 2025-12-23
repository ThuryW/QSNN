import torch
import torch.nn as nn
from ..quant_layers import QuantConv2d, QuantLinear
from .neurons import IFNeuron, Encoder, Decoder


class VGG(nn.Module):

    def __init__(
        self,
        bits: int,
        timesteps: int,
        feature_cfg: list = None,
        classifier_cfg: list = None,
        num_in_features: int = 512,
        num_classes: int = 10
    ):
        super().__init__()
        features, classifier = self.make_layers(
            bits, timesteps, feature_cfg, classifier_cfg, 3, num_in_features, num_classes
        )
        self.features = nn.ModuleList(features)
        self.classifier = nn.ModuleList(classifier)
        self.encoder = Encoder(timesteps)
        self.decoder = Decoder(timesteps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for module in self.features:
            x = module(x)
        x = x.flatten(1)
        for module in self.classifier:
            x = module(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def make_layers(
        bits,
        timesteps,
        feature_cfg,
        classifier_cfg,
        num_in_channels,
        num_in_features,
        num_classes
    ) -> tuple[list, list]:
        
        features = []
        in_channels = num_in_channels
        for value in feature_cfg:
            if value == 'p':
                features += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                features += [
                    QuantConv2d(bits, in_channels, value, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(value),
                    IFNeuron(timesteps)
                ]
                in_channels = value

        classifier = []
        in_features = num_in_features
        for value in classifier_cfg:
            classifier += [
                QuantLinear(bits, in_features, value, bias=False),
                nn.BatchNorm1d(value),
                IFNeuron(timesteps)
            ]
            in_features = value
        classifier += [QuantLinear(bits, in_features, num_classes)]
        
        return features, classifier


def spk_vgg11(bits: int, timesteps: int, num_classes: int=10):
    feature_cfg = [
        64, 'p', 
        128, 'p', 
        256, 256, 'p', 
        512, 512, 'p', 
        512, 512, 'p'
    ]
    if num_classes == 1000:
        classifier_cfg = [4096, 4096]
        num_in_features = 512 * 49
    else:
        classifier_cfg = []
        num_in_features = 512

    return VGG(bits, timesteps, feature_cfg, classifier_cfg, num_in_features, num_classes)


if __name__ == '__main__':
    pass