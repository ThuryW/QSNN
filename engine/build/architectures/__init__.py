__all__ = [
    'vgg11', 'spk_vgg11',
    'resnet20', 'spk_resnet20'
]

from .ann.vgg import vgg11
from .ann.resnet import resnet20
from .snn.vgg import spk_vgg11
from .snn.resnet import spk_resnet20