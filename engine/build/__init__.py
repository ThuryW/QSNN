from .architectures import *
from .datasets import static_img_dataset
from typing import Optional


ARCH_POOL = ['vgg11', 'resnet20']
DATASET_POOL = ['cifar10', 'cifar100', 'imagenet']


def dataset_dict(
    dataset_name: str, 
    arch_name: str,
    data_dir: str
):
    if dataset_name not in DATASET_POOL:
        raise NotImplementedError(f"Not supported dataset: Expecting one of {DATASET_POOL} but got {dataset_name}.")
    if arch_name not in ARCH_POOL:
        raise NotImplementedError(f"Not supported architecture: Expecting one of {ARCH_POOL} but got {arch_name}.")
    
    dataset_map = {
        'cifar10': static_img_dataset,
        'cifar100': static_img_dataset,
        'imagenet': static_img_dataset,
    }
    
    return dataset_map[dataset_name](dataset_name, arch_name, data_dir)


def arch_dict(
    spiking: bool,
    bits: int,
    timesteps: Optional[int] = None,
    arch_name: str = None,
    dataset_name: str = None
):
    if dataset_name not in DATASET_POOL:
        raise NotImplementedError(f"Not supported dataset: Expecting one of {DATASET_POOL} but got {dataset_name}.")
    if arch_name not in ARCH_POOL:
        raise NotImplementedError(f"Not supported architecture: Expecting one of {ARCH_POOL} but got {arch_name}.")
    
    num_classes_map = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000,
    }

    if spiking:
        arch_map = {
            'vgg11': spk_vgg11,
            'resnet20': spk_resnet20
        }

        return arch_map[arch_name](bits, timesteps, num_classes_map[dataset_name])
    
    else:
        arch_map = {
            'vgg11': vgg11,
            'resnet20': resnet20
        }

        return arch_map[arch_name](bits, num_classes_map[dataset_name])

    
