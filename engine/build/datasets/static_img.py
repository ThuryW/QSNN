import os
from torchvision import datasets, transforms
import numpy as np
from PIL import Image


def get_dataset(dataset_name: str, arch_name: str, data_dir: str):
    train_transform_map = {
        'cifar10': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]),
        'cifar100': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            Cutout(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]),
        'imagenet_vgg16': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250]),
        ]),
        'imagenet_resnet34': transforms.Compose([
            transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4850, 0.4560, 0.4060],[0.2290, 0.2240, 0.2250])
        ])
    }
    test_transoform_map = {
        'cifar10': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ]),
        'cifar100': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]),
        'imagenet_vgg16': transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])
        ]),
        'imagenet_resnet34': transforms.Compose([
            transforms.Resize(size=235, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.4850, 0.4560, 0.4060],[0.2290, 0.2240, 0.2250])
        ])
    }

    train_transform = train_transform_map.get(f'{dataset_name}_{arch_name}', train_transform_map.get(dataset_name))
    test_transform = test_transoform_map.get(f'{dataset_name}_{arch_name}', test_transoform_map.get(dataset_name))

    # cifar10
    if dataset_name == 'cifar10':
        train_set = datasets.CIFAR10(root=os.path.join(data_dir, 'CIFAR10'), transform=train_transform)
        test_set = datasets.CIFAR10(root=os.path.join(data_dir, 'CIFAR10'), train=False, transform=test_transform)

    # cifar100
    elif dataset_name == 'cifar100':
        train_set = datasets.CIFAR100(root=os.path.join(data_dir, 'CIFAR100'), transform=train_transform)
        test_set = datasets.CIFAR100(root=os.path.join(data_dir, 'CIFAR100'), train=False, transform=test_transform)

    # imagenet
    elif dataset_name == 'imagenet':
        train_set = datasets.ImageFolder(root=os.path.join(data_dir, 'ImageNet2012', 'train'), transform=train_transform)
        test_set = datasets.ImageFolder(root=os.path.join(data_dir, 'ImageNet2012', 'val'), transform=test_transform)
    
    return train_set, test_set


class Cutout():

    def __init__(
        self, 
        length: int = 16
    ):
        self.length = length

    def __call__(self, img):
        img = np.array(img)

        top = np.random.randint(0 - self.length // 2, img.shape[0] - self.length)
        left = np.random.randint(0 - self.length // 2, img.shape[1] - self.length)
        bottom = top + self.length
        right = left + self.length
        
        top = 0 if top < 0 else top
        left = 0 if left < 0 else top
        
        img[top:bottom, left:right, :] = 0.
        
        img = Image.fromarray(img)
        return img