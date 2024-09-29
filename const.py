import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from model import wide_resnet_40x10


# Define choices for dataset, model, and loss function
DATASETS = {
    "cifar10": torchvision.datasets.CIFAR10,
    "mnist": torchvision.datasets.MNIST,
    "fashionmnist": torchvision.datasets.FashionMNIST,
}

NUM_CLASSES = {
    "cifar10": 10,
    "mnist": 10,
    "fashionmnist": 10,
}

MODELS = {
    "wide_resnet_40x10": wide_resnet_40x10,
    "wide_resnet_50x2": torchvision.models.wide_resnet50_2,
    "resnet18": torchvision.models.resnet18,
}

LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}

TRANSFORMS_TRAIN = {
    "cifar10": transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "mnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "fashionmnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
}

TRANSFORMS_TEST = {
    "cifar10": transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
    "mnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
    "fashionmnist": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
}
