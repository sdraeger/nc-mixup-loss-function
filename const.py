import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from model.wideresnet import wide_resnet_40x10, wide_resnet_50x2
from model.resnet import resnet18, resnet50
from loss import ScaledMSELoss


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

NUM_CHANNELS = {
    "cifar10": 3,
    "mnist": 1,
    "fashionmnist": 1,
}

MODELS = {
    "wide_resnet_40x10": wide_resnet_40x10,
    "wide_resnet_50x2": wide_resnet_50x2,
    "resnet18": resnet18,
    "resnet50": resnet50,
}

LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
    "scaled_mse": ScaledMSELoss,
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
