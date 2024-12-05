import numpy as np
from torch.utils.data import DataLoader, Subset


def get_data(dataset_cls, transform_train, transform_test):
    """
    Retrieves and prepares the data for training and testing.
    Args:
        dataset_cls (torchvision.datasets.Dataset): The dataset class to use.
        transform_train (torchvision.transforms.Transform): The transformation to apply to the training data.
        transform_test (torchvision.transforms.Transform): The transformation to apply to the testing data.
    Returns:
        dict: A dictionary containing the following keys:
            - trainloader (torch.utils.data.DataLoader): The data loader for the training set.
            - testloader (torch.utils.data.DataLoader): The data loader for the testing set.
            - targets_subset (list): A list of randomly chosen target labels.
            - train_subset_loader (torch.utils.data.DataLoader): The data loader for the subset of the training set.
    """

    batch_size = 128
    trainset = dataset_cls("data", train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    testset = dataset_cls("data", train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8
    )

    targets_subset = list(np.random.choice(10, 3, replace=False))
    indices = [i for i, label in enumerate(trainset.targets) if label in targets_subset]

    # Subset the data
    dataset_subset = Subset(trainset, indices)
    train_subset_loader = DataLoader(
        dataset_subset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return dict(
        trainloader=trainloader,
        testloader=testloader,
        targets_subset=targets_subset,
        train_subset_loader=train_subset_loader,
    )
