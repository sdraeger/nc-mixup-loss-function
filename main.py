import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
import click
import polars as pl

from const import (
    DATASETS,
    NUM_CLASSES,
    MODELS,
    LOSSES,
    TRANSFORMS_TRAIN,
    TRANSFORMS_TEST,
)
from mixup import mixup_data
from loss import mixup_criterion
from utils import get_color, plot_last_layer, get_classifier_layer


def train(net, epoch, trainloader, criterion, device, optimizer, num_classes):
    """
    Trains the neural network model for one epoch.

    Args:
        net (torch.nn.Module): The neural network model.
        epoch (int): The current epoch number.
        trainloader (torch.utils.data.DataLoader): The data loader for training data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to perform computations on.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.

    Returns:
        tuple: A tuple containing the average training loss and the accuracy of the model.
    """

    pbar = tqdm(total=len(trainloader), position=0, leave=True)

    print(f"\nEpoch: {epoch}")

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lambda_ = mixup_data(
            inputs, targets, alpha=1.0, device=device, num_classes=num_classes
        )

        outputs = net(inputs)
        loss_func = mixup_criterion(targets_a, targets_b, lambda_)
        loss = loss_func(criterion, outputs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.update(1)
        pbar.set_description(
            f"Train\t\tEpoch: {epoch} [{batch_idx}/{len(trainloader)} ({(100.0 * batch_idx / len(trainloader)):.0f}%)] \t"
            f"Batch Loss: {loss.item():.6f} \t"
        )

    pbar.close()

    acc = 100.0 * correct / total
    return train_loss / (batch_idx + 1), acc


def test(net, epoch, testloader, criterion, device):
    """
    Evaluate the performance of a neural network model on a test dataset.
    Args:
        net (torch.nn.Module): The neural network model to be evaluated.
        epoch (int): The current epoch number.
        testloader (torch.utils.data.DataLoader): The data loader for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.
        device (torch.device): The device on which the evaluation will be performed.
    Returns:
        tuple: A tuple containing the average test loss and the accuracy of the model on the test dataset.
    """

    pbar = tqdm(total=len(testloader), position=0, leave=True)

    net.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.update(1)
            pbar.set_description(
                f"Test\t\tEpoch: {epoch} [{batch_idx}/{len(testloader)} ({(100.0 * batch_idx / len(testloader)):.0f}%)] \t"
                f"Batch Loss: {loss.item():.6f} \t"
                f"Batch Accuracy: {(100.0 * correct / total):.6f}"
            )

    pbar.close()

    acc = 100.0 * correct / total
    return test_loss / (batch_idx + 1), acc


def get_last_layer(
    train_subset_loader,
    net,
    device,
    num_classes,
    distribution="uniform",
    alph=1.0,
    num_loops=1,
):
    """
    Retrieves the last layer features of a neural network model.
    Args:
        train_subset_loader (torch.utils.data.DataLoader): The data loader for the training subset.
        net (torch.nn.Module): The neural network model.
        device (torch.device): The device to perform computations on.
        distribution (str, optional): The distribution type for mixup data augmentation. Defaults to "uniform".
        alph (float, optional): The alpha value for mixup data augmentation. Defaults to 1.0.
        num_loops (int, optional): The number of loops to iterate over the training subset. Defaults to 1.
    Returns:
        torch.Tensor: The last layer features of the neural network model.
        torch.Tensor: The color class check values.
    """

    print("Getting last layer features")

    class features:
        pass

    def hook(self, input, output):
        features.value = input[0].clone()

    classifier = get_classifier_layer(net)
    classifier.register_forward_hook(hook)
    net.train()

    count = 0
    for i in range(num_loops):
        with torch.no_grad():
            for inputs, targets in train_subset_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                inputs_mixed, targets_a, targets_b, lambda_ = mixup_data(
                    inputs,
                    targets,
                    dist=distribution,
                    alpha=alph,
                    device=device,
                    num_classes=num_classes,
                )

                color_class_check = get_color(
                    targets_a, targets_b, lam=lambda_, class_check=True, use_cuda=True
                )

                _ = net(inputs_mixed)
                h = features.value.data.view(inputs_mixed.shape[0], -1).detach()

                if count == 0 and i == 0:
                    H = h
                    colors_class_check = color_class_check
                    count += 1
                else:
                    H = torch.cat((H, h))
                    colors_class_check = torch.cat(
                        (colors_class_check, color_class_check)
                    )
                    count += 1

    return H, colors_class_check


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

    trainset = dataset_cls(
        "./data", train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = dataset_cls(
        "./data", train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    targets_subset = list(np.random.choice(10, 3, replace=False))
    indices = [i for i, label in enumerate(trainset.targets) if label in targets_subset]

    # Subset the data
    dataset_subset = Subset(trainset, indices)
    train_subset_loader = DataLoader(
        dataset_subset, batch_size=128, shuffle=True, drop_last=True
    )

    return dict(
        trainloader=trainloader,
        testloader=testloader,
        targets_subset=targets_subset,
        train_subset_loader=train_subset_loader,
    )


@click.command()
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--dataset", required=True)
@click.option("--model", required=True)
@click.option("--seed", default=0)
@click.option("--loss_fun", required=True)
@click.option("--lr", default=0.1)
@click.option("--weight_decay", default=1e-4)
def main(device, dataset, model, seed, loss_fun, lr, weight_decay):
    # Set the seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    num_classes = NUM_CLASSES[dataset]
    epochs = 500
    epochs_list = [0, 2, 4, 8, 16, 32, 64, 128, 200, 250, 300, 350, 400, 499]

    # Define the model, optimizer, criterion, and scheduler
    net_cls = MODELS[model]
    net = net_cls(num_classes=num_classes)
    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    criterion = LOSSES[loss_fun]()
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.3 * epochs), int(0.5 * epochs), int(0.9 * epochs)],
        gamma=0.1,
    )

    transform_train = TRANSFORMS_TRAIN[dataset]
    transform_test = TRANSFORMS_TEST[dataset]
    dataset_cls = DATASETS[dataset]
    data_dict = get_data(dataset_cls, transform_train, transform_test)
    trainloader = data_dict["trainloader"]
    testloader = data_dict["testloader"]
    targets_subset = data_dict["targets_subset"]

    # Create a DataFrame to store metrics
    metrics = pl.DataFrame(
        {
            "epoch": [],
            "train_loss": [],
            "test_loss": [],
            "test_acc": [],
        }
    )

    try:
        # Train the model
        for epoch in range(epochs):
            train_loss, train_acc = train(
                net, epoch, trainloader, criterion, device, optimizer
            )
            test_loss, test_acc = test(net, epoch, testloader, criterion, device)

            metrics.vstack(
                [
                    pl.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [train_loss],
                            "train_acc": [train_acc],
                            "test_loss": [test_loss],
                            "test_acc": [test_acc],
                        }
                    )
                ],
                in_place=True,
            )

            scheduler.step()

            # If epoch is in the list of epochs to plot, get the last layer features and plot them
            if epoch in epochs_list:
                W = net.linear.weight[targets_subset].T.cpu().data.numpy()
                H, colors_class = get_last_layer(num_loops=1)

                colors_class = colors_class.cpu().data.numpy()
                H = H.cpu()
                H = np.array(H)

                plot_title = f"{dataset_cls.__name__} {net_cls.__name__} Epoch {epoch}"
                fig, _ = plot_last_layer(H, W, colors_class, epoch, title=plot_title)
                fig.savefig(
                    f"plots/{dataset}_{model}_{loss_fun}_epoch_{epoch}_seed_{seed}.png"
                )
    except KeyboardInterrupt:
        print("\nCTRL+C detected. Saving metrics to CSV and exiting gracefully...")

    metrics.write_csv(f"logs/{dataset}_{model}_{loss_fun}_seed_{seed}_metrics.csv")


if __name__ == "__main__":
    main()
