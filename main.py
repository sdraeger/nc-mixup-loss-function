import os
import os.path as osp

import torch
import torch.optim as optim
from tqdm import tqdm
import click
import polars as pl
from torchmetrics.classification import CalibrationError
import dill

from const import (
    DATASETS,
    NUM_CLASSES,
    NUM_CHANNELS,
    MODELS,
    LOSSES,
    TRANSFORMS_TRAIN,
    TRANSFORMS_TEST,
)
from mixup import mixup_data
from loss import mixup_criterion
from utils import set_seed, plot_last_layer, get_last_layer, get_classifier_layer
from data import get_data


def train(
    net,
    epoch,
    trainloader,
    criterion,
    device,
    optimizer,
    num_classes,
    use_mixup=True,
):
    """
    Trains the neural network model for one epoch.

    Args:
        net (torch.nn.Module): The neural network model.
        epoch (int): The current epoch number.
        trainloader (torch.utils.data.DataLoader): The data loader for training data.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to perform computations on.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        num_classes (int): The number of classes in the dataset.
        use_mixup (bool): Whether to use mixup.

    Returns:
        dict: A dictionary containing the average loss, accuracy, and ECE for the epoch
    """

    pbar = tqdm(total=len(trainloader), position=0, leave=True)

    print(f"\nEpoch: {epoch}")

    net.train()

    train_loss = 0
    correct = 0
    total = 0

    ece = CalibrationError(task="multiclass", num_classes=num_classes)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        progress = 100.0 * batch_idx / len(trainloader)
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = net(inputs)

        if use_mixup:
            inputs, targets_a, targets_b, lambda_ = mixup_data(
                inputs, targets, alpha=1.0, device=device, num_classes=num_classes
            )
            loss_func = mixup_criterion(targets_a, targets_b, lambda_)

        loss = (
            loss_func(criterion, outputs) if use_mixup else criterion(outputs, targets)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ece.update(outputs.softmax(dim=1), targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.update(1)
        pbar.set_description(
            f"Train\t\tEpoch: {epoch} [{batch_idx}/{len(trainloader)} ({progress:.0f}%)]\t"
            f"Batch Loss: {loss.item():.6f}\t"
            f"Batch Accuracy: {(100.0 * correct / total):.6f}\t"
            f"Batch ECE: {ece.compute().item():.6f}"
        )

    pbar.close()

    acc = 100.0 * correct / total
    avg_loss = train_loss / (batch_idx + 1)
    ece_value = ece.compute().item()

    return dict(loss=avg_loss, accuracy=acc, ece=ece_value)


def test(net, epoch, testloader, criterion, device, num_classes):
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

    ece = CalibrationError(task="multiclass", num_classes=num_classes)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            progress = 100.0 * batch_idx / len(testloader)

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            ece.update(outputs.softmax(dim=1), targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.update(1)
            pbar.set_description(
                f"Test\t\tEpoch: {epoch} [{batch_idx}/{len(testloader)} ({progress:.0f}%)]\t"
                f"Batch Loss: {loss.item():.6f}\t"
                f"Batch Accuracy: {(100.0 * correct / total):.6f}\t"
                f"Batch ECE: {ece.compute().item():.6f}"
            )

    pbar.close()

    acc = 100.0 * correct / total
    avg_loss = test_loss / (batch_idx + 1)
    ece_value = ece.compute().item()
    return dict(loss=avg_loss, accuracy=acc, ece=ece_value)


@click.command()
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
@click.option("--dataset", type=click.Choice(DATASETS.keys()), required=True)
@click.option("--model", type=click.Choice(MODELS.keys()), required=True)
@click.option("--seed", default=0)
@click.option("--loss_fun", type=click.Choice(LOSSES.keys()), required=True)
@click.option("--lr", default=0.1)
@click.option("--weight_decay", default=1e-4)
@click.option("--no_mixup", is_flag=True, default=False)
@click.option("--loss_kw", "-lkw", type=(str, float), multiple=True)
def main(
    device,
    dataset,
    model,
    seed,
    loss_fun,
    lr,
    weight_decay,
    no_mixup,
    loss_kw,
):
    # Set the seed for reproducibility
    set_seed(seed)

    num_classes = NUM_CLASSES[dataset]
    num_channels = NUM_CHANNELS[dataset]

    log_subdir = f"{dataset}_{model}_{loss_fun}_seed_{seed}"

    epochs = 500
    epochs_list = [0, 2, 4, 8, 16, 32, 64, 128, 200, 250, 300, 350, 400, 499]

    # Define the model, optimizer, criterion, and scheduler
    net_cls = MODELS[model]
    net = net_cls(num_classes=num_classes, num_channels=num_channels)
    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
    )
    criterion = LOSSES[loss_fun](**dict(loss_kw))
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
    train_subset_loader = data_dict["train_subset_loader"]

    os.makedirs(osp.join("logs", log_subdir), exist_ok=True)
    os.makedirs(osp.join("plots", log_subdir), exist_ok=True)

    # Create a DataFrame to store metrics
    metrics = pl.DataFrame(
        {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }
    )

    try:
        # Train the model
        for epoch in range(1, epochs + 1):
            train_dict = train(
                net,
                epoch,
                trainloader,
                criterion,
                device,
                optimizer,
                num_classes,
                use_mixup=not no_mixup,
            )
            test_dict = test(net, epoch, testloader, criterion, device, num_classes)

            metrics = pl.concat(
                [
                    metrics,
                    pl.DataFrame(
                        {
                            "epoch": [epoch],
                            "train_loss": [train_dict["loss"]],
                            "train_acc": [train_dict["accuracy"]],
                            "test_loss": [test_dict["loss"]],
                            "test_acc": [test_dict["accuracy"]],
                        }
                    ),
                ],
                how="vertical_relaxed",
            )
            metrics.write_csv(osp.join("logs", log_subdir, "metrics.csv"))

            scheduler.step()

            # If the 0-indexed epoch is in the list of epochs to plot, get the last layer features and plot them
            if epoch - 1 in epochs_list:
                fc_layer = get_classifier_layer(net)
                W = fc_layer.weight[targets_subset].T.cpu().data.numpy()
                H, colors_class = get_last_layer(
                    train_subset_loader=train_subset_loader,
                    net=net,
                    device=device,
                    num_classes=num_classes,
                    num_loops=1,
                )

                colors_class = colors_class.cpu().data.numpy()
                H = H.cpu().numpy()

                # Save H, W, colors_class to disk
                save_dict = {
                    "H": H,
                    "W": W,
                    "colors_class": colors_class,
                }
                with open(
                    osp.join("logs", log_subdir, f"H_W_colors_class_epoch_{epoch}.pkl"),
                    "wb",
                ) as f:
                    dill.dump(save_dict, f)

                plot_title = f"{dataset_cls.__name__} {net_cls.__name__} Epoch {epoch}"
                fig, _ = plot_last_layer(H, W, colors_class, epoch, title=plot_title)
                fig.savefig(
                    osp.join(
                        "plots",
                        log_subdir,
                        f"last_layer_epoch_{epoch}.png",
                    )
                )
    except KeyboardInterrupt:
        print("\nCTRL+C detected. Saving metrics to CSV and exiting gracefully...")


if __name__ == "__main__":
    main()
