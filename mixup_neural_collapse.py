import torch
import torchvision

import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Subset
import click

from model import wide_resnet_40x10
from mixup import mixup_data
from loss import mixup_criterion
from utils import get_color, plot_last_layer


# Set seeds
torch.manual_seed(0)
torch.cuda.manual_seed(0)


def train(net, epoch, trainloader, criterion, device, optimizer):
    pbar = tqdm(total=len(trainloader), position=0, leave=True)

    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets_a, targets_b, lambda_ = mixup_data(inputs, targets, alpha=1.0)

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
            "Train\t\tEpoch: {} [{}/{} ({:.0f}%)] \t"
            "Batch Loss: {:.6f} \t".format(
                epoch,
                batch_idx,
                len(trainloader),
                100.0 * batch_idx / len(trainloader),
                loss.item(),
            )
        )

    pbar.close()

    return train_loss / (batch_idx + 1)


def test(net, epoch, testloader, criterion, device):
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
                "Test\t\tEpoch: {} [{}/{} ({:.0f}%)] \t"
                "Batch Loss: {:.6f} \t"
                "Batch Accuracy: {:.6f}".format(
                    epoch,
                    batch_idx,
                    len(testloader),
                    100.0 * batch_idx / len(testloader),
                    loss.item(),
                    100.0 * correct / total,
                )
            )

    pbar.close()
    acc = 100.0 * correct / total
    return test_loss / (batch_idx + 1), acc


def get_last_layer(
    train_subset_loader, net, device, distribution="uniform", alph=1.0, num_loops=1
):
    print("Getting last layer features")

    class features:
        pass

    def hook(self, input, output):
        features.value = input[0].clone()

    classifier = net.linear
    classifier.register_forward_hook(hook)
    net.train()

    count = 0
    for i in range(num_loops):
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(train_subset_loader, start=1):
                inputs, targets = inputs.to(device), targets.to(device)

                inputs_mixed, targets_a, targets_b, lambda_ = mixup_data(
                    inputs, targets, dist=distribution, alpha=alph, use_cuda=True
                )

                colour_class_check = get_color(
                    targets_a, targets_b, lam=lambda_, class_check=True, use_cuda=True
                )

                outputs_mixed = net(inputs_mixed)
                h = features.value.data.view(inputs_mixed.shape[0], -1).detach()

                if count == 0 and i == 0:
                    H = h
                    colours_class_check = colour_class_check
                    count += 1

                else:
                    H = torch.cat((H, h))
                    colours_class_check = torch.cat(
                        (colours_class_check, colour_class_check)
                    )
                    count += 1

    return H, colours_class_check


def get_data():
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        "./data", train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        "./data", train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    targets_subset = list(np.random.choice(10, 3, replace=False))
    indices = [i for i, label in enumerate(trainset.targets) if label in targets_subset]

    # Subset the data
    dataset_subset = Subset(trainset, indices)
    train_subset_loader = torch.utils.data.DataLoader(
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
def main(device):
    learning_rate = 0.1
    weight_decay = 1e-4
    num_classes = 10
    epochs = 500
    epochs_list = [0, 2, 4, 8, 16, 32, 64, 128, 200, 250, 300, 350, 400, 499]

    net = wide_resnet_40x10()
    net = net.to(device)

    optimizer = optim.SGD(
        net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.3 * epochs), int(0.5 * epochs), int(0.9 * epochs)],
        gamma=0.1,
    )

    data_dict = get_data()
    trainloader = data_dict["trainloader"]
    testloader = data_dict["testloader"]
    targets_subset = data_dict["targets_subset"]

    for epoch in range(0, 500):
        train(net, epoch, trainloader)
        scheduler.step()

        loss, acc = test(net, epoch, testloader)

        if epoch in epochs_list:
            W = net.linear.weight[targets_subset].T.cpu().data.numpy()
            H, colours_class = get_last_layer(num_loops=1)

            colours_class = colours_class.cpu().data.numpy()
            H = H.cpu()
            H = np.array(H)

            plot_last_layer(
                H, W, colours_class, epoch, title="CIFAR10 WRN40x10 Epoch " + str(epoch)
            )


if __name__ == "__main__":
    main()
