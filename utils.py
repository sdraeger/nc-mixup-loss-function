import random
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import dill
import fire

from mixup import mixup_data


def usetex(tex: bool = True):
    plt.rcParams.update(
        {
            "text.usetex": tex,
        }
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_color(y_a, y_b, lam, device, class_check=False):
    batch_size = y_a.size()[0]
    colors = torch.zeros((batch_size, 3)).to(device=device)

    for i in range(batch_size):
        if class_check:
            idx = 0 if y_a[i] == y_b[i] else 2
            colors[i][idx] = 2 * abs(lam - 0.5)
        else:
            colors[i][y_a[i]] += lam
            colors[i][y_b[i]] += 1 - lam

    return colors


def plot_last_layer(
    features,
    classifier,
    color,
    epoch,
    title=None,
):
    print("Generating plots for Epoch " + str(epoch))

    H = torch.Tensor(features)
    M_ = torch.Tensor(classifier).T

    mu_g = torch.mean(M_, dim=0, keepdim=True)
    M = M_ - mu_g

    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    idxs = S > 1e-5
    U = U[:, idxs]
    S = S[idxs]
    Vh = Vh[idxs, :]

    Q = U @ Vh
    A = (
        2**0.5
        * torch.tensor([[0.5, -0.5, 0], [0, 0, (3 / 4) ** 0.5]])
        @ (torch.eye(3) - torch.ones(3) / 3)
    )

    X = (A @ Q @ (H - mu_g).T).T.cpu().data.numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X[:, 0], X[:, 1], c=color, marker=".", s=2.5)
    fig.suptitle(title)

    return fig, ax


def plot_last_layer_cli(
    pkl_dict_fname,
    epoch,
    out_filename,
    title=None,
):
    with open(pkl_dict_fname, "rb") as f:
        save_dict = dill.load(f)

    features = save_dict["H"]
    classifier = save_dict["W"]
    color = save_dict["colors_class"]

    fig, _ = plot_last_layer(features, classifier, color, epoch, title)
    fig.savefig(out_filename)


def get_save_dict_mean(*save_dict_list, out_fname):
    H = []
    W = []
    colors_class = []

    for save_dict_fname in save_dict_list:
        with open(save_dict_fname, "rb") as f:
            save_dict = dill.load(f)
        H.append(save_dict["H"])
        W.append(save_dict["W"])
        colors_class.append(save_dict["colors_class"])

    H = np.mean(np.array(H), axis=0)
    W = np.mean(np.array(W), axis=0)
    colors_class = np.mean(np.array(colors_class), axis=0)

    mean_dict = {"H": H, "W": W, "colors_class": colors_class}

    with open(out_fname, "wb") as f:
        dill.dump(mean_dict, f)


def get_classifier_layer(model: nn.Module):
    for attr in ["linear", "fc", "heads"]:
        if hasattr(model, attr):
            if attr == "heads":
                return model.heads.head
            else:
                return getattr(model, attr)

    return None


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
                    targets_a,
                    targets_b,
                    lam=lambda_,
                    device=device,
                    class_check=True,
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


class Selector:
    def usetex(self, tex: bool = True):
        usetex(tex)

    def plot_last_layer(
        self,
        pkl_dict_fname,
        epoch,
        out_filename,
        title=None,
    ):
        return plot_last_layer_cli(
            pkl_dict_fname,
            epoch,
            out_filename,
            title=title,
        )

    def get_save_dict_mean(self, *save_dict_list, out_fname):
        return get_save_dict_mean(*save_dict_list, out_fname=out_fname)


if __name__ == "__main__":
    fire.Fire(Selector)
