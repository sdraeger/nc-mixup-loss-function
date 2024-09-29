import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def get_color(y_a, y_b, lam, use_cuda=True, class_check=False):
    batch_size = y_a.size()[0]

    if use_cuda:
        colours = torch.zeros((batch_size, 3)).cuda()
    else:
        colours = torch.zeros((batch_size, 3))

    for i in range(batch_size):
        if class_check:
            if y_a[i] == y_b[i]:
                colours[i][0] = 2 * abs(lam - 0.5)
            else:
                colours[i][2] = 2 * abs(lam - 0.5)

        else:
            colours[i][y_a[i]] += lam
            colours[i][y_b[i]] += 1 - lam

    return colours


def plot_last_layer(
    features,
    classifier,
    colour,
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
    ax.scatter(X[:, 0], X[:, 1], c=colour, marker=".", s=2.5)
    fig.title(title)

    return fig, ax


def get_classifier_layer(model: nn.Module):
    for attr in ["linear", "fc"]:
        if hasattr(model, attr):
            return getattr(model, attr)

    return None
