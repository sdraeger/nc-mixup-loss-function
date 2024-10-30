import torch
import numpy as np


def mixup_data(
    x,
    y,
    device,
    num_classes,
    dist="beta",
    lambda_=None,
    alpha=1.0,
    same_class=False,
):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""

    # Find batch size
    batch_size = x.size()[0]

    # Pick distribution to sample lambda from.
    if lambda_ is None:
        if dist == "beta":
            lam = 0 if alpha == 0 else np.random.beta(alpha, alpha)
        elif dist == "uniform":
            lam = np.random.uniform(0, 1)
    else:
        lam = lambda_

    # Check if same class mixup or regular mixup.
    if same_class:
        index = torch.zeros(y.shape).long().to(device=device)
        for i in range(num_classes):
            mask = torch.nonzero(torch.where(y == i, True, False))
            index[torch.flatten(mask)] = torch.flatten(
                mask[torch.randperm(mask.shape[0])]
            )
    else:
        index = torch.randperm(batch_size).to(device=device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
