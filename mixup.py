import torch
import numpy as np


def mixup_data(
    x,
    y,
    dist="beta",
    lambda_=None,
    use_cuda=True,
    alpha=1.0,
    same_class=False,
    num_classes=10,
):
    """Compute the mixup data. Return mixed inputs, pairs of targets, and lambda"""

    # Find batch size
    batch_size = x.size()[0]

    # Pick distribution to sample lambda from.
    if lambda_ == None:
        if dist == "beta":
            if alpha == 0:
                lam = 0
            elif alpha > 0:
                lam = np.random.beta(alpha, alpha)
        elif dist == "uniform":
            lam = np.random.uniform(0, 1)
    else:
        lam = lambda_

    # Check if same class mixup or regular mixup.
    if same_class:
        if use_cuda:
            index = torch.zeros(y.shape).long().cuda()
            for i in range(num_classes):
                mask = torch.nonzero(torch.where(y == i, True, False))
                index[torch.flatten(mask)] = torch.flatten(
                    mask[torch.randperm(mask.shape[0])]
                )
        else:
            index = torch.zeros(y.shape).long().cuda()
            for i in range(num_classes):
                mask = torch.nonzero(torch.where(y == i, True, False))
                index[torch.flatten(mask)] = torch.flatten(
                    mask[torch.randperm(mask.shape[0])]
                )

    else:
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam
