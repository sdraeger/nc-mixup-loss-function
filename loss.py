import torch.nn as nn


LOSSES = {
    "ce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "mae": nn.L1Loss,
    "smooth_l1": nn.SmoothL1Loss,
}


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )
