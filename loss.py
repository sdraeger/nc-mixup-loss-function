import torch.nn as nn
import torch.nn.functional as F


def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(
        pred, y_b
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, size_average="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average == "mean":
            return loss.mean()
        else:
            return loss.sum()


class ScaledMSELoss(nn.MSELoss):
    """Scaled MSE Loss"""

    def __init__(self, *args, kappa: float = 1.0, gamma: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = kappa
        self.gamma = gamma

    # def forward(self, ypred, y):
    #     idx_y = y
    #     y = F.one_hot(y, num_classes=ypred.size(1)).type(torch.float32)

    #     kappa_sqrt = torch.sqrt(self.kappa * torch.tensor(1))

    #     y_hat = ypred.clone()
    #     y_gt = y.clone()

    #     y_hat[torch.arange(y_hat.size(0)), idx_y] *= kappa_sqrt
    #     y_gt[torch.arange(y_gt.size(0)), idx_y] = self.gamma * kappa_sqrt

    #     return super().forward(ypred, y)

    def forward(self, ypred, y):
        return F.mse_loss(
            ypred,
            self.kappa * F.one_hot(y, num_classes=ypred.size(1)).float(),
        )
