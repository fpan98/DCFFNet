import torch.nn as nn
from torch.nn import functional as F


def dice_coef(output, target):
    smooth = 1e-4
    b = output.shape[0]
    output = output.view(b, -1)
    target = target.view(b, -1)
    intersection = (output * target).sum(1)
    union = output.sum(1) + target.sum(1)
    coff = (2. * intersection + smooth) / (union + smooth)
    return 1 - coff.mean(0)


class dice_bce_loss(nn.Module):
    def __init__(self):
        super(dice_bce_loss, self).__init__()

    def forward(self, output, target):
        bce = F.binary_cross_entropy(output, target)
        dice = dice_coef(output, target)
        return bce + dice