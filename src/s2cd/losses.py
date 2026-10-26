"""Losses: weighted CrossEntropy + soft Dice (change pixels are sparse)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CEDiceLoss(nn.Module):
    """loss = ce_weight * CE + dice_weight * Dice (on the change class)."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, target)
        if self.dice_weight == 0:
            return self.ce_weight * ce

        # soft dice on the change class (index 1)
        probs = F.softmax(logits, dim=1)[:, 1]
        target_f = (target == 1).float()
        dims = (1, 2)
        inter = (probs * target_f).sum(dim=dims)
        card = probs.sum(dim=dims) + target_f.sum(dim=dims)
        dice = 1 - ((2 * inter + 1e-6) / (card + 1e-6)).mean()

        return self.ce_weight * ce + self.dice_weight * dice