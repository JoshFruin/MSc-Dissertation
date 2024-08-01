import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.class_weights = None

    def update_class_weights(self, targets):
        # Compute class weights once and store them
        if self.class_weights is None:
            class_weights = compute_class_weight('balanced', classes=np.unique(targets.cpu()),
                                                 y=targets.cpu().numpy())
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32).to(targets.device)

    def forward(self, inputs, targets):
        self.update_class_weights(targets)

        # Apply class weights to logits
        weighted_inputs = inputs * self.class_weights.unsqueeze(0)

        ce_loss = nn.functional.cross_entropy(weighted_inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss