import torch
import torch.nn as nn
import torch.nn.functional as F

### Focal Loss ####
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is None else torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()

        focal_weight = (1 - probs).pow(self.gamma)
        log_probs = torch.log(probs)
        loss = -focal_weight * log_probs * targets_one_hot

        if self.alpha is not None:
            alpha_t = self.alpha[targets].view(-1, 1)
            loss = loss * alpha_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

