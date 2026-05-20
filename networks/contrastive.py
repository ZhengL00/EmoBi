import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        device = features.device
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(mask.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row  = torch.sum(positives_mask , axis=1)
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature

        loss = loss.mean()
        return loss