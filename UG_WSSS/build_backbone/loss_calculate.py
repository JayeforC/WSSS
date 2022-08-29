import torch
import torch.nn as nn 
import torch.nn.functional as F


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)
    return (bg_loss + fg_loss) * 0.5

def get_aff_loss(inputs, targets):

    pos_label = (targets == 1).type(torch.int16)
    pos_count = pos_label.sum() + 1
    neg_label = (targets == 0).type(torch.int16)
    neg_count = neg_label.sum() + 1
    #inputs = torch.sigmoid(input=inputs)

    pos_loss = torch.sum(pos_label * (1 - inputs)) / pos_count
    neg_loss = torch.sum(neg_label * (inputs)) / neg_count

    return 0.5 * pos_loss + 0.5 * neg_loss, pos_count, neg_count

def get_uncertainty_loss(estimate_map,pseudo_label,BCE_coefficient=0.5,KL_coefficient=0.1):
    BCE_criterior = nn.CrossEntropyLoss(ignore_index=255)
    KL_divergence = nn.KLDivLoss(size_average=False,reduce=False)
    uncertainty_loss = BCE_coefficient * BCE_criterior(estimate_map,pseudo_label) + KL_coefficient * KL_divergence(estimate_map,pseudo_label)
    return uncertainty_loss

def get_cls_loss(cls_,cls_labels):
    cls_loss = F.multilabel_soft_margin_loss(cls_, cls_labels)
    return cls_loss