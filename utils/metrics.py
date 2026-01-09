# import numpy as np
# import torch
# import torch.nn.functional as F
#
# from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
#
#
#
# def dice_coef(output, target):
#     smooth = 1e-5
#
#     output = torch.sigmoid(output).view(-1).data.cpu().numpy()
#     target = target.view(-1).data.cpu().numpy()
#     intersection = (output * target).sum()
#
#     return (2. * intersection + smooth) / \
#         (output.sum() + target.sum() + smooth)
#
#
# def iou_score(output, target):
#     smooth = 1e-5  # 新增平滑项
#     if torch.is_tensor(output):
#         output = torch.sigmoid(output).data.cpu().numpy()
#     if torch.is_tensor(target):
#         target = target.data.cpu().numpy()
#     output_ = output > 0.5
#     target_ = target > 0.5
#
#     # 原有指标计算
#     iou_ = jc(output_, target_)
#     dice_ = dc(output_, target_)
#     try:
#         hd_ = hd(output_, target_)
#     except:
#         hd_ = 0
#     try:
#         hd95_ = hd95(output_, target_)
#     except:
#         hd95_ = 0
#     recall_ = recall(output_, target_)
#     specificity_ = specificity(output_, target_)
#     precision_ = precision(output_, target_)
#
#     # 计算TP/TN/FP/FN用于计算准确率
#     tp = (output_ & target_).sum()
#     fp = (output_ & ~target_).sum()
#     fn = (~output_ & target_).sum()
#     tn = (~output_ & ~target_).sum()
#
#     # 计算准确率(Accuracy)
#     acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
#
#     f1 = (2 * precision_ * recall_ + smooth) / (precision_ + recall_ + smooth)
#
#     # 返回值新增f1_（顺序：iou, dice, hd, hd95, recall, specificity, precision, acc, f1）
#     return iou_, dice_, hd95_, recall_, specificity_, precision_, acc, f1


import numpy as np
import torch
import torch.nn.functional as F
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision, assd


def dice_coef(output, target):
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def iou_score(output, target):
    smooth = 1e-5
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    # 计算基础指标
    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)

    # 计算距离指标（包含异常处理）
    try:
        hd_ = hd(output_, target_)
    except:
        hd_ = 0
    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0
    try:
        asd_ = assd(output_, target_)
    except:
        asd_ = 0

    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)

    # 计算TP/TN/FP/FN用于计算准确率
    tp = (output_ & target_).sum()
    fp = (output_ & ~target_).sum()
    fn = (~output_ & target_).sum()
    tn = (~output_ & ~target_).sum()

    # 计算准确率(Accuracy)
    acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
    f1 = (2 * precision_ * recall_ + smooth) / (precision_ + recall_ + smooth)

    # 返回值顺序：iou, dice, hd, hd95, asd, recall, specificity, precision, acc, f1
    return iou_, dice_,  hd95_,  recall_, specificity_, precision_, acc, f1,asd_