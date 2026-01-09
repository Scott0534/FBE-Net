import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from typing import List, Dict, Tuple
import numpy as np
# # 修正2：设置edt别名，解决函数未定义问题
# from scipy.ndimage import distance_transform_edt as edt
#
__all__ = [ 'EdgeStructureCombinedLoss',"BCEDiceLoss"]





# 标准二分类Dice Loss（无加权，边界损失专用）
class StandardBinaryDiceLoss(nn.Module):
    """
    标准二分类Dice Loss
    不引入像素权重，仅计算预测与目标的基础Dice相似度
    输入：pred_logits（模型输出logits）、target（二值标签）
    输出：dice_loss（1 - dice系数）
    """
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Logits转0-1概率
        # pred = torch.sigmoid(pred_logits)

        # 展平计算
        pred_flat = pred_logits.view(-1)
        target_flat = target.view(-1)

        # 标准Dice公式
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_coeff

class BinaryDiceFocalLoss(nn.Module):
    """适用于超声边界的Dice+Focal结合损失（二分类，通用最优参数版）"""

    def __init__(
            self,
            smooth: float = 1e-5,  # 通用平滑项，无需轻易调整
            focal_gamma: float = 2.0,  # 最优聚焦系数，聚焦模糊边界
            focal_alpha: float = 0.25,  # 最优正负平衡，适配边界稀疏场景0.75
            loss_weight: float = 0.7  # 最优权重分配，平衡两个损失
    ):
        super().__init__()
        self.smooth = smooth
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.loss_weight = loss_weight

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        参数：
            pred_logits: 模型输出的边界预测logits（未经过sigmoid，数值更稳定）
            target: 真实边界掩码（二值，0=背景，1=边界，需与pred_logits同形状）
        返回：
            total_loss: Dice+Focal结合损失
        """
        # 1. 计算Dice Loss（优化边界重叠度，解决样本不均衡）
        # pred_prob = torch.sigmoid(pred_logits)  # logits转0-1概率
        # 展平张量（方便计算全局交集/并集）
        pred_flat =pred_logits.view(-1)
        target_flat = target.view(-1).float()  # 确保数据类型一致

        intersection = (pred_flat * target_flat).sum()  # 交集：预测为边界且真实为边界
        union = pred_flat.sum() + target_flat.sum()  # 并集：预测边界 + 真实边界
        dice_coeff = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_coeff  # Dice系数越大越好，损失取1-系数

        # 2. 计算Binary Focal Loss（聚焦难分样本，优化模糊边界）
        # 基于logits计算BCE，避免sigmoid后数值不稳定
        # bce_loss = F.binary_cross_entropy_with_logits(
        #     pred_logits, target.float(), reduction="none"
        # )
        bce_loss = F.binary_cross_entropy(pred_logits, target.float(), reduction="none")
        pt = torch.exp(-bce_loss)  # pt：模型对样本的置信度（越接近1，置信度越高）
        # Focal Loss核心：置信度越低（难分样本），损失权重越大
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        focal_loss = focal_loss.mean()  # 全局平均，得到最终Focal Loss

        # 3. 加权结合两个损失
        total_loss = self.loss_weight * dice_loss + (1 - self.loss_weight) * focal_loss
        return total_loss

class EdgeStructureCombinedLoss(nn.Module):
    """
    边界+结构组合损失，所有输入均为LOGITS，内部统一处理sigmoid
    输入：模型6个输出（边界预测logits、aux1-aux4 logits、最终预测logits）
    损失：边界(Dice+Focal)损失 + 辅助/最终的Structure Loss（加权和，含Hausdorff提升HD95）
    """

    def __init__(self, loss_weights: List[float] = None):
        super().__init__()
        # 边界损失：Dice+Focal结合损失（适配超声边界任务）
        self.edge_loss = BinaryDiceFocalLoss()
        # 设置默认权重：给final loss更高权重（更贴合实际训练需求）
        if loss_weights is None:
            loss_weights = [1, 1, 1, 1, 1,1]  # 修正：提升final loss权重
            # loss_weights = [1.0, 0.2, 0.3, 0.4, 0.5, 1.0]
        # 权重校验：6个权重对应（边界、aux1、aux2、aux3、aux4、final）
        assert len(loss_weights) == 6, "loss_weights需传入6个值，对应（边界、aux1、aux2、aux3、aux4、final）损失权重"
        self.loss_weights = loss_weights
        # 创建独立的Hausdorff损失实例
        self.BCEDiceLoss=BCEDiceLoss()

    def forward(
            self,
            model_outputs: List[torch.Tensor],  # 所有输出均为logits
            mask: torch.Tensor,  # 原始分割掩码
            edge_mask: torch.Tensor  # 真实边缘掩码
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        assert len(model_outputs) == 6, f"模型需返回6个输出，当前{len(model_outputs)}个"

        # 解包输出：均为logits
        edge_pred_logits, aux1_pred_logits, aux2_pred_logits, \
            aux3_pred_logits,  aux4_pred_logits,final_pred_logits = model_outputs

        # 1. 边界损失：Dice+Focal结合损失
        edge_loss = self.edge_loss(edge_pred_logits, edge_mask)

        # 2. 辅助/最终预测损失：Structure Loss
        aux1_loss = self.BCEDiceLoss(aux1_pred_logits, mask)
        aux2_loss = self.BCEDiceLoss(aux2_pred_logits, mask)
        aux3_loss = self.BCEDiceLoss(aux3_pred_logits, mask)
        aux4_loss = self.BCEDiceLoss(aux4_pred_logits, mask)
        final_loss = self.BCEDiceLoss(final_pred_logits, mask)

        # 3. 总损失（加权和）
        total_loss = (
                self.loss_weights[0] * edge_loss +
                self.loss_weights[1] * aux1_loss +
                self.loss_weights[2] * aux2_loss +
                self.loss_weights[3] * aux3_loss +
                self.loss_weights[4] * aux4_loss +
                self.loss_weights[5] * final_loss
        )

        # 4. 损失详情
        loss_dict = {
            "total_loss": total_loss.item(),
            "edge_loss": edge_loss.item(),
            "aux1_loss": aux1_loss.item(),
            "aux2_loss": aux2_loss.item(),
            "aux3_loss": aux3_loss.item(),
            "aux4_loss": aux4_loss.item(),
            "final_loss": final_loss.item(),
        }

        return total_loss, loss_dict



class BCEDiceLoss(nn.Module):
    """
    BCE-Dice组合损失（二分类），接收LOGITS输入，内部自动做sigmoid
    """
    def __init__(self):
        super().__init__()

    def forward(self, input_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. 计算BCE损失（直接用with_logits，内部已处理sigmoid）
        bce = F.binary_cross_entropy_with_logits(input_logits, target)
        input_sigmoid = torch.sigmoid(input_logits)  # 仅用于Dice计算
        # bce = F.binary_cross_entropy(input_logits, target)

        # 2. 计算Dice损失
        smooth = 1e-5
        num = target.size(0)
        input_flat = input_sigmoid .view(num, -1)
        target_flat = target.view(num, -1)

        intersection = (input_flat * target_flat).sum(1)
        dice = (2. * intersection + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        dice = 1 - dice.sum() / num

        # 3. 加权合并BCE和Dice
        return 0.5 * bce + dice
#

