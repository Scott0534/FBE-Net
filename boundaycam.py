import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.new3 import Network  # 请确保你的网络导入正确
import matplotlib.pyplot as plt





# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------- Grad-CAM 核心类 --------------------------
class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.model.eval()
        self.target_layer = self._find_target_layer(target_layer_name)

        self.feature_map = None
        self.gradients = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _find_target_layer(self, layer_name):
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")

    def _forward_hook(self, module, input, output):
        self.feature_map = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_img, target_index=None):
        outputs = self.model(input_img)
        final_output = outputs[-1]  # shape: (1,1,256,256)

        self.model.zero_grad()
        if target_index is None:
            loss = final_output.mean()
        else:
            loss = final_output[..., target_index].mean()
        loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature_map, dim=1).squeeze()  # shape: (256,256)
        cam = F.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # 确保维度正确：cam是(256,256)，pred_mask是(1,1,256,256)
        cam_np = cam.detach().cpu().numpy()
        pred_mask_np = final_output.detach().sigmoid().cpu().numpy()

        # 打印维度（调试用，可删除）
        print(f"cam shape: {cam_np.shape}, pred_mask shape: {pred_mask_np.shape}")
        return cam_np, pred_mask_np


# -------------------------- 图像预处理 --------------------------
def preprocess_image(img_path, img_size=256):
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"无法读取图像：{img_path}")
    # 先检查原始图像通道数
    if len(original_img.shape) != 3 or original_img.shape[2] != 3:
        print(f"⚠️ 图像 {img_path} 不是3通道RGB图，自动转换")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    h, w = original_img.shape[:2]

    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 补充默认归一化参数
        ToTensorV2()
    ])

    augmented = transform(image=original_img)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    return img_tensor, original_img, (h, w)


# -------------------------- 边界提取辅助函数 --------------------------
def extract_boundary(mask, low_threshold=50, high_threshold=150):
    """
    从预测掩码中提取边界（Canny边缘检测）
    Args:
        mask: 单通道掩码数组 (H,W)，值范围0-1或0-255
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值
    Returns:
        boundary: 二值边界掩码 (H,W)，边界处为255，其余为0
    """
    # 归一化到0-255并转为uint8
    if mask.max() <= 1.0:
        mask = (mask * 255).astype(np.uint8)
    # Canny边缘检测
    boundary = cv2.Canny(mask, low_threshold, high_threshold)
    # 膨胀边界（可选，让边界更明显）
    kernel = np.ones((2, 2), np.uint8)
    boundary = cv2.dilate(boundary, kernel, iterations=1)
    return boundary


# -------------------------- 边界热力图可视化（核心修改） --------------------------
def visualize_cam(cam, original_img, pred_mask, img_size, save_path):
    """
    核心修改：
    1. 提取预测掩码的边界
    2. 生成仅显示边界区域的热力图
    3. 新增边界相关子图展示
    """
    # 1. 基础处理（确保是2D数组）
    if len(cam.shape) != 2:
        cam = cam.squeeze()
    cam_resized = cv2.resize(cam, (img_size[1], img_size[0]))  # 匹配原始图像尺寸
    pred_mask = pred_mask.squeeze()
    pred_mask_resized = cv2.resize(pred_mask, (img_size[1], img_size[0]))

    # 2. 提取预测掩码的边界
    pred_mask_binary = (pred_mask_resized > 0.5).astype(np.uint8) * 255
    boundary = extract_boundary(pred_mask_binary)  # 边界掩码 (H,W)

    # 3. 生成边界热力图（仅保留边界区域的热力值）
    cam_boundary = cam_resized * (boundary / 255)  # 边界处保留热力值，其余为0
    # 归一化边界热力图（避免全黑）
    if cam_boundary.max() > 0:
        cam_boundary = (cam_boundary - cam_boundary.min()) / (cam_boundary.max() - cam_boundary.min() + 1e-8)

    # 4. 处理普通热力图（原有逻辑）
    cam_colored = (cam_resized * 255).astype(np.uint8)
    cam_colored = cv2.applyColorMap(cam_colored, cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)

    # 5. 处理边界热力图（上色）
    cam_boundary_colored = (cam_boundary * 255).astype(np.uint8)
    cam_boundary_colored = cv2.applyColorMap(cam_boundary_colored, cv2.COLORMAP_JET)
    cam_boundary_colored = cv2.cvtColor(cam_boundary_colored, cv2.COLOR_BGR2RGB)

    # 6. 处理预测掩码和边界（可视化用）
    pred_mask_vis = cv2.cvtColor(pred_mask_binary, cv2.COLOR_GRAY2RGB)
    boundary_vis = cv2.cvtColor(boundary, cv2.COLOR_GRAY2RGB)

    # 7. 融合图像（原图像+边界热力图）
    fusion_boundary_img = 0.6 * original_img + 0.4 * cam_boundary_colored
    fusion_boundary_img = fusion_boundary_img.astype(np.uint8)

    # 8. 绘制子图（1行5列：原图、普通热力图、预测掩码、边界、边界热力图）
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(cam_colored)
    axes[1].set_title('Normal Grad-CAM', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(pred_mask_vis)
    axes[2].set_title('Prediction Mask', fontsize=12)
    axes[2].axis('off')

    axes[3].imshow(boundary_vis)
    axes[3].set_title('Mask Boundary', fontsize=12)
    axes[3].axis('off')

    axes[4].imshow(fusion_boundary_img)
    axes[4].set_title('Boundary Grad-CAM', fontsize=12)
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


# -------------------------- 主函数 --------------------------
def main():
    # 你的参数
    model_path = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/result/消融/sobel边界/Network_best_20260106_1157.pth"
    img_dir = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/pre/image1"
    save_dir = "./999heatmap_boundary_results"
    target_layer = "rfb4_1"
    img_size = 256

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"边界热力图将保存到：{os.path.abspath(save_dir)}")

    # 加载模型（忽略不匹配参数）
    print("正在加载模型...")
    model = Network(imagenet_pretrained=False).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model_state_dict = model.state_dict()
        matched_params = {}

        for k, v in checkpoint.items():
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                matched_params[k] = v
            else:
                print(f"⚠️ 跳过不匹配参数：{k} | 权重形状：{v.shape} | 模型形状：{model_state_dict.get(k, '不存在')}")

        model_state_dict.update(matched_params)
        model.load_state_dict(model_state_dict, strict=False)
        print("✅ 模型加载成功（已忽略不匹配参数）！")
    except Exception as e:
        print(f"⚠️ 模型加载警告：{e}，将使用随机初始化模型测试")

    model.eval()

    # 初始化Grad-CAM
    print(f"可视化目标层：{target_layer}")
    grad_cam = GradCAM(model, target_layer)

    # 处理图像
    img_list = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_list) == 0:
        raise ValueError(f"在目录 {img_dir} 中未找到图像文件")

    print(f"找到 {len(img_list)} 张图像，开始生成边界热力图...")
    for idx, img_name in enumerate(img_list):
        try:
            img_path = os.path.join(img_dir, img_name)
            input_tensor, original_img, ori_size = preprocess_image(img_path, img_size)

            # 生成热力图
            cam, pred_mask = grad_cam.generate_cam(input_tensor)

            # 可视化并保存（边界热力图）
            save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_boundary_heatmap.png")
            visualize_cam(cam, original_img, pred_mask, ori_size, save_path)

            print(f"[{idx + 1}/{len(img_list)}] 已生成：{save_path}")
        except Exception as e:
            print(f"❌ 处理 {img_name} 失败：{str(e)[:200]}")
            continue

    print("\n✅ 边界热力图生成完成！")
    print(f"所有热力图已保存到：{os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()