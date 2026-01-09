# import torch
# import torch.nn.functional as F
# import numpy as np
# import cv2
# import os
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from lib.new import Network  # 导入你的网络
# import matplotlib.pyplot as plt
#
# # 设备配置
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# # -------------------------- Grad-CAM 核心类 --------------------------
# class GradCAM:
#     def __init__(self, model, target_layer_name):
#         self.model = model
#         self.model.eval()
#         self.target_layer = self._find_target_layer(target_layer_name)
#
#         self.feature_map = None
#         self.gradients = None
#
#         self.target_layer.register_forward_hook(self._forward_hook)
#         self.target_layer.register_full_backward_hook(self._backward_hook)
#
#     def _find_target_layer(self, layer_name):
#         for name, module in self.model.named_modules():
#             if name == layer_name:
#                 return module
#         raise ValueError(f"Layer {layer_name} not found in model")
#
#     def _forward_hook(self, module, input, output):
#         self.feature_map = output.detach()
#
#     def _backward_hook(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()
#
#     def generate_cam(self, input_img, target_index=None):
#         outputs = self.model(input_img)
#         final_output = outputs[-1]  # shape: (1,1,256,256)
#
#         self.model.zero_grad()
#         if target_index is None:
#             loss = final_output.mean()
#         else:
#             loss = final_output[..., target_index].mean()
#         loss.backward()
#
#         weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
#         cam = torch.sum(weights * self.feature_map, dim=1).squeeze()  # shape: (256,256)
#         cam = F.relu(cam)
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#
#         # 确保维度正确：cam是(256,256)，pred_mask是(1,1,256,256)
#         cam_np = cam.detach().cpu().numpy()
#         pred_mask_np = final_output.detach().sigmoid().cpu().numpy()
#
#         # 打印维度（调试用，可删除）
#         print(f"cam shape: {cam_np.shape}, pred_mask shape: {pred_mask_np.shape}")
#         return cam_np, pred_mask_np
#
#
# # -------------------------- 图像预处理 --------------------------
# def preprocess_image(img_path, img_size=256):
#     original_img = cv2.imread(img_path)
#     if original_img is None:
#         raise ValueError(f"无法读取图像：{img_path}")
#     # 先检查原始图像通道数
#     if len(original_img.shape) != 3 or original_img.shape[2] != 3:
#         print(f"⚠️ 图像 {img_path} 不是3通道RGB图，自动转换")
#         original_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
#     original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#     h, w = original_img.shape[:2]
#
#     transform = A.Compose([
#         A.Resize(img_size, img_size),
#         A.Normalize(),
#         ToTensorV2()
#     ])
#
#     augmented = transform(image=original_img)
#     img_tensor = augmented['image'].unsqueeze(0).to(device)
#
#     return img_tensor, original_img, (h, w)
#
#
# # -------------------------- 热力图可视化（核心修复） --------------------------
# def visualize_cam(cam, original_img, pred_mask, img_size, save_path):
#     """
#     修复点：
#     1. 强制压缩pred_mask的维度到单通道(H,W)
#     2. 增加维度检查，避免通道数错误
#     3. 确保cv2转换前是单通道灰度图
#     """
#     # 1. 处理热力图（确保是2D数组）
#     if len(cam.shape) != 2:
#         cam = cam.squeeze()  # 压缩所有多余维度
#     cam = cv2.resize(cam, (img_size[1], img_size[0]))
#     cam = (cam * 255).astype(np.uint8)
#     cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
#     cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
#
#     # 2. 处理预测掩码（核心修复：强制转为单通道灰度图）
#     # 压缩所有多余维度 → 从(1,1,H,W) → (H,W)
#     pred_mask = pred_mask.squeeze()  # 关键：去掉所有维度为1的轴
#     # 调整尺寸到原始图像大小
#     pred_mask = cv2.resize(pred_mask, (img_size[1], img_size[0]))
#     # 二值化并转为uint8
#     pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
#     # 确保是单通道后，再转RGB
#     if len(pred_mask.shape) == 2:  # 单通道灰度图
#         pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
#     else:  # 异常情况：强制转单通道
#         pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
#         pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
#
#     # 3. 融合图像
#     fusion_img = 0.5 * original_img + 0.5 * cam
#     fusion_img = fusion_img.astype(np.uint8)  # 避免浮点值导致显示异常
#
#     # 4. 绘制子图
#     fig, axes = plt.subplots(1, 4, figsize=(20, 5))
#     axes[0].imshow(original_img)
#     axes[0].set_title('Original Image')
#     axes[0].axis('off')
#
#     axes[1].imshow(cam)
#     axes[1].set_title('Grad-CAM Heatmap')
#     axes[1].axis('off')
#
#     axes[2].imshow(fusion_img)
#     axes[2].set_title('Heatmap + Original')
#     axes[2].axis('off')
#
#     axes[3].imshow(pred_mask)
#     axes[3].set_title('Prediction Mask')
#     axes[3].axis('off')
#
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches='tight', dpi=300)
#     plt.close()
#
#
# # -------------------------- 主函数 --------------------------
# def main():
#     # 你的参数
#     model_path = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/result/1.8/用来绘图的权重se4/Network_epoch_20_20260105_2018.pth"
#     img_dir = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/pre/image1"
#     save_dir = "./preheatmap_results"
#     target_layer = "rfb4_1"
#     img_size = 256
#
#     # 创建保存目录
#     os.makedirs(save_dir, exist_ok=True)
#     print(f"热力图将保存到：{os.path.abspath(save_dir)}")
#
#     # 加载模型（忽略不匹配参数）
#     print("正在加载模型...")
#     model = Network(imagenet_pretrained=False).to(device)
#     try:
#         checkpoint = torch.load(model_path, map_location=device)
#         model_state_dict = model.state_dict()
#         matched_params = {}
#
#         for k, v in checkpoint.items():
#             if k in model_state_dict and model_state_dict[k].shape == v.shape:
#                 matched_params[k] = v
#             else:
#                 print(f"⚠️ 跳过不匹配参数：{k} | 权重形状：{v.shape} | 模型形状：{model_state_dict.get(k, '不存在')}")
#
#         model_state_dict.update(matched_params)
#         model.load_state_dict(model_state_dict, strict=False)
#         print("✅ 模型加载成功（已忽略不匹配参数）！")
#     except Exception as e:
#         raise ValueError(f"加载模型失败：{e}")
#     model.eval()
#
#     # 初始化Grad-CAM
#     print(f"可视化目标层：{target_layer}")
#     grad_cam = GradCAM(model, target_layer)
#
#     # 处理图像
#     img_list = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
#     if len(img_list) == 0:
#         raise ValueError(f"在目录 {img_dir} 中未找到图像文件")
#
#     print(f"找到 {len(img_list)} 张图像，开始生成热力图...")
#     for idx, img_name in enumerate(img_list):
#         try:
#             img_path = os.path.join(img_dir, img_name)
#             input_tensor, original_img, ori_size = preprocess_image(img_path, img_size)
#
#             # 生成热力图
#             cam, pred_mask = grad_cam.generate_cam(input_tensor)
#
#             # 可视化并保存
#             save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_heatmap.png")
#             visualize_cam(cam, original_img, pred_mask, ori_size, save_path)
#
#             print(f"[{idx + 1}/{len(img_list)}] 已生成：{save_path}")
#         except Exception as e:
#             print(f"❌ 处理 {img_name} 失败：{str(e)[:200]}")  # 只打印前200字符，避免过长
#             continue
#
#     print("\n✅ 热力图生成完成！")
#     print(f"所有热力图已保存到：{os.path.abspath(save_dir)}")
#
#
# if __name__ == "__main__":
#     main()
#


import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from lib.new6 import Network  # 导入你的真实网络
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
        A.Normalize(),
        ToTensorV2()
    ])

    augmented = transform(image=original_img)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    return img_tensor, original_img, (h, w)


# -------------------------- 热力图可视化 --------------------------
def visualize_cam(cam, original_img, pred_mask, img_size, save_path):
    # 1. 处理热力图（确保是2D数组）
    if len(cam.shape) != 2:
        cam = cam.squeeze()  # 压缩所有多余维度
    cam = cv2.resize(cam, (img_size[1], img_size[0]))
    cam = (cam * 255).astype(np.uint8)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

    # 2. 处理预测掩码
    pred_mask = pred_mask.squeeze()  # 关键：去掉所有维度为1的轴
    pred_mask = cv2.resize(pred_mask, (img_size[1], img_size[0]))
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    # 确保是单通道后，再转RGB
    if len(pred_mask.shape) == 2:  # 单通道灰度图
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)
    else:  # 异常情况：强制转单通道
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2RGB)

    # 3. 融合图像
    fusion_img = 0.5 * original_img + 0.5 * cam
    fusion_img = fusion_img.astype(np.uint8)  # 避免浮点值导致显示异常

    # 4. 绘制子图
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cam)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    axes[2].imshow(fusion_img)
    axes[2].set_title('Heatmap + Original')
    axes[2].axis('off')

    axes[3].imshow(pred_mask)
    axes[3].set_title('Prediction Mask')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


# -------------------------- 主函数（核心修复：正确解析checkpoint） --------------------------
def main():
    # 你的参数
    model_path = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/result/消融/拼接/Network_best_20260107_1519.pth"
    img_dir = "/home/xfusion/Wangchen/ZYB/SINet-V2-main/pre/image1"
    save_dir = "./相加preheatmap_results"
    target_layer = "rfb4_1"
    img_size = 256

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"热力图将保存到：{os.path.abspath(save_dir)}")

    # 加载模型（核心修复：正确解析checkpoint层级）
    print("正在加载模型...")
    model = Network(imagenet_pretrained=False).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # ========== 核心修复：优先提取model_state_dict ==========
        # 第一步：先判断checkpoint是否包含model_state_dict（这是真正的模型权重）
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']  # 提取真正的权重字典
            print(f"✅ 提取到model_state_dict，共 {len(model_weights)} 个模型参数")
        else:
            model_weights = checkpoint  # 没有包裹层，直接用checkpoint

        # 第二步：加载模型权重（只过滤非张量，不再跳过model_state_dict）
        model_state_dict = model.state_dict()
        matched_params = {}
        for k, v in model_weights.items():
            # 只过滤真正的非张量（如数字、字符串），不再误过滤model_state_dict
            if not isinstance(v, torch.Tensor):
                print(f"⚠️ 跳过非张量参数：{k} (类型：{type(v)})")
                continue

            # 安全获取模型参数形状
            model_shape = model_state_dict[k].shape if k in model_state_dict else "不存在"

            # 检查参数是否匹配
            if k in model_state_dict and model_state_dict[k].shape == v.shape:
                matched_params[k] = v
            else:
                print(f"⚠️ 跳过不匹配参数：{k} | 权重形状：{v.shape} | 模型形状：{model_shape}")

        # 更新并加载模型参数
        model_state_dict.update(matched_params)
        model.load_state_dict(model_state_dict, strict=False)
        print(f"✅ 模型加载成功！共匹配 {len(matched_params)} 个参数")

    except Exception as e:
        raise ValueError(f"加载模型失败：{e}")
    model.eval()

    # 初始化Grad-CAM
    print(f"可视化目标层：{target_layer}")
    grad_cam = GradCAM(model, target_layer)

    # 处理图像
    img_list = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if len(img_list) == 0:
        raise ValueError(f"在目录 {img_dir} 中未找到图像文件")

    print(f"找到 {len(img_list)} 张图像，开始生成热力图...")
    for idx, img_name in enumerate(img_list):
        try:
            img_path = os.path.join(img_dir, img_name)
            input_tensor, original_img, ori_size = preprocess_image(img_path, img_size)

            # 生成热力图
            cam, pred_mask = grad_cam.generate_cam(input_tensor)

            # 可视化并保存
            save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_heatmap.png")
            visualize_cam(cam, original_img, pred_mask, ori_size, save_path)

            print(f"[{idx + 1}/{len(img_list)}] 已生成：{save_path}")
        except Exception as e:
            print(f"❌ 处理 {img_name} 失败：{str(e)[:200]}")
            continue

    print("\n✅ 热力图生成完成！")
    print(f"所有热力图已保存到：{os.path.abspath(save_dir)}")


if __name__ == "__main__":
    main()