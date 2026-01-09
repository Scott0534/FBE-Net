# import os
# import cv2
# from torch.utils.data import Dataset
# import numpy as np
# import torch
#
#
#
# class MedicalDataSets(Dataset):
#     def __init__(
#             self,
#             base_dir=None,
#             split="train",
#             transform=None,
#             train_file_dir="train.txt",
#             val_file_dir="val.txt",
#             edge_kernel_size=3
#     ):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         self.transform = transform
#         self.edge_kernel_size = edge_kernel_size
#         self.kernel = cv2.getStructuringElement(
#             cv2.MORPH_RECT,
#             (self.edge_kernel_size, self.edge_kernel_size)
#         )
#
#         if self.split == "train":
#             with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
#                 self.sample_list = f1.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#         elif self.split == "val":
#             with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
#                 self.sample_list = f.readlines()
#             self.sample_list = [item.replace("\n", "") for item in self.sample_list]
#
#         print(f"total {len(self.sample_list)} {self.split} samples")
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     # def _compute_distance_map(self, label_2d: np.ndarray) -> np.ndarray:
#     #     """生成距离变换图"""
#     #     foreground = (label_2d == 1).astype(bool)  # 修正：np.bool_ -> bool
#     #     background = ~foreground
#     #
#     #     if np.any(foreground):
#     #         dist_foreground = -distance_transform_edt(foreground)
#     #         dist_background = distance_transform_edt(background)
#     #         dist_map = dist_foreground * foreground + dist_background * background
#     #     else:
#     #         dist_map = distance_transform_edt(background)
#     #
#     #     # 归一化
#     #     max_abs = np.max(np.abs(dist_map))
#     #     if max_abs > 0:
#     #         dist_map = dist_map / max_abs
#     #
#     #     return dist_map.astype(np.float32)
#
#     def _compute_edge_mask(self, label_2d: np.ndarray) -> np.ndarray:
#         """生成边缘掩码"""
#         label_uint8 = (label_2d * 255).astype(np.uint8)
#         dilated = cv2.dilate(label_uint8, self.kernel, iterations=1)
#         eroded = cv2.erode(label_uint8, self.kernel, iterations=1)
#         edge = dilated - eroded
#         edge = (edge > 0).astype(np.float32)
#         return edge
#
#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#
#         # 1. 读取原始数据
#         image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png'))
#         if image is None:
#             raise FileNotFoundError(f"图像文件不存在: {os.path.join(self._base_dir, 'images', case + '.png')}")
#
#         label_path = os.path.join(self._base_dir, 'masks', '0', case + '_mask.png')
#         label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
#         if label is None:
#             raise FileNotFoundError(f"掩码文件不存在: {label_path}")
#
#         # 2. 预处理：将mask二值化
#         label_2d = label.squeeze()
#         if label_2d.max() > 1:
#             label_2d = (label_2d / 255.0).astype(np.float32)
#         label_2d = (label_2d > 0.5).astype(np.float32)
#
#         # 3. 计算边缘掩码
#         edge_mask = self._compute_edge_mask(label_2d)
#
#         # 4. 数据增强（同时对image, mask, edge_mask）
#         if self.transform is not None:
#             # 确保维度正确 (H, W, 1)
#             label_3d = label_2d[..., None] if len(label_2d.shape) == 2 else label_2d
#             edge_mask_3d = edge_mask[..., None] if len(edge_mask.shape) == 2 else edge_mask
#
#             augmented = self.transform(
#                 image=image,
#                 mask=label_3d.astype(np.float32),
#                 edge_mask=edge_mask_3d.astype(np.float32)
#             )
#
#             image = augmented['image']
#             label_2d = augmented['mask'].squeeze()
#             edge_mask = augmented['edge_mask'].squeeze()
#         else:
#             # 无增强时的处理
#             image = image.astype(np.float32) / 255.0
#
#         # 5. 重新二值化（增强后可能有插值）
#         label_2d = (label_2d > 0.5).astype(np.float32)
#         edge_mask = (edge_mask > 0.5).astype(np.float32)
#
#         # 6. 计算距离图（在增强后计算，确保对齐）
#         # dist_map = self._compute_distance_map(label_2d)
#
#         # 7. 维度转换
#         image = image.transpose(2, 0, 1)  # (C, H, W)
#         label_output = label_2d[None, ...]  # (1, H, W)
#         edge_mask = edge_mask[None, ...]  # (1, H, W)
#         # dist_map = dist_map[None, ...]  # (1, H, W)
#
#         # 8. 返回样本
#         sample = {
#             "image": torch.from_numpy(image).float(),
#             "label": torch.from_numpy(label_output).float(),
#             "edge_mask": torch.from_numpy(edge_mask).float(),
#             # "dist_map": torch.from_numpy(dist_map).float(),  # 可选
#             "case": case
#         }
#
#         return sample


import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional


class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir: str,
            split: str = "train",
            transform=None,
            train_file_dir: str = "train.txt",
            val_file_dir: str = "val.txt",
            edge_kernel_size: int = 3,  # 边缘宽度控制5.7
            img_extension: str = '.png',
            mask_extension: str = '.png'
    ):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform
        self.img_extension = img_extension
        self.mask_extension = mask_extension

        # 预先定义形态学核，避免在 __getitem__ 中重复创建
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (edge_kernel_size, edge_kernel_size)
        )

        # 加载文件列表
        list_file = train_file_dir if split == "train" else val_file_dir
        list_path = os.path.join(base_dir, list_file)

        with open(list_path, "r") as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        print(f"[{split.upper()}] Loaded {len(self.sample_list)} samples from {list_path}")

    def __len__(self):
        return len(self.sample_list)

    def _generate_edge_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        基于二值 Mask 生成边缘带
        输入: (H, W) 0/1 float32
        输出: (H, W) 0/1 float32
        """
        # 转换为 uint8 进行形态学操作
        mask_u8 = (mask * 255).astype(np.uint8)

        # 膨胀 - 腐蚀 = 边缘
        dilated = cv2.dilate(mask_u8, self.kernel, iterations=1)
        eroded = cv2.erode(mask_u8, self.kernel, iterations=1)
        edge = dilated - eroded

        # 归一化回 0-1
        return (edge > 127).astype(np.float32)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]

        # 1. 路径构建 (根据你的目录结构调整)
        # 假设结构: images/name.png, masks/0/name_mask.png
        img_path = os.path.join(self.base_dir, 'images', case_name + self.img_extension)
        mask_path = os.path.join(self.base_dir, 'masks', '0', case_name + '_mask' + self.mask_extension)
        # mask_path = os.path.join(self.base_dir, 'masks', '0', case_name  + self.mask_extension)
        # 2. 读取图像
        image = cv2.imread(img_path)
        if image is None:
            # 容错处理：打印错误路径
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转为RGB，Albumentations默认RGB

        # 3. 读取掩码
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # 4. 数据增强
        # 关键修改：此时只增强 Image 和 Mask，不增强 Edge
        # 这样可以确保 Edge 是基于最终变形后的 Mask 生成的，宽度恒定
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # 无增强时的预处理
            image = image.astype(np.float32) / 255.0
            image = (image - 0.5) / 0.5  # 简单的标准化示例，如果 transform 中有 Normalize 则不需要
            image = image.transpose(2, 0, 1)  # HWC -> CHW

            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=0)  # HW -> 1HW

        # 5. 后处理 Mask (确保是 Tensor 且二值化)
        if isinstance(mask, torch.Tensor):
            mask = mask.numpy()

        # 如果 transform 只有 Resize 没有 ToTensor/Normalize，mask 可能是 (H, W)
        if mask.ndim == 3 and mask.shape[0] == 1:  # (1, H, W)
            mask_2d = mask.squeeze(0)
        elif mask.ndim == 3 and mask.shape[2] == 1:  # (H, W, 1)
            mask_2d = mask.squeeze(2)
        else:
            mask_2d = mask

        # 确保二值化 (增强过程中的插值可能产生小数)
        mask_2d = (mask_2d > 0.5).astype(np.float32)

        # 6. 生成边缘 Mask (在增强之后生成！)
        # 优势：无论图片怎么缩放、旋转，边缘监督信号永远是 kernel_size 宽度的连贯线条
        edge_mask_2d = self._generate_edge_mask(mask_2d)

        # 7. 转回 Tensor
        # image 已经在 albumentations 中处理为 Tensor (如果用了 ToTensorV2)
        # 如果没有用 ToTensorV2，这里需要手动转换
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            if image.shape[0] != 3:  # 确保是 CHW
                image = image.permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask_2d).unsqueeze(0).float()  # (1, H, W)
        edge_mask_tensor = torch.from_numpy(edge_mask_2d).unsqueeze(0).float()  # (1, H, W)

        return {
            "image": image,
            "label": mask_tensor,
            "edge_mask": edge_mask_tensor,
            "case": case_name
        }