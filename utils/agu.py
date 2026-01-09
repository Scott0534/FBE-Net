import albumentations as A
import math
import cv2  # 显式导入cv2，参数用cv2常量更易读


def transform_to_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num


def medical_augment(level=5):
    # 像素级变换（亮度、对比度、模糊等）
    pixel_transforms = [
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, transform_to_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level)
    ]

    # 空间级变换（旋转、翻转、仿射等）
    spatial_transforms = [
        # 核心修正：移除 rotate_method 和 crop_border 参数（新版不支持）
        A.Rotate(
            limit=4 * level,
            interpolation=cv2.INTER_LINEAR,  # 1 对应 cv2.INTER_LINEAR
            border_mode=cv2.BORDER_CONSTANT,  # 0 对应 cv2.BORDER_CONSTANT
            value=0,
            mask_value=None,
            p=0.2 * level  # 移除了 crop_border 参数
        ),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        # A.ElasticTransform(
        #     alpha=(10 * level, 30 * level),  # 变形强度，随level增大（level5时50-150）
        #     sigma=(4, 6),  # 高斯核标准差（控制变形平滑度，固定值避免过度扭曲）
        #     alpha_affine=(2 * level, 5 * level),  # 仿射变换强度，随level增大
        #     interpolation=cv2.INTER_LINEAR,
        #     border_mode=cv2.BORDER_CONSTANT,
        #     value=0,  # 边界填充值（适配超声图像黑色背景）
        #     mask_value=None,
        #     approximate=False,  # 关闭近似计算，保证变形精度
        #     p=0.2 * level  # 概率与其他空间变换一致
        # ),
        A.Affine(
            scale=(1 - 0.04 * level, 1 + 0.04 * level),
            translate_percent=None,
            translate_px=None,
            rotate=None,
            shear=None,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,  # 0 对应 cv2.INTER_NEAREST
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            keep_ratio=True,
            p=0.2 * level
        ),
        A.Affine(
            scale=None,
            translate_percent=None,
            translate_px=None,
            rotate=None,
            shear={'x': (0, 2 * level), 'y': (0, 0)},
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            keep_ratio=True,
            p=0.2 * level
        ),
        A.Affine(
            scale=None,
            translate_percent=None,
            translate_px=None,
            rotate=None,
            shear={'x': (0, 0), 'y': (0, 2 * level)},
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            keep_ratio=True,
            p=0.2 * level
        ),
        A.Affine(
            scale=None,
            translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)},
            translate_px=None,
            rotate=None,
            shear=None,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            keep_ratio=True,
            p=0.2 * level
        ),
        A.Affine(
            scale=None,
            translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)},
            translate_px=None,
            rotate=None,
            shear=None,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            cval=0,
            cval_mask=0,
            mode=cv2.BORDER_CONSTANT,
            fit_output=False,
            keep_ratio=True,
            p=0.2 * level
        )
    ]

    # 变换组合：1个像素变换 + 2个空间变换（不同顺序）
    transforms_1_2 = [
        A.Compose([A.OneOf(pixel_transforms, p=1), A.OneOf(spatial_transforms, p=1), A.OneOf(spatial_transforms, p=1)], p=1/3),
        A.Compose([A.OneOf(spatial_transforms, p=1), A.OneOf(pixel_transforms, p=1), A.OneOf(spatial_transforms, p=1)], p=1/3),
        A.Compose([A.OneOf(spatial_transforms, p=1), A.OneOf(spatial_transforms, p=1), A.OneOf(pixel_transforms, p=1)], p=1/3)
    ]

    # 变换组合：3个空间变换
    transforms_0_3 = [A.Compose([A.OneOf(spatial_transforms, p=1)]*3, p=1)]

    # 变换组合：2个空间变换
    transforms_0_2 = [A.Compose([A.OneOf(spatial_transforms, p=1)]*2, p=1)]

    # 变换组合：1个像素变换 + 1个空间变换（不同顺序）
    transforms_1_1 = [
        A.Compose([A.OneOf(pixel_transforms, p=1), A.OneOf(spatial_transforms, p=1)], p=1/2),
        A.Compose([A.OneOf(spatial_transforms, p=1), A.OneOf(pixel_transforms, p=1)], p=1/2)
    ]

    # 最终组合所有变换策略
    MedAugment = A.OneOf([
        A.OneOf(transforms_1_2, p=1),
        A.OneOf(transforms_0_3, p=1),
        A.OneOf(transforms_0_2, p=1),
        A.OneOf(transforms_1_1, p=1)
    ], p=1)

    return MedAugment