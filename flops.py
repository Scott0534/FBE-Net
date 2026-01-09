import torch
import os
import logging
from thop import profile
import numpy as np
from lib.new import Network
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 关键新增行
# 初始化模型
# config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
# config_vit.n_skip = 3
# config_vit.patches.grid = (int(256 / 16), int(256 / 16))
# model = TransUnet(config_vit, img_size=256, num_classes=1)
# model.load_from(weights=np.load(config_vit.pretrained_path))
model = Network()

# 创建模拟输入（batch_size=1, 通道=3, 高度=256, 宽度=256）
input_tensor = torch.randn(1, 3, 256, 256)

# 计算FLOPs和参数量
flops, params = profile(model, inputs=(input_tensor,))

# 输出结果
print(f"模型参数量: {params / 1e6:.2f} M")  # 转换为百万单位
print(f"计算量: {flops / 1e9:.2f} GFLOPs")  # 转换为十亿单位