import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import ViTModel, ViTConfig
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2lab
import os
import json
from PIL import Image
from torchvision import transforms

def process_images_to_tensor(image_folder, json_path, output_path):
    # 1. 加载 JSON 数据
    with open(json_path, 'r', encoding='utf-8') as f:
        label_data = json.load(f)

    # 2. 定义图像预处理 (缩放到 384x384 并转为 Tensor)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(), # 自动归一化到 [0, 1]
    ])

    image_tensors = []
    values = []

    # 3. 遍历文件夹中的图片
    # 假设图片后缀为 .jpg, .png 等
    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(supported_exts)]
    
    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for filename in image_files:
        # 获取不带后缀的文件名作为 key
        name_key = os.path.splitext(filename)[0]
        
        if name_key in label_data:
            try:
                # 处理图片
                img_path = os.path.join(image_folder, filename)
                img = Image.open(img_path).convert('RGB') # 确保是3通道
                img_tensor = transform(img)
                
                image_tensors.append(img_tensor)
                
                # 获取对应的 -1 到 1 的数值
                val = float(label_data[name_key])
                values.append(val)
                
            except Exception as e:
                print(f"处理图片 {filename} 时出错: {e}")
        else:
            print(f"跳过: {filename} 在 JSON 中找不到索引")

    if not image_tensors:
        print("没有处理任何数据。")
        return

    # 4. 合成最终张量
    # images_batch 形状: [N, 3, 384, 384]
    images_batch = torch.stack(image_tensors)
    # values_tensor 形状: [N]
    values_tensor = torch.tensor(values, dtype=torch.float32)

    # 将图片和数值封装在一个字典里保存，或者根据需求分开
    result = {
        'images': images_batch,
        'labels': values_tensor
    }

    # 5. 保存文件
    torch.save(result, output_path)
    print(f"处理完成！文件已保存至: {output_path}")
    print(f"张量形状: {images_batch.shape}")

process_images_to_tensor('ICQA_2K_train', 'ICQA_2K_train_CF_mos.json', 'ICQA_2K_train_CF.pt')
process_images_to_tensor('ICQA_2K_test', 'ICQA_2K_test_CF_mos.json', 'ICQA_2K_test_CF.pt')

# ==========================================
# 1. 专项改造的模型结构
# ==========================================
class ViTColorQualityRegressor(nn.Module):
    def __init__(self, local_path):
        super().__init__()
        # 基础 ViT 主干
        self.vit = ViTModel.from_pretrained(local_path)
        hidden_size = self.vit.config.hidden_size
        
        # 颜色斑驳特征提取分支 (针对 Lab 的 ab 通道)
        self.chroma_head = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((12, 12))  # 缩小尺寸与特征图匹配
        )
        
        # 最终回归器
        # 输入包括：[CLS] token + Patch 标准差 + 色度统计特征
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Tanh() # 假设 MOS 已归一化至 [-1, 1]
        )

    def forward(self, x_rgb, x_lab):
        # x_rgb: (B, 3, 384, 384)
        # x_lab: (B, 3, 384, 384)
        
        # 1. ViT 提取全局和局部语义
        vit_outputs = self.vit(pixel_values=x_rgb)
        last_hidden_state = vit_outputs.last_hidden_state  # (B, 577, 768)
        
        cls_token = last_hidden_state[:, 0, :]  # 全局语义
        patch_tokens = last_hidden_state[:, 1:, :]  # 局部 Patch 特征
        
        # 2. 计算 Patch 间的波动 (针对斑驳感)
        # 斑驳严重的图像，Patch 间的特征方差通常会异常
        patch_std = torch.std(patch_tokens, dim=1) 
        
        # 3. 提取专门的色度污渍特征
        chroma_info = x_lab[:, 1:, :, :] # 提取 a, b 通道
        chroma_feat = self.chroma_head(chroma_info)
        chroma_feat = torch.flatten(chroma_feat, 1)
        chroma_feat = nn.functional.adaptive_avg_pool1d(chroma_feat.unsqueeze(1), 32).squeeze(1)

        # 4. 特征拼接并回归
        combined = torch.cat([cls_token, patch_std, chroma_feat], dim=-1)
        return self.regressor(combined)

# ==========================================
# 2. 数据处理辅助函数 (增加 Lab 空间转换)
# ==========================================
def process_data_to_lab(imgs_tensor):
    """ 将 RGB Tensor 转换为 Lab Tensor """
    imgs_np = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()
    lab_list = []
    for i in range(imgs_np.shape[0]):
        # skimage 要求输入范围通常是 [0, 1]
        lab = rgb2lab(np.clip(imgs_np[i], 0, 1))
        # 归一化 Lab 到适合神经网络的范围 (简单处理)
        lab[:,:,0] /= 100.0
        lab[:,:,1:] /= 128.0
        lab_list.append(torch.from_numpy(lab).permute(2, 0, 1))
    return torch.stack(lab_list).float()

def get_dataloader(file_path, batch_size=16, shuffle=True):
    data = torch.load(file_path)
    rgb_images = data['images']
    labels = data['labels'].float()
    
    # 预处理生成 Lab 图像
    print(f"正在转换 {file_path} 颜色空间...")
    lab_images = process_data_to_lab(rgb_images)
    
    dataset = TensorDataset(rgb_images, lab_images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ==========================================
# 3. 训练主程序
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = ViTColorQualityRegressor('./pretrain/').to(device)

# 分层设置学习率: Backbone 慢一点，新 head 快一点
optimizer = optim.AdamW([
    {'params': model.vit.parameters(), 'lr': 1e-6},
    {'params': model.chroma_head.parameters(), 'lr': 1e-4},
    {'params': model.regressor.parameters(), 'lr': 1e-4}
], weight_decay=1e-2)

criterion = nn.MSELoss()

# 加载数据
train_loader = get_dataloader('ICQA_2K_train_CF.pt', batch_size=12)
test_loader = get_dataloader('ICQA_2K_test_CF.pt', batch_size=12, shuffle=False)

epochs = 400
train_losses, test_losses = [], []

print("开始专项训练...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for b_rgb, b_lab, b_labels in train_loader:
        b_rgb, b_lab, b_labels = b_rgb.to(device), b_lab.to(device), b_labels.to(device).unsqueeze(1)

        preds = model(b_rgb, b_lab)
        loss = criterion(preds, b_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    
    # 验证
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for b_rgb, b_lab, b_labels in test_loader:
            b_rgb, b_lab, b_labels = b_rgb.to(device), b_lab.to(device), b_labels.to(device).unsqueeze(1)
            preds = model(b_rgb, b_lab)
            running_test_loss += criterion(preds, b_labels).item()
    
    avg_test_loss = running_test_loss / len(test_loader)

    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)

    print(f"Epoch [{epoch+1}/{epochs}] Train: {avg_train_loss:.4f} Test: {avg_test_loss:.4f}")

    if (avg_train_loss <= 0.001):
        break

# ==========================================
# 4. 绘图与保存
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.legend()
plt.savefig('CF_training_curve.png')

torch.save(model.state_dict(), "CF.pth")
print("专项模型已保存。")