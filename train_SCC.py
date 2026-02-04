import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms

class ColorScoringDataset(Dataset):
    def __init__(self, img_dir, txt_dir, pixel_color_conf_json_path, image_score_json_path, max_objs=20):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.max_objs = max_objs
        
        # 加载 JSON 数据
        with open(pixel_color_conf_json_path, 'r') as f:
            self.pixel_color_conf_data = json.load(f) # 每种物体每个像素RGB置信度
        with open(image_score_json_path, 'r') as f:
            self.image_score_data = json.load(f) # 每张图片综合打分
            
        self.img_names = list(self.image_score_data.keys())
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        # 1. 加载图像
        img_path = os.path.join(self.img_dir, img_name + ".jpg") # 假设是jpg
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 2. 加载 YOLO 标签和颜色置信度
        txt_path = os.path.join(self.txt_dir, img_name + ".txt")
        obj_features = []
        
        # 确定颜色特征维度: R(256) + G(256) + B(256) = 768
        COLOR_FEAT_DIM = 3 * 256 

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                for line in lines[:self.max_objs]:
                    parts = line.strip().split()
                    cls_id_str = parts[0] # 类别ID作为字符串从txt读取
                    coords = [float(x) for x in parts[1:]] # x, y, w, h (归一化)
                    
                    # 从第一个JSON获取颜色特征 (3 * 256 维)
                    color_conf_vector = []
                    
                    # 尝试获取该类别ID的R/G/B通道置信度
                    obj_color_data = self.pixel_color_conf_data.get(cls_id_str, {})
                    
                    # 循环获取 R, G, B 通道的数据
                    for channel in ['r', 'g', 'b']:
                        channel_data = obj_color_data.get(channel, {})
                        # 构建256维向量，如果某个亮度值不存在则默认为0
                        channel_vec = [channel_data.get(str(i), 0.0) for i in range(256)]
                        color_conf_vector.extend(channel_vec)
                    
                    # 如果某个类别没有颜色数据，则填充零
                    if not color_conf_vector:
                        color_conf_vector = [0.0] * COLOR_FEAT_DIM
                    
                    # 拼接：类别(1) + 坐标(4) + 颜色置信度(768)
                    # 注意：这里我们将类别ID直接作为一个数值特征。更严谨的做法是使用Embedding层。
                    # 为了简化，这里直接作为数值，如果类别ID本身含义不大，建议使用Embedding。
                    feat = [float(cls_id_str)] + coords + color_conf_vector
                    obj_features.append(feat)

        # 3. 处理变长物体数量 (Padding)
        num_objs = len(obj_features)
        feat_dim = 1 + 4 + COLOR_FEAT_DIM # 类别ID + 坐标 + 颜色特征
        padded_objs = np.zeros((self.max_objs, feat_dim), dtype=np.float32)
        mask = np.ones(self.max_objs, dtype=bool) # True 表示被屏蔽(padding部分)
        
        if num_objs > 0:
            padded_objs[:num_objs] = np.array(obj_features, dtype=np.float32)
            mask[:num_objs] = False
            
        # 4. 获取标签分数
        label = torch.tensor([self.image_score_data[img_name]], dtype=torch.float32)
        
        return image, torch.from_numpy(padded_objs), torch.from_numpy(mask), label

class ColorAttentionNet(nn.Module):
    # obj_feat_dim = 1 (类别ID) + 4 (坐标) + 768 (颜色特征) = 773
    def __init__(self, obj_feat_dim=773, hidden_dim=256): 
        super(ColorAttentionNet, self).__init__()
        
        # 提取全局视觉特征
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # 输出 [B, 512, 1, 1]
        
        # 物体类别 Embedding (如果类别ID作为离散特征处理)
        # 这里假设有81个类别 (0-80)，你可以根据实际情况调整
        self.class_embedding = nn.Embedding(num_embeddings=81, embedding_dim=32)
        
        # 物体特征编码 (现在需要处理的维度是 4 (coords) + 768 (color) + 32 (class_embedding))
        self.obj_encoder = nn.Sequential(
            nn.Linear(4 + 768 + 32, hidden_dim), # 更新输入维度
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 注意力机制：根据物体特征计算其对总分的贡献权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # 最终回归层
        self.regressor = nn.Sequential(
            nn.Linear(512 + hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, img, obj_feats, mask):
        # 1. 图像特征
        v_img = self.backbone(img).view(img.size(0), -1) # [B, 512]
        
        # 2. 物体特征处理
        # 假设 obj_feats 的第一列是类别ID
        class_ids = obj_feats[:, :, 0].long() # 提取类别ID，转为long类型
        coords = obj_feats[:, :, 1:5] # 提取坐标
        color_confs = obj_feats[:, :, 5:] # 提取颜色置信度
        
        # 类别ID进行 Embedding
        class_emb = self.class_embedding(class_ids) # [B, Max_Objs, 32]
        
        # 拼接所有物体特征
        combined_obj_features = torch.cat([class_emb, coords, color_confs], dim=-1)
        
        v_obj = self.obj_encoder(combined_obj_features) # [B, Max_Objs, Hidden]
        
        # 3. 带 Mask 的注意力池化
        attn_weights = self.attention(v_obj).squeeze(-1) # [B, Max_Objs]
        
        # 将 padding 位置的注意力权重设为极小值，确保 softmax 后这些位置权重接近0
        attn_weights = attn_weights.masked_fill(mask, -1e9) 
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(-1) # [B, Max_Objs, 1]
        
        # 加权求和得到物体集合的综合表示
        v_obj_agg = torch.sum(v_obj * attn_weights, dim=1) # [B, Hidden]
        
        # 4. 融合预测
        combined = torch.cat([v_img, v_obj_agg], dim=1)
        return self.regressor(combined)

def train(train_dir, test_dir, txt_dir, pixel_color_conf_json_path, train_score_json_path, test_score_json_path):
    # 参数设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    epochs = 400
    
    train_dataset = ColorScoringDataset(
        train_dir, 
        txt_dir,
        pixel_color_conf_json_path,
        train_score_json_path
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ColorScoringDataset(
        test_dir, 
        txt_dir,
        pixel_color_conf_json_path,
        test_score_json_path
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # 注意这里 obj_feat_dim 已经更新为 773
    model = ColorAttentionNet(obj_feat_dim=773).to(device) 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    print(f"Starting training on device: {device}")

    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch_idx, (imgs, objs, masks, labels) in enumerate(train_dataloader):
            imgs, objs, masks, labels = imgs.to(device), objs.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs, objs, masks)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        total_test_loss = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, (imgs, objs, masks, labels) in enumerate(test_dataloader):
                imgs, objs, masks, labels = imgs.to(device), objs.to(device), masks.to(device), labels.to(device)
            
                outputs = model(imgs, objs, masks)
                total_test_loss += criterion(outputs, labels).item()
        
        avg_train_loss = total_loss/len(train_dataloader)
        avg_test_loss = total_test_loss/len(test_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss:.4f}, Average Test Loss: {avg_test_loss:.4f}")
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)

        if avg_train_loss <= 0.001:
            break

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('SCC_training_curve.png')

    torch.save(model.state_dict(), "SCC.pth")

if __name__ == "__main__":
    train(
        train_dir='ICQA_2K_train', 
        test_dir='ICQA_2K_test',
        txt_dir='labels',
        pixel_color_conf_json_path='color_.json',
        train_score_json_path='ICQA_2K_train_SCC_mos.json',
        test_score_json_path='ICQA_2K_test_SCC_mos.json'
    )