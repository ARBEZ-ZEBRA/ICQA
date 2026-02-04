import torch
import torch.nn as nn
from torchvision import models

class ColorQualityFusionNet(nn.Module):
    def __init__(self):
        super(ColorQualityFusionNet, self).__init__()
        
        res50 = models.resnet50(pretrained=True)
        self.naturalness_extractor = nn.Sequential(*list(res50.children())[:-1]) 
        
        self.score_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(2048 + 16, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512), 
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, img, sub_scores):
        feat_nat = self.naturalness_extractor(img).view(img.size(0), -1)
        
        feat_score = self.score_fc(sub_scores)
        
        combined = torch.cat([feat_nat, feat_score], dim=1)
        
        return self.regressor(combined)

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.optim as optim

class FusionDataset(Dataset):
    def __init__(self, img_dir, hue_json, obj_json, target_mos_json, transform=None):
        self.img_dir = img_dir
        
        # 加载三个数据源
        with open(hue_json, 'r') as f:
            self.hue_scores = json.load(f)  
        with open(obj_json, 'r') as f:
            self.obj_scores = json.load(f)  
        with open(target_mos_json, 'r') as f:
            self.targets = json.load(f)      

        self.img_names = list(self.targets.keys())
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        
        # 1. 加载图像
        img_path = os.path.join(self.img_dir, name + ".jpg")
        img = self.transform(Image.open(img_path).convert('RGB'))
        
        # 2. 获取预存的两个分数
        s1 = self.hue_scores.get(name, 0.0)
        s2 = self.obj_scores.get(name, 0.0)
        sub_scores = torch.tensor([s1, s2], dtype=torch.float32)
        
        # 3. 获取目标 MOS
        target = torch.tensor([self.targets[name]], dtype=torch.float32)
        
        return img, sub_scores, target

def train_fusion_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    CONFIG = {
        "train_img_dir": "./ICQA_2K_train/",
        "train_CF_json": "ICQA_2K_train_CF_result.json",
        "train_SCC_json": "ICQA_2K_train_SCC_result.json",
        "train_target_json": "ICQA_2K_train_PR_mos.json", 
        "test_img_dir": "./ICQA_2K_test/",
        "test_CF_json": "ICQA_2K_test_CF_result.json",
        "test_SCC_json": "ICQA_2K_test_SCC_result.json",
        "test_target_json": "ICQA_2K_test_PR_mos.json", 
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 500
    }

    # 数据准备
    train_dataset = FusionDataset(CONFIG["train_img_dir"], CONFIG["train_CF_json"], 
                             CONFIG["train_SCC_json"], CONFIG["train_target_json"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_dataset = FusionDataset(CONFIG["test_img_dir"], CONFIG["test_CF_json"], 
                             CONFIG["test_SCC_json"], CONFIG["test_target_json"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=True)

    # 模型初始化
    model = ColorQualityFusionNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.MSELoss()
    train_losses = []
    test_losses = []

    print("Starting training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        for imgs, scores, targets in train_loader:
            imgs, scores, targets = imgs.to(device), scores.to(device), targets.to(device)
            
            optimizer.zero_grad()
            preds = model(imgs, scores)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            
        train_losses.append(epoch_train_loss/len(train_loader))

        model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for imgs, scores, targets in test_loader:
                imgs, scores, targets = imgs.to(device), scores.to(device), targets.to(device)
            
                preds = model(imgs, scores)
                loss = criterion(preds, targets)
            
                epoch_test_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}], Train Loss: {epoch_train_loss/len(train_loader):.4f}, Test Loss: {epoch_test_loss/len(test_loader):.4f}")
        test_losses.append(epoch_test_loss/len(test_loader))

        if epoch_train_loss/len(train_loader) <= 0.001:
            break

    torch.save(model.state_dict(), "PR.pth")
    print("Model saved!")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('PR_training_curve.png')

if __name__ == "__main__":
    train_fusion_model()