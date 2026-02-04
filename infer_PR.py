import os
import json
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

# --- 1. 模型架构定义 (必须与训练时保持完全一致) ---
class ColorQualityFusionNet(nn.Module):
    def __init__(self):
        super(ColorQualityFusionNet, self).__init__()
        # 全局自然度提取器
        res50 = models.resnet50(pretrained=False)
        self.naturalness_extractor = nn.Sequential(*list(res50.children())[:-1]) 
        
        # 分数处理层
        self.score_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # 融合回归头 (2048 + 16 = 2064)
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

# --- 2. 推理主函数 ---
def run_overall_inference(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # A. 加载预存的子项分数
    with open(config["hue_json"], 'r') as f:
        hue_results = json.load(f)
    with open(config["obj_json"], 'r') as f:
        obj_results = json.load(f)

    # B. 初始化并加载融合模型权重
    model = ColorQualityFusionNet().to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    # C. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    final_results = {}
    img_files = [f for f in os.listdir(config["img_dir"]) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Total images to process: {len(img_files)}")

    with torch.no_grad():
        for filename in tqdm(img_files):
            name = os.path.splitext(filename)[0]
            
            # 1. 获取子项分数 (若缺失则默认为0)
            s1 = hue_results.get(name, 0.0)
            s2 = obj_results.get(name, 0.0)
            sub_scores = torch.tensor([[s1, s2]], dtype=torch.float32).to(device)

            # 2. 处理图像
            img_path = os.path.join(config["img_dir"], filename)
            try:
                img_pil = Image.open(img_path).convert('RGB')
                img_tensor = transform(img_pil).unsqueeze(0).to(device)

                # 3. 模型预测
                prediction = model(img_tensor, sub_scores)
                score = prediction.item()
                
                final_results[name] = score
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # D. 保存最终结果
    with open(config["output_json"], 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    print(f"Inference complete. Results saved to {config['output_json']}")

# --- 3. 配置与运行 ---
if __name__ == "__main__":
    INFERENCE_CONFIG = {
        "img_dir": "./ICQA_2K_train/",                 # 测试图片文件夹
        "hue_json": "ICQA_2K_train_CF_result.json",   # 之前生成的污渍分数JSON
        "obj_json": "ICQA_2K_train_SCC_result.json",# 之前生成的物体分数JSON
        "model_path": "PR.pth",    # 训练好的融合模型权重
        "output_json": "ICQA_2K_train_PR_result.json"   # 最终输出结果
    }

    run_overall_inference(INFERENCE_CONFIG)